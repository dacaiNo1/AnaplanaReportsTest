#!/usr/bin/env python3
# app.py — P&L Trend Analysis + Forecast (Upload / S3 / Anaplan Export)
# - Robust CSV/TSV loader (wide/long, odd headers, synonyms)
# - Trend vs Forecast windows (native cadence; no aggregation)
# - Per-line forecasting; single “Forecast” version CSV export (A–D preserved)
# - Export-from-Anaplan loader (FIX: download chunks with Accept: application/octet-stream)
# - Push-to-Anaplan import with Import/Server File dropdowns + time header conversion
# - S3 upload of generated forecast when source is “Load from S3”
# - S3 source now lists keys in a folder (prefix) and lets you select from a dropdown
# - TLS toggle (for corp proxies)
# - Correlation analysis fixed (handles alignment + fallback monthly)
# - Default trend/forecast-history end = 2024-03-30

import os
import re
import io
import math
import time
from io import BytesIO, StringIO
from typing import Optional, List, Tuple, Dict
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import boto3

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

st.set_page_config(page_title="P&L Trend Analysis + Forecast", layout="wide")

# --------------------------- Loader helpers ---------------------------
MONTH_FORMATS = [
    '%b %y', '%b %Y', '%B %y', '%B %Y',
    '%Y-%m', '%Y %m',
    '%y-%b', '%y %b', '%b-%y', '%B-%y'
]
MONTH_WORDS = {
    'jan','feb','mar','apr','may','jun','jul','aug','sep','sept','oct','nov','dec',
    'january','february','march','april','june','july','august','september','october','november','december'
}
LINE_ITEM_SYNS = [
    "Line Items","Line Item","Account","Account Name","Row Label","Report Row","Item","P&L Line Item","GL Account"
]
DATE_COL_SYNS = ["Date","Period","Month","Time","As Of"]
AMOUNT_COL_SYNS = ["Amount","Value","Amount (Base)","Amount LC","USD","CAD","Balance"]
FILTER_SYNS = {
    "Versions": ["Versions","Version","Scenario"],
    "Legal Entity": ["Legal Entity","Entity","Company","Company Name","Legal Entity L2 SS: Active"],
    "Department": ["Department","Dept","Department Name","Cost Center","Cost Center Name","Cost Center L3 SS: Active"],
    "Line of Business": ["Line of Business","LOB","Business Unit","Segment","Line of Business L2 SS: Active"],
    "Cost Center": ["Cost Center","Cost Center L2 SS: Active"]
}

def _to_bytes(file_like) -> bytes:
    if isinstance(file_like, BytesIO):
        return file_like.getvalue()
    if isinstance(file_like, StringIO):
        return file_like.getvalue().encode("utf-8", "ignore")
    if hasattr(file_like, "getvalue"):
        return file_like.getvalue()
    if isinstance(file_like, (str, os.PathLike)) and os.path.exists(file_like):
        with open(file_like, "rb") as f:
            return f.read()
    if hasattr(file_like, "read"):
        pos = file_like.tell() if hasattr(file_like, "tell") else None
        data = file_like.read()
        try:
            if pos is not None and hasattr(file_like, "seek"):
                file_like.seek(pos)
        except Exception:
            pass
        return data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8", "ignore")
    return b""

def _clean_header(s: str) -> str:
    return str(s).strip().strip("'").strip('"').replace("\u2013", "-").replace("\u2014", "-")

def _looks_like_month(token: str) -> bool:
    t = _clean_header(token)
    low = t.lower()
    if any(x in low for x in ["q1","q2","q3","q4","fy","ytd","total","variance"]):
        return False
    if any(m in low for m in MONTH_WORDS):
        return True
    t2 = t.replace("-", " ").replace("_", " ").strip()
    for fmt in MONTH_FORMATS:
        try:
            datetime.strptime(t2, fmt)
            return True
        except Exception:
            continue
    return False

def _detect_month_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if _looks_like_month(c)]

def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([_clean_header(str(x)) for x in tup if str(x) != "nan"]).strip()
                      for tup in df.columns.to_list()]
    return df

def _choose_header_row(sample_df: pd.DataFrame, max_scan: int = 6) -> Optional[int]:
    best_row, best_score = None, 0
    scan_rows = min(max_scan, len(sample_df))
    for i in range(scan_rows):
        row = sample_df.iloc[i].astype(str).tolist()
        score = sum(_looks_like_month(x) for x in row)
        if score > best_score:
            best_score, best_row = score, i
    return best_row if (best_row is not None and best_score >= 3) else None

def _find_first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns: return n
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def _infer_line_items_column(df: pd.DataFrame, id_vars: List[str]) -> str:
    cand = _find_first_present(df, LINE_ITEM_SYNS)
    if cand: return cand
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    date_like = {c for c in df.columns if any(w in c.lower() for w in ["date","period","month","time"])}
    for c in id_vars:
        if c in numeric_cols or c in date_like: continue
        if df[c].nunique(dropna=True) >= 2:
            return c
    return id_vars[0] if id_vars else df.columns[0]

def _clean_amount_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("\u2212","-", regex=False)
         .str.replace(",","", regex=False)
         .str.replace("$","", regex=False)
         .str.replace("%","", regex=False)
         .str.replace(r"\((.*)\)", r"-\1", regex=True)
         .replace({"—": np.nan, "": np.nan, "nan": np.nan})
         .pipe(pd.to_numeric, errors="coerce")
    )

def _guess_month_fmt_and_sep(period_cols: List[str]) -> Tuple[Optional[str], str]:
    if not period_cols:
        return None, " "
    first = str(period_cols[0])
    sep = "-" if "-" in first else ("_" if "_" in first else " ")
    cleaned = [str(c).replace("-", " ").replace("_", " ").strip() for c in period_cols]
    best_fmt, best_hits = None, -1
    for fmt in MONTH_FORMATS:
        hits = 0
        for c in cleaned:
            try:
                datetime.strptime(c, fmt); hits += 1
            except Exception:
                pass
        if hits > best_hits:
            best_hits, best_fmt = hits, fmt
    return best_fmt, sep

def _label_from_date(dt: pd.Timestamp, month_fmt: Optional[str], sep: str) -> str:
    fmt = month_fmt or "%b %y"
    lab = pd.to_datetime(dt).strftime(fmt)
    return lab.replace(" ", sep)

@st.cache_data
def load_and_melt(file_like) -> Optional[pd.DataFrame]:
    bytes_data = _to_bytes(file_like)
    if not bytes_data:
        st.error("Empty file or unreadable input.")
        return None

    def read_csv_clone(**kwargs):
        return pd.read_csv(BytesIO(bytes_data), low_memory=False, **kwargs)

    try:
        df0 = read_csv_clone(sep=None, engine="python")
    except Exception:
        df0 = read_csv_clone()

    df0.columns = [_clean_header(c) for c in df0.columns]
    df0 = _flatten_multiindex_columns(df0)

    # keep original first 4 headers (A–D)
    orig_first_cols = list(df0.columns[:4])

    if len(df0) > 0 and (set(df0.columns) & set(df0.iloc[0].astype(str).tolist())):
        df0 = df0.iloc[1:].reset_index(drop=True)

    period_cols = _detect_month_columns(df0.columns.tolist())

    if len(period_cols) < 3:
        raw = read_csv_clone(header=None, sep=None, engine="python")
        hdr_row = _choose_header_row(raw)
        if hdr_row is not None:
            df0 = read_csv_clone(header=hdr_row, sep=None, engine="python")
            df0.columns = [_clean_header(c) for c in df0.columns]
            df0 = _flatten_multiindex_columns(df0)
            period_cols = _detect_month_columns(df0.columns.tolist())

    if len(period_cols) < 3:
        try:
            df_tmp = read_csv_clone(header=[0,1], sep=None, engine="python")
            df_tmp = _flatten_multiindex_columns(df_tmp)
            df_tmp.columns = [_clean_header(c) for c in df_tmp.columns]
            pc2 = _detect_month_columns(df_tmp.columns.tolist())
            if len(pc2) >= 3:
                df0, period_cols = df_tmp, pc2
                orig_first_cols = list(df0.columns[:4])
        except Exception:
            pass

    long_date_col = _find_first_present(df0, DATE_COL_SYNS)
    long_amt_col  = _find_first_present(df0, AMOUNT_COL_SYNS)

    month_fmt, month_sep = _guess_month_fmt_and_sep(period_cols)

    meta = {
        "source_kind": "wide" if len(period_cols) >= 3 else "long",
        "period_cols": period_cols,
        "id_vars": [c for c in df0.columns if c not in period_cols] if len(period_cols) >= 3 else [],
        "month_fmt": month_fmt,
        "month_sep": month_sep,
        "long_date_col": long_date_col,
        "long_amt_col": long_amt_col,
        "orig_first_cols": orig_first_cols,
    }
    st.session_state["anaplan_meta"] = meta

    if len(period_cols) >= 3:
        id_vars = [c for c in df0.columns if c not in period_cols]
        li_col = _infer_line_items_column(df0, id_vars)
        if li_col != "Line Items":
            df0 = df0.rename(columns={li_col: "Line Items"})
            id_vars = [("Line Items" if c == li_col else c) for c in id_vars]

        m = df0.melt(id_vars=id_vars, value_vars=period_cols,
                     var_name="Period", value_name="Amount")
        clean_p = m["Period"].astype(str).str.replace("-", " ").str.replace("_", " ").str.strip()
        dt_series = None
        for fmt in MONTH_FORMATS:
            try:
                dt_series = pd.to_datetime(clean_p, format=fmt, errors='raise')
                break
            except Exception:
                continue
        m["Date"] = dt_series if isinstance(dt_series, pd.Series) else pd.to_datetime(clean_p, errors='coerce')
        m["Amount"] = _clean_amount_series(m["Amount"])
        return m

    if long_date_col and long_amt_col:
        df = df0.copy()
        df["Date"] = pd.to_datetime(df[long_date_col], errors="coerce")
        if df["Date"].isna().all():
            ok = False
            for fmt in MONTH_FORMATS:
                try:
                    df["Date"] = pd.to_datetime(df[long_date_col].astype(str).str.strip(), format=fmt, errors="raise")
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                df["Date"] = pd.to_datetime(df[long_date_col], errors="coerce")
        df["Amount"] = _clean_amount_series(df[long_amt_col])

        id_vars = [c for c in df.columns if c not in [long_date_col, long_amt_col, "Date"]]
        li_col = _infer_line_items_column(df, id_vars) if id_vars else "Line Items"
        if li_col == "Line Items" and "Line Items" not in df.columns:
            df["Line Items"] = "Total"
        elif li_col != "Line Items":
            df = df.rename(columns={li_col: "Line Items"})
        return df

    st.error("Could not detect monthly data. Ensure months are columns (e.g., 23-Jan / Jan 23) or there is a Date column.")
    return None

# --------------------------- Frequency inference ---------------------------
def infer_freq_info(idx: pd.DatetimeIndex) -> Tuple[str, str, Optional[int], str]:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx = idx.dropna().sort_values().unique()
    inferred = None
    if len(idx) >= 3:
        try:
            inferred = pd.infer_freq(idx)
        except Exception:
            inferred = None
    label = "native"
    if not inferred:
        if len(idx) >= 2:
            gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
            med = int(np.median(gaps))
        else:
            med = 30
        if 27 <= med <= 31:
            inferred, label = "MS", "monthly"
        elif 6 <= med <= 8:
            inferred, label = "W", "weekly"
        elif 89 <= med <= 92:
            inferred, label = "Q", "quarterly"
        elif 364 <= med <= 366:
            inferred, label = "A", "annual"
        else:
            inferred, label = "D", "daily"
    else:
        up = inferred.upper()
        if up.startswith("M"): inferred, label = "MS", "monthly"
        elif up.startswith("W"): inferred, label = "W", "weekly"
        elif up.startswith("Q"): inferred, label = "Q", "quarterly"
        elif up.startswith("A") or up.startswith("Y"): inferred, label = "A", "annual"
        elif up.startswith("D"): inferred, label = "D", "daily"
    if inferred in ("M", "MS"):
        prophet_freq, sp = "MS", 12
    elif inferred and inferred.startswith("W"):
        prophet_freq, sp = "W", 52
    elif inferred and inferred.startswith("Q"):
        prophet_freq, sp = "Q", 4
    elif inferred and (inferred.startswith("A") or inferred.startswith("Y")):
        prophet_freq, sp = "A", 1
    elif inferred == "D":
        prophet_freq, sp = "D", 7
    else:
        prophet_freq, sp = inferred or "MS", None
    return inferred or "MS", prophet_freq, sp, label

# --------------------------- TLS / FRED / Anaplan helpers ---------------------------
def get_verify_flag() -> bool:
    return not st.session_state.get("INSECURE_HTTPS", False)

def requests_kwargs() -> Dict:
    verify = get_verify_flag()
    kwargs: Dict = {"timeout": 30, "verify": verify}
    if not verify:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    return kwargs

def _get_fred_key():
    general = (st.secrets.get("general", {}) if hasattr(st, "secrets") else {}) or {}
    key = general.get("FRED_API_KEY") or os.getenv("FRED_API_KEY")
    return key, None

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def _fred_observations(series_id: str, start: datetime, end: datetime, api_key: str) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": pd.to_datetime(start).strftime("%Y-%m-%d"),
        "observation_end": pd.to_datetime(end).strftime("%Y-%m-%d"),
    }
    r = requests.get(FRED_BASE, params=params, **requests_kwargs())
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)
    dates = pd.to_datetime([o["date"] for o in obs])
    vals = []
    for o in obs:
        v = o.get("value", ".")
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    s = pd.Series(vals, index=dates).dropna()
    return s.resample("M").last().dropna()

@st.cache_data
def fetch_macro_data(start: datetime, end: datetime) -> pd.DataFrame:
    start = pd.to_datetime(start).floor("D")
    end = min(pd.to_datetime(end).floor("D"), pd.Timestamp.today().floor("D"))
    key, _ = _get_fred_key()
    if not key:
        return pd.DataFrame()
    series_map = {
        "CPIAUCSL": "Inflation (CPI)",
        "UNRATE": "Unemployment Rate",
        "FEDFUNDS": "Fed Funds Rate",
        "GS10": "10Y Treasury Yield",
        "INDPRO": "Industrial Production",
        "M2SL": "Money Supply (M2)",
        "UMCSENT": "Consumer Sentiment",
        "PPIACO": "PPI – Finished Goods",
        "NAPM": "ISM Manufacturing PMI",
        "DCOILWTICO": "Crude Oil Price (WTI)",
        "JTSJOR": "Job Openings (JOLTS)",
        "RSAFS": "Retail Sales (Total)",
        "TCU": "Capacity Utilization",
    }
    frames = []
    last_err = None
    for code, name in series_map.items():
        try:
            s = _fred_observations(code, start, end, key)
            if not s.empty:
                frames.append(s.rename(name))
        except Exception as e:
            last_err = e
            continue
    if not frames:
        if last_err:
            st.session_state["_last_fred_error"] = repr(last_err)
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()

# -------- Anaplan HTTP helpers --------
AUTH_URL = "https://auth.anaplan.com/token/authenticate"
BASE = "https://api.anaplan.com/2/0"

def _api_headers(token: str, content_type: Optional[str] = None, accept: str = "application/json"):
    h = {"Authorization": f"AnaplanAuthToken {token}", "Accept": accept}
    if content_type: h["Content-Type"] = content_type
    return h

def get_token(username: str, password: str) -> str:
    r = requests.post(AUTH_URL, auth=(username, password), **requests_kwargs())
    r.raise_for_status()
    return r.json()["tokenInfo"]["tokenValue"]

def list_exports(token: str, wid: str, mid: str) -> list[dict]:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/exports"
    r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
    r.raise_for_status()
    return r.json().get("exports", [])

def run_export(token: str, wid: str, mid: str, export_id: str, locale: str = "en_US") -> str:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/exports/{export_id}/tasks"
    r = requests.post(url, headers=_api_headers(token, "application/json"),
                      json={"localeName": locale}, **requests_kwargs())
    r.raise_for_status()
    return r.json()["task"]["taskId"]

def wait_for_export_task(token: str, wid: str, mid: str, export_id: str, task_id: str,
                         poll_seconds: int = 2, timeout_seconds: int = 600) -> None:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/exports/{export_id}/tasks/{task_id}"
    start = time.time()
    while True:
        r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
        r.raise_for_status()
        task = r.json()["task"]
        if task.get("taskState") == "COMPLETE":
            if task.get("result", {}).get("successful", False):
                return
            raise RuntimeError(f"Export finished but failed: {task}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for export to finish.")
        time.sleep(poll_seconds)

def list_chunks(token: str, wid: str, mid: str, file_id: str) -> List[str]:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/files/{file_id}/chunks/"
    r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
    r.raise_for_status()
    chunks = r.json().get("chunks", [])
    return [c["id"] for c in chunks] if chunks else ["0"]

def get_chunk(token: str, wid: str, mid: str, file_id: str, chunk_id: str) -> bytes:
    # FIX: Export file chunks are binary; request with Accept: application/octet-stream
    url = f"{BASE}/workspaces/{wid}/models/{mid}/files/{file_id}/chunks/{chunk_id}"
    r = requests.get(url, headers=_api_headers(token, accept="application/octet-stream"), **requests_kwargs())
    r.raise_for_status()
    return r.content

def list_imports(token: str, wid: str, mid: str) -> list[dict]:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/imports"
    r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
    r.raise_for_status()
    return r.json().get("imports", [])

def get_import_details(token: str, wid: str, mid: str, import_id: str) -> dict:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/imports/{import_id}"
    r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
    r.raise_for_status()
    return r.json().get("import", {})

def get_import_source(token: str, wid: str, mid: str, import_id: str) -> dict | None:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/imports/{import_id}/source"
    r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def list_files(token: str, wid: str, mid: str) -> list[dict]:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/files"
    r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
    r.raise_for_status()
    return r.json().get("files", [])

def resolve_file_id_for_import(token: str, wid: str, mid: str, import_id: str) -> tuple[Optional[str], Optional[str]]:
    try:
        src = get_import_source(token, wid, mid, import_id)
    except Exception:
        src = None
    if src:
        fid = str(src.get("fileId") or src.get("id") or "").strip() or None
        fname = (src.get("name") or src.get("fileName") or None)
        if fid:
            return fid, fname
    try:
        imp = get_import_details(token, wid, mid, import_id)
    except Exception:
        imp = {}
    possible_names = [
        (imp.get("source", {}) or {}).get("name"),
        imp.get("sourceFileName"),
        imp.get("name"),
    ]
    possible_names = [str(x).strip() for x in possible_names if x]
    try:
        files = list_files(token, wid, mid)
    except Exception:
        files = []
    by_name = {f.get("name"): f for f in files if f.get("name")}
    for nm in possible_names:
        if nm in by_name:
            return by_name[nm].get("id"), nm
    if len(files) == 1:
        return files[0].get("id"), files[0].get("name")
    return None, None

def run_import(token: str, wid: str, mid: str, import_id: str, locale: str = "en_US") -> str:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/imports/{import_id}/tasks"
    r = requests.post(url, headers=_api_headers(token, "application/json"),
                      json={"localeName": locale}, **requests_kwargs())
    r.raise_for_status()
    return r.json()["task"]["taskId"]

def wait_for_import_task(token: str, wid: str, mid: str, import_id: str, task_id: str,
                         poll_seconds: int = 2, timeout_seconds: int = 600) -> Dict:
    url = f"{BASE}/workspaces/{wid}/models/{mid}/imports/{import_id}/tasks/{task_id}"
    start = time.time()
    while True:
        r = requests.get(url, headers=_api_headers(token), **requests_kwargs())
        r.raise_for_status()
        task = r.json()["task"]
        if task.get("taskState") == "COMPLETE":
            return task
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for import to finish.")
        time.sleep(poll_seconds)

# --------------------------- Time-header conversion helpers ---------------------------
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MON2NUM = {m: i+1 for i, m in enumerate(MONTH_ABBR)}

def _parse_period_hdr(s: str):
    s = str(s).strip()
    m = re.fullmatch(r"(\d{2})[-_/ ]([A-Za-z]{3})", s)  # 23-Jan
    if m and m.group(2).title() in MON2NUM:
        return (2000 + int(m.group(1))), MON2NUM[m.group(2).title()]
    m = re.fullmatch(r"([A-Za-z]{3})[-_/ ](\d{2})", s)  # Jan-23 / Jan 23
    if m and m.group(1).title() in MON2NUM:
        return (2000 + int(m.group(2))), MON2NUM[m.group(1).title()]
    m = re.fullmatch(r"(20\d{2})[-_/](0?[1-9]|1[0-2])", s)  # 2023-01
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(0?[1-9]|1[0-2])[-_/](20\d{2})", s)  # 01/2023
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.fullmatch(r"([A-Za-z]{3})\s+(20\d{2})", s)  # Jan 2023
    if m and m.group(1).title() in MON2NUM:
        return int(m.group(2)), MON2NUM[m.group(1).title()]
    return None

def _fmt_period_hdr(yyyy: int, mm: int, target: str) -> str:
    mon = MONTH_ABBR[mm-1]
    yy = str(yyyy)[-2:]
    if target == "KEEP":     return f"{mon} {yy}"
    if target == "MMM YY":   return f"{mon} {yy}"
    if target == "MMM-YY":   return f"{mon}-{yy}"
    if target == "YY-MMM":   return f"{yy}-{mon}"
    if target == "MMM YYYY": return f"{mon} {yyyy}"
    if target == "YYYY-MM":  return f"{yyyy}-{mm:02d}"
    if target == "MM/YYYY":  return f"{mm:02d}/{yyyy}"
    return f"{mon} {yy}"

def convert_time_headers(csv_bytes: bytes, target_format: str) -> bytes:
    if target_format == "KEEP":
        return csv_bytes
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str)
    out_cols = []
    for c in df.columns:
        parsed = _parse_period_hdr(c)
        out_cols.append(_fmt_period_hdr(*parsed, target_format) if parsed else c)
    df.columns = out_cols
    out = io.BytesIO()
    df.to_csv(out, index=False)
    return out.getvalue()

# --------------------------- S3 upload helper ---------------------------
def upload_forecast_csv_to_s3(
    csv_bytes: bytes,
    bucket: str,
    region: str | None = None,
    prefix: str = "",
    filename: str = "anaplan_forecast_version_all_lines.csv",
    verify_tls: bool | None = None,
) -> str:
    if not csv_bytes:
        raise ValueError("No CSV bytes provided.")
    key = f"{prefix.strip('/')}/{filename}" if prefix.strip() else filename

    def _sec(k, default=None):
        try: return st.secrets.get(k, default)
        except Exception: return default

    region = region or _sec("aws_region")
    aws_access_key = _sec("aws_access_key")
    aws_secret_key = _sec("aws_secret_key")

    client_kwargs = dict(region_name=region)
    if aws_access_key and aws_secret_key:
        client_kwargs.update(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
    if verify_tls is False:
        client_kwargs["verify"] = False

    s3 = boto3.client("s3", **client_kwargs)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_bytes,
        ContentType="text/csv",
        CacheControl="no-cache",
        ACL="private",
        ServerSideEncryption="AES256",
    )
    return f"s3://{bucket}/{key}"

# --------------------------- Forecast utilities ---------------------------
def fit_and_forecast_monthly(s: pd.Series, method: str, horizon: int,
                             roll_window: int = 3, arima_order: Tuple[int,int,int]=(1,1,0)) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    last = s.index.max()
    start = (last + pd.offsets.MonthBegin(1))
    if method == "Linear Trend":
        x = np.arange(len(s)); m_, b_ = np.polyfit(x, s.values, 1)
        future_x = np.arange(len(s), len(s) + horizon)
        fut = pd.Series(m_ * future_x + b_, index=pd.date_range(start, periods=horizon, freq="MS"))
        return fut
    if method == "Rolling Average":
        w = max(1, min(int(roll_window), len(s)))
        avg = s.rolling(w).mean().iloc[-1]
        return pd.Series(np.repeat(avg, horizon), index=pd.date_range(start, periods=horizon, freq="MS"))
    if method == "ARIMA":
        p,d,q = arima_order
        try:
            mod = ARIMA(s, order=(int(p), int(d), int(q))).fit()
            fut = mod.forecast(steps=horizon)
            return pd.Series(fut.values, index=fut.index)
        except Exception:
            pass
    if method == "Prophet" and HAS_PROPHET:
        dfp = s.reset_index(); dfp.columns = ["ds","y"]
        mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        mp.fit(dfp)
        fut = mp.make_future_dataframe(periods=horizon, freq="MS")
        pf = mp.predict(fut).set_index("ds")["yhat"].iloc[-horizon:]
        return pf
    try:
        hw = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12).fit()
        fut = hw.forecast(horizon)
        return pd.Series(fut.values, index=fut.index)
    except Exception:
        return pd.Series(np.repeat(s.values[-1], horizon), index=pd.date_range(start, periods=horizon, freq="MS"))

def to_month_start(dt_like) -> pd.Timestamp:
    ts = pd.to_datetime(dt_like)
    return ts.to_period("M").to_timestamp()

def month_start_after(dt_like) -> pd.Timestamp:
    return to_month_start(dt_like) + pd.offsets.MonthBegin(1)

def months_between(ms_start: pd.Timestamp, ms_end: pd.Timestamp) -> int:
    return (ms_end.year - ms_start.year) * 12 + (ms_end.month - ms_start.month) + 1

def _default_end_within(min_d: date, max_d: date, desired: date) -> date:
    if min_d <= desired <= max_d:
        return desired
    return max_d

def _na_eq_mask(series: pd.Series, val) -> pd.Series:
    if pd.isna(val):
        return series.isna()
    return series == val

# --------------------------- Sidebar Filters ---------------------------
def pick_side_filters(data: pd.DataFrame) -> pd.DataFrame:
    vcol = _find_first_present(data, FILTER_SYNS["Versions"])
    if vcol:
        versions = sorted(data[vcol].dropna().unique())
        if len(versions):
            sel_version = st.sidebar.selectbox("Version", versions, key="flt_ver")
            data = data[data[vcol] == sel_version]
    for label_key in ["Line of Business", "Legal Entity", "Department", "Cost Center"]:
        syns = FILTER_SYNS[label_key]
        col = _find_first_present(data, syns)
        if col:
            opts = sorted(data[col].dropna().unique())
            if len(opts):
                selected = st.sidebar.multiselect(label_key, opts, default=opts, key=f"flt_{label_key}")
                data = data[data[col].isin(selected)]
    return data

# --------------------------- Forecast cache helpers ---------------------------
def save_forecast_cache(payload: Dict):
    st.session_state["forecast_cache"] = payload

def get_forecast_cache() -> Optional[Dict]:
    return st.session_state.get("forecast_cache")

def clear_forecast_cache():
    st.session_state.pop("forecast_cache", None)

# --------------------------- Main ---------------------------
def main():
    st.title("📊 P&L Trend Analysis + Forecast")

    # Network & security
    with st.sidebar.expander("Network & security"):
        insecure = st.checkbox("Allow insecure HTTPS (NOT recommended)", value=st.session_state.get("INSECURE_HTTPS", False), key="tls_insecure")
        st.session_state["INSECURE_HTTPS"] = insecure
        st.caption("Only enable if a corporate proxy breaks TLS certificates.")

    # 1) Source
    st.markdown("### Step 1: Choose File Source")
    source = st.radio("File Source:", ["Manual Upload", "Load from S3", "Export from Anaplan"], horizontal=True, key="src_mode")
    st.session_state["source_mode"] = source

    if source == "Manual Upload":
        uploaded = st.file_uploader("Upload P&L CSV/TSV", type=["csv", "tsv"], key="upl_csv")
        if uploaded:
            st.session_state.buffer = uploaded
            st.session_state.pop("anaplan_creds", None)

    elif source == "Load from S3":
        # ---- NEW: list keys under a prefix and pick from a dropdown ----
        def _sec(k, default=""):
            try: return st.secrets.get(k, default)
            except Exception: return default

        with st.form("s3_pick_form", clear_on_submit=False):
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                bucket = st.text_input("S3 Bucket Name", value=_sec("s3_bucket", "your-bucket"), key="s3_bucket")
                prefix = st.text_input("Folder (prefix)", value=_sec("s3_source_prefix", ""), help="e.g., finance/exports/", key="s3_prefix")
            with c2:
                region = st.text_input("AWS Region", value=_sec("aws_region", "us-east-1"), key="s3_region")
                list_btn = st.form_submit_button("🔄 List files", use_container_width=True)
            with c3:
                st.caption("Only .csv / .tsv shown")

            # list keys (persist in session)
            if list_btn:
                try:
                    client_kwargs = dict(region_name=region or None)
                    ak = _sec("aws_access_key", None); sk = _sec("aws_secret_key", None)
                    if ak and sk:
                        client_kwargs.update(aws_access_key_id=ak, aws_secret_access_key=sk)
                    if not get_verify_flag():
                        client_kwargs["verify"] = False
                    s3 = boto3.client('s3', **client_kwargs)

                    paginator = s3.get_paginator("list_objects_v2")
                    pages = paginator.paginate(Bucket=bucket, Prefix=prefix or "")
                    found: list[str] = []
                    for page in pages:
                        for obj in page.get("Contents", []) or []:
                            key = obj.get("Key")
                            if key and key.lower().endswith((".csv", ".tsv")):
                                found.append(key)
                    st.session_state["s3_keys_list"] = sorted(found)
                    st.success(f"Found {len(found)} file(s) under '{prefix}'.")
                except Exception as e:
                    st.error(f"Failed to list objects: {e}")

            keys = st.session_state.get("s3_keys_list", [])
            selected_key = None
            if keys:
                selected_key = st.selectbox("S3 File Key", options=keys, index=0, key="s3_key_select")
            else:
                # fallback: allow manual key entry if listing not used yet
                selected_key = st.text_input("S3 File Key", value=_sec("s3_key", ""), key="s3_key")

            load_btn = st.form_submit_button("Load CSV from S3", use_container_width=True)

        if load_btn:
            try:
                client_kwargs = dict(region_name=st.session_state.get("s3_region") or _sec("aws_region", None))
                ak = _sec("aws_access_key", None); sk = _sec("aws_secret_key", None)
                if ak and sk:
                    client_kwargs.update(aws_access_key_id=ak, aws_secret_access_key=sk)
                if not get_verify_flag():
                    client_kwargs["verify"] = False
                s3 = boto3.client('s3', **client_kwargs)
                resp = s3.get_object(Bucket=st.session_state.get("s3_bucket") or bucket, Key=selected_key)
                content = resp['Body'].read()
                st.session_state.buffer = BytesIO(content)
                st.success(f"Loaded from S3: {selected_key}")
                st.session_state.pop("anaplan_creds", None)
            except Exception as e:
                st.error(f"Failed to load from S3: {e}")

    else:  # Export from Anaplan
        st.markdown("#### Connect to Anaplan & Run Export")
        with st.form("anaplan_export_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                email = st.text_input("Anaplan Email", value="hshan@lionpointgroup.com", key="ana_email")
                wid = st.text_input("Workspace ID", value="8a81b08e575246890157b4fc78560a18", key="ana_wid")
            with c2:
                password = st.text_input("Password", type="password", key="ana_pwd")
                mid = st.text_input("Model ID", value="8B59897CA9AF46A18CD858D27FE483B4", key="ana_mid")

            exp_col = st.columns([2,1,2])
            with exp_col[0]:
                refresh_exports = st.form_submit_button("🔄 Refresh export actions")
            with exp_col[2]:
                locale = st.text_input("Locale (optional)", value="en_US", key="ana_locale")

            exports_list = st.session_state.get("ana_exports", [])
            if refresh_exports:
                try:
                    token_dbg = get_token(email, password)
                    exports_list = list_exports(token_dbg, wid, mid)
                    st.session_state["ana_exports"] = exports_list
                    st.success(f"Loaded {len(exports_list)} export action(s).")
                except Exception as e:
                    st.error(f"Failed to list exports: {type(e).__name__}: {e}")

            selected_export_id = None
            if exports_list:
                labels = [f"{e.get('name','(unnamed)')} — {e.get('id')}" for e in exports_list]
                ids = [e.get("id") for e in exports_list]
                idx = 0
                if "ana_export_selected_id" in st.session_state and st.session_state["ana_export_selected_id"] in ids:
                    idx = ids.index(st.session_state["ana_export_selected_id"])
                sel = st.selectbox("Export Action", labels, index=idx, key="ana_export_sel_label")
                selected_export_id = ids[labels.index(sel)]
                st.session_state["ana_export_selected_id"] = selected_export_id
            else:
                st.info("No export actions loaded yet. Click **Refresh export actions** after entering credentials/WID/MID.")

            run_export_btn = st.form_submit_button("Run Export & Load", use_container_width=True)

        if run_export_btn:
            if not all([email, password, wid, mid, selected_export_id]):
                st.error("Please fill in Email, Password, Workspace ID, Model ID, and choose an Export Action.")
            else:
                with st.spinner("Exporting from Anaplan…"):
                    try:
                        token_dbg = get_token(email, password)
                        task_id = run_export(token_dbg, wid, mid, selected_export_id, locale=locale)
                        wait_for_export_task(token_dbg, wid, mid, selected_export_id, task_id)

                        file_id = None
                        for e in st.session_state.get("ana_exports", []):
                            if e.get("id") == selected_export_id:
                                file_id = e.get("fileId") or selected_export_id
                                break
                        file_id = file_id or selected_export_id
                        chunk_ids = list_chunks(token_dbg, wid, mid, file_id)
                        buf_dl = BytesIO()
                        for cid in chunk_ids:
                            buf_dl.write(get_chunk(token_dbg, wid, mid, file_id, cid))
                        st.session_state.buffer = BytesIO(buf_dl.getvalue())
                        st.success("Export completed and loaded!")
                        st.session_state["anaplan_creds"] = {
                            "email": email, "password": password,
                            "wid": wid, "mid": mid, "locale": locale
                        }
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {e}")

    buf = st.session_state.get("buffer", None)
    if buf is None:
        st.info("Please upload or load a file to proceed.")
        return

    # 2) Load
    data_all = load_and_melt(buf)
    if data_all is None or data_all.empty:
        return

    # 3) Filters
    st.sidebar.header("Data Filters")
    data_all = pick_side_filters(data_all)

    st.subheader("Filtered Data Preview")
    st.dataframe(data_all, use_container_width=True)

    if "Line Items" not in data_all.columns:
        st.error("The dataset must include or infer a 'Line Items' column for analysis.")
        return

    # Selection (charts only; export includes ALL)
    items = sorted(data_all["Line Items"].dropna().unique())
    sel_for_charts = st.multiselect("Select Line Items for CHARTS ONLY (export includes ALL line items)",
                                    items, default=items[:3], key="chart_lines")
    if not sel_for_charts:
        sel_for_charts = items[:3]

    # ========== TREND ==========
    st.markdown("## 📈 Trend Analysis")
    min_d, max_d = data_all["Date"].min().date(), data_all["Date"].max().date()
    desired_end = date(2024, 3, 30)
    trend_end_default = _default_end_within(min_d, max_d, desired_end)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        trend_start = st.date_input("Trend Start Date", min_value=min_d, max_value=max_d, value=min_d, key="trend_start")
    with c2:
        trend_end = st.date_input("Trend End Date", min_value=min_d, max_value=max_d, value=trend_end_default, key="trend_end")
    with c3:
        trend_chart = st.selectbox("Trend Chart Type", ["Line", "Area", "Bar"], key="trend_chart")
    if trend_start > trend_end:
        st.error("Trend start date must be on or before trend end date.")
        return

    trend_data = data_all[(data_all["Date"] >= pd.to_datetime(trend_start)) &
                          (data_all["Date"] <= pd.to_datetime(trend_end))]
    trend_df = trend_data[trend_data["Line Items"].isin(sel_for_charts)]
    pivot_trend = pd.DataFrame()
    if not trend_df.empty:
        pivot_trend = trend_df.pivot_table(index="Date", columns="Line Items", values="Amount", aggfunc="sum").sort_index()
        pct_change_table = pivot_trend.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100
        st.subheader("Period-over-period % Change (Trend)")
        st.dataframe(pct_change_table.style.format("{:+.2f}%"), use_container_width=True)
        st.subheader(f"{trend_chart} Chart — Trend Window")
        if trend_chart == "Line":
            fig_t = px.line(pivot_trend, x=pivot_trend.index, y=pivot_trend.columns, labels={"value": "Amount","Date": "Date"})
        elif trend_chart == "Area":
            fig_t = px.area(pivot_trend, x=pivot_trend.index, y=pivot_trend.columns, labels={"value": "Amount","Date": "Date"})
        else:
            dfm = pivot_trend.reset_index().melt(id_vars="Date", var_name="Line Items", value_name="Amount")
            fig_t = px.bar(dfm, x="Date", y="Amount", color="Line Items", barmode="group")
        fig_t.update_layout(hovermode="x unified", yaxis_tickformat=",")
        st.plotly_chart(fig_t, use_container_width=True)

    # ---- Trend correlations vs Macro ----
    st.markdown("### 🔍 Trend Analysis & Detailed Narratives — Macro correlation")
    corr_mode = st.radio("Correlation mode", ["Levels", "Period-over-period % change"], horizontal=True, key="corr_mode")
    macro = fetch_macro_data(pd.to_datetime(trend_start), pd.to_datetime(trend_end))
    if macro.empty or pivot_trend.empty:
        st.info("No macro or P&L data for the Trend window.")
    else:
        macro_first = min([s.first_valid_index() for _, s in macro.items() if s.first_valid_index() is not None] or [None])
        macro_last  = max([s.last_valid_index()  for _, s in macro.items() if s.last_valid_index()  is not None] or [None])
        trend_first, trend_last = pivot_trend.index.min(), pivot_trend.index.max()
        if macro_first is not None and macro_last is not None:
            overlap_start = max(trend_first, macro_first)
            overlap_end   = min(trend_last, macro_last)
        else:
            overlap_start = overlap_end = None

        def compute_corr(pvt: pd.DataFrame, mac: pd.DataFrame, mode: str):
            if overlap_start is not None and overlap_end is not None and overlap_start <= overlap_end:
                pvt = pvt.loc[(pvt.index >= overlap_start) & (pvt.index <= overlap_end)]
                mac = mac.loc[(mac.index >= overlap_start) & (mac.index <= overlap_end)]
            mac_aligned = mac.reindex(pvt.index).ffill()
            if mode == "Period-over-period % change":
                pvt_use = pvt.pct_change(); mac_use = mac_aligned.pct_change()
            else:
                pvt_use = pvt.copy(); mac_use = mac_aligned.copy()
            pvt_use = pvt_use.dropna(how="all"); mac_use = mac_use.dropna(how="all")
            common = pvt_use.index.intersection(mac_use.index)
            pvt_use, mac_use = pvt_use.loc[common], mac_use.loc[common]
            if len(common) < 2:
                return None
            corr_table = pd.DataFrame(index=mac_use.columns, columns=pvt_use.columns, dtype=float)
            for li in pvt_use.columns:
                cors = mac_use.corrwith(pvt_use[li], axis=0).dropna()
                corr_table[li] = cors
            return corr_table.dropna(how="all")

        corr_table = compute_corr(pivot_trend, macro, corr_mode)
        used_fallback = False
        if corr_table is None or corr_table.empty:
            pivot_m = pivot_trend.resample("M").sum(min_count=1)
            macro_m = macro.resample("M").last()
            corr_table = compute_corr(pivot_m, macro_m, corr_mode)
            used_fallback = corr_table is not None and not corr_table.empty

        if corr_table is None or corr_table.empty:
            st.info("Insufficient overlap between Trend window data and Macro series to compute correlation.")
        else:
            if used_fallback: st.caption("Correlation computed on monthly-aligned data (fallback).")
            left, right = st.columns([1,2])
            with left:
                st.subheader("Top drivers")
                for li in corr_table.columns:
                    cors = corr_table[li].dropna()
                    if cors.empty:
                        st.caption(f"• **{li}** — no overlapping macro data")
                        continue
                    top3 = cors.reindex(cors.abs().sort_values(ascending=False).head(3).index)
                    st.markdown(f"**{li}**")
                    for name, val in top3.items():
                        sign = "+" if val > 0 else ""
                        st.markdown(f"- {name}: **{sign}{val:.2f}** {'(together)' if val>0 else '(inverse)'}")
                st.subheader("Line item movement (Trend window)")
                for item in sel_for_charts:
                    if item not in pivot_trend.columns: continue
                    s = pivot_trend[item].dropna()
                    if s.empty: continue
                    sv, ev = s.iloc[0], s.iloc[-1]
                    change_pct = (ev - sv) / (abs(sv) if sv != 0 else 1) * 100
                    direction = "increase" if change_pct >= 0 else "decrease"
                    d1 = s.index[0].strftime("%m/%d/%Y"); d2 = s.index[-1].strftime("%m/%d/%Y")
                    st.markdown(f"• **{item}**: {direction} **{change_pct:+.2f}%** (from {sv:,.0f} on {d1} to {ev:,.0f} on {d2})")
            with right:
                st.subheader("Correlation heatmap (macro × line items)")
                fig_hm = px.imshow(corr_table, text_auto=".2f", zmin=-1, zmax=1,
                                   aspect="auto", labels=dict(x="Line Item", y="Macro Series", color="Corr"))
                fig_hm.update_layout(margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_hm, use_container_width=True)

    # Macro summary (Trend window)
    macro_sum = fetch_macro_data(pd.to_datetime(trend_start), pd.to_datetime(trend_end))
    st.subheader("📈 Macro & Micro Context (Trend window)")
    if macro_sum.empty:
        st.info("No macro data available. Check FRED_API_KEY / dates.")
    else:
        def first_valid(x):
            x = x.dropna(); return x.iloc[0] if not x.empty else np.nan
        def last_valid(x):
            x = x.dropna(); return x.iloc[-1] if not x.empty else np.nan
        ms = macro_sum.apply(first_valid, axis=0); me = macro_sum.apply(last_valid, axis=0)
        def pct(a, b):
            if pd.isna(a) or pd.isna(b) or a == 0: return np.nan
            return (b - a) / abs(a) * 100
        for name in macro_sum.columns:
            a, b = ms[name], me[name]
            if pd.isna(a) or pd.isna(b): continue
            delta = pct(a, b)
            trend_lbl = "up" if b > a else "down" if b < a else "flat"
            st.markdown(f"• **{name}**: {trend_lbl}, from **{a:,.2f}** to **{b:,.2f}** ({delta:+.2f}%)")

    # ========== FORECAST ==========
    st.markdown("## 🔮 Forecast Analysis")
    desired_end = date(2024, 3, 30)
    min_d, max_d = data_all["Date"].min().date(), data_all["Date"].max().date()
    fc_end_default = _default_end_within(min_d, max_d, desired_end)

    f1, f2, f3, f4 = st.columns([1,1,1,1])
    with f1:
        fc_hist_start = st.date_input("Forecast History Start", min_value=min_d, max_value=max_d, value=min_d, key="fc_hist_start")
    with f2:
        fc_hist_end = st.date_input("Forecast History End", min_value=min_d, max_value=max_d, value=fc_end_default, key="fc_hist_end")
    with f3:
        forecast_method = st.selectbox("Method", ["None", "Linear Trend", "Rolling Average", "ARIMA", "Prophet", "Holt-Winters"], key="fc_method")
    with f4:
        horizon = st.slider("Forecast Horizon (months)", 1, 24, 6, key="fc_horizon")
    c_arima, c_roll = st.columns([1,1])
    with c_roll:
        roll_window = st.slider("Rolling window", 1, 24, 3, key="fc_roll") if forecast_method == "Rolling Average" else 3
    with c_arima:
        if forecast_method == "ARIMA":
            arima_p = st.number_input("ARIMA p", 0, 5, 1, key="fc_p")
            arima_d = st.number_input("ARIMA d", 0, 2, 1, key="fc_d")
            arima_q = st.number_input("ARIMA q", 0, 5, 0, key="fc_q")
        else:
            arima_p, arima_d, arima_q = 1, 1, 0

    run_forecast_clicked = st.button("Run Forecast", key="btn_run_fc")

    if fc_hist_start > fc_hist_end:
        st.error("Forecast history start must be on or before forecast history end.")
        return

    if run_forecast_clicked:
        if forecast_method == "None":
            st.info("Select a forecast method to run the forecast.")
        elif forecast_method == "Prophet" and not HAS_PROPHET:
            st.error("Prophet is not installed in this environment.")
        else:
            # Prepare CHART series
            fc_data_chart = data_all[(data_all["Date"] >= pd.to_datetime(fc_hist_start)) &
                                     (data_all["Date"] <= pd.to_datetime(fc_hist_end))]
            fc_df_chart = fc_data_chart[fc_data_chart["Line Items"].isin(sel_for_charts)]
            fc_fig = None
            forecast_info = {}
            if not fc_df_chart.empty:
                fc_pivot = fc_df_chart.pivot_table(index="Date", columns="Line Items", values="Amount", aggfunc="sum").sort_index()
                if not fc_pivot.empty:
                    freq_str, prophet_freq, seasonal_periods, freq_label = infer_freq_info(fc_pivot.index)
                    last_period = fc_pivot.index.max()
                    try:
                        offset = pd.tseries.frequencies.to_offset(freq_str)
                    except Exception:
                        offset = pd.tseries.frequencies.to_offset("MS")
                    forecast_start = last_period + offset

                    fc_fig = go.Figure()
                    for col in fc_pivot.columns:
                        s_full = fc_pivot[col].dropna()
                        s = s_full.loc[:last_period].dropna()
                        if s.empty: continue
                        x = np.arange(len(s))
                        if forecast_method == "Linear Trend":
                            m_, b_ = np.polyfit(x, s.values, 1)
                            future_x = np.arange(len(s), len(s) + horizon)
                            future_y = m_ * future_x + b_
                            dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                            method = "Linear Trend"
                        elif forecast_method == "Rolling Average":
                            w = max(1, min(int(roll_window), len(s)))
                            avg = s.rolling(w).mean().iloc[-1]
                            future_y = np.repeat(avg, horizon)
                            dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                            method = f"Rolling Avg (w={w})"
                        elif forecast_method == "ARIMA":
                            try:
                                mod = ARIMA(s, order=(int(arima_p), int(arima_d), int(arima_q))).fit()
                                fut = mod.forecast(steps=horizon)
                                future_y, dates = fut.values, fut.index
                            except Exception:
                                future_y = np.repeat(s.values[-1], horizon)
                                dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                            method = f"ARIMA({arima_p},{arima_d},{arima_q})"
                        elif forecast_method == "Prophet":
                            dfp = s.reset_index().rename(columns={"Date": "ds", col: "y"})
                            mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                            mp.fit(dfp)
                            fut = mp.make_future_dataframe(periods=horizon, freq=prophet_freq)
                            pf = mp.predict(fut).set_index("ds")["yhat"].iloc[-horizon:]
                            future_y, dates = pf.values, pf.index
                            method = "Prophet"
                        else:
                            try:
                                if seasonal_periods and seasonal_periods >= 2:
                                    hw = ExponentialSmoothing(s, trend="add", seasonal="add",
                                                              seasonal_periods=seasonal_periods).fit()
                                else:
                                    hw = ExponentialSmoothing(s, trend="add", seasonal=None).fit()
                                fut = hw.forecast(horizon)
                                future_y, dates = fut.values, fut.index
                            except Exception:
                                future_y = np.repeat(s.values[-1], horizon)
                                dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                            method = "Holt-Winters"

                        forecast_info[col] = {"method": method, "last_hist": float(s.values[-1]), "last_fcst": float(future_y[-1])}
                        fc_fig.add_trace(go.Scatter(x=s_full.index, y=s_full.values, mode="lines", name=col))
                        fc_fig.add_trace(go.Scatter(x=dates, y=future_y, mode="lines",
                                                    name=f"{col} forecast", line=dict(dash="dash")))

                    fc_fig.update_layout(title=f"Forecast (history {fc_hist_start} → {fc_hist_end}, native {freq_label})",
                                         xaxis_title="Date", yaxis_title="Amount",
                                         hovermode="x unified", yaxis_tickformat=",")

            # ===== Build single EXPORT =====
            meta = st.session_state.get("anaplan_meta", {}) or {}
            orig_first_cols: List[str] = (meta.get("orig_first_cols") or [])[:4] or ["Legal Entity","Department","Versions","Line Items"]
            period_cols: List[str] = meta.get("period_cols", [])
            month_fmt: Optional[str] = meta.get("month_fmt")
            month_sep: str = meta.get("month_sep", " ")

            versions_col = next((c for c in orig_first_cols if c.lower() in {"versions","version","scenario"}), "Versions")
            li_col_in_AD = next((c for c in orig_first_cols if c in data_all.columns and c.lower() in {n.lower() for n in LINE_ITEM_SYNS}), None)
            line_item_col_name = li_col_in_AD or "Line Items"

            actual_aliases = {"actual","actuals","act","a"}
            df_actual = data_all.copy()
            if versions_col in df_actual.columns:
                df_actual = df_actual[df_actual[versions_col].astype(str).str.strip().str.lower().isin(actual_aliases)]

            id_cols = [c for c in orig_first_cols if c != versions_col]
            id_cols = [c for c in id_cols if c in data_all.columns]
            if line_item_col_name not in id_cols:
                id_cols.append(line_item_col_name)
            if not id_cols:
                id_cols = ["Line Items"]

            fc_start_global = month_start_after(fc_hist_end)
            months_out_index = pd.date_range(fc_start_global, periods=int(horizon), freq="MS")

            def lab(d): return _label_from_date(d, month_fmt, month_sep)
            existing_labels = list(period_cols)
            add_labels = [lab(d) for d in months_out_index if lab(d) not in existing_labels]
            month_cols_out = existing_labels + add_labels

            forecast_map: Dict[Tuple, pd.Series] = {}
            fit_basis = data_all.copy()
            if versions_col in fit_basis.columns:
                fit_basis = fit_basis[fit_basis[versions_col].astype(str).str.strip().str.lower().isin(actual_aliases)]
            fit_basis = fit_basis[(fit_basis["Date"] >= pd.to_datetime(fc_hist_start)) &
                                  (fit_basis["Date"] <= pd.to_datetime(fc_hist_end))]

            base_keys_df = data_all[id_cols].drop_duplicates()

            for _, id_row in base_keys_df.iterrows():
                key_vals = tuple(id_row.get(c, np.nan) for c in id_cols)
                mask = np.ones(len(fit_basis), dtype=bool)
                for c, v in zip(id_cols, key_vals):
                    mask &= _na_eq_mask(fit_basis[c], v)
                s_hist = fit_basis.loc[mask].set_index("Date")["Amount"].sort_index()
                if s_hist.empty:
                    df_all_actual = data_all.copy()
                    if versions_col in df_all_actual.columns:
                        df_all_actual = df_all_actual[df_all_actual[versions_col].astype(str).str.strip().str.lower().isin(actual_aliases)]
                    mask_all = np.ones(len(df_all_actual), dtype=bool)
                    for c, v in zip(id_cols, key_vals):
                        mask_all &= _na_eq_mask(df_all_actual[c], v)
                    s_hist = df_all_actual.loc[mask_all].set_index("Date")["Amount"].sort_index()

                s_m = s_hist.resample("MS").sum(min_count=1).dropna()
                if s_m.empty:
                    fc_series = pd.Series(0.0, index=months_out_index)
                else:
                    last_hist = s_m.index.max()
                    start_next = last_hist + pd.offsets.MonthBegin(1)
                    fc_end_global = months_out_index[-1]
                    horizon_needed = months_between(start_next, fc_end_global)
                    horizon_needed = int(max(horizon_needed, 1))
                    fut_full = fit_and_forecast_monthly(
                        s_m, forecast_method, horizon_needed,
                        roll_window=int(roll_window),
                        arima_order=(int(arima_p), int(arima_d), int(arima_q))
                    )
                    fc_series = fut_full.reindex(months_out_index)
                    fc_series = fc_series.fillna(method="bfill").fillna(method="ffill").fillna(0.0)

                forecast_map[key_vals] = fc_series

            out_rows = []
            for _, id_row in base_keys_df.iterrows():
                key_vals = tuple(id_row.get(c, np.nan) for c in id_cols)
                mask_a = np.ones(len(df_actual), dtype=bool)
                for c, v in zip(id_cols, key_vals):
                    mask_a &= _na_eq_mask(df_actual[c], v)
                s_act = df_actual.loc[mask_a].set_index("Date")["Amount"].sort_index()
                s_act_m = s_act.resample("MS").sum(min_count=1).dropna()

                fc_series = forecast_map.get(key_vals, pd.Series(0.0, index=months_out_index))

                row_dict = {}
                for c in orig_first_cols:
                    if c == versions_col:
                        row_dict[c] = "Forecast"
                    elif c in id_cols:
                        row_dict[c] = id_row.get(c, "")
                    else:
                        row_dict[c] = ""
                for m in month_cols_out:
                    row_dict[m] = np.nan

                if not s_act_m.empty:
                    for d, v in s_act_m.items():
                        if d < months_out_index[0]:
                            lbl = lab(d)
                            if lbl in row_dict:
                                row_dict[lbl] = float(v)
                for d, v in fc_series.items():
                    lbl = lab(d)
                    if lbl in row_dict:
                        row_dict[lbl] = float(v)

                if any(pd.notna(row_dict[m]) for m in month_cols_out):
                    out_rows.append(row_dict)

            export_df = pd.DataFrame(out_rows)
            ordered_cols = list((meta.get("orig_first_cols") or [])[:4] or ["Legal Entity","Department","Versions","Line Items"])
            ordered_cols = ordered_cols + [c for c in month_cols_out if c not in ordered_cols]
            ordered_cols = [c for c in ordered_cols if c in export_df.columns] + [c for c in export_df.columns if c not in ordered_cols]
            export_df = export_df[ordered_cols]
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")

            save_forecast_cache({
                "fig": fc_fig,
                "forecast_info": forecast_info,
                "export_df": export_df,
                "csv_bytes": csv_bytes,
            })

    # ===== Render cached forecast =====
    cache = get_forecast_cache()
    if cache:
        st.markdown("### 📊 Forecast Results")
        if cache.get("fig") is not None:
            st.plotly_chart(cache["fig"], use_container_width=True)
        if cache.get("forecast_info"):
            st.subheader("🗣️ Forecast Narratives")
            for col, info in cache["forecast_info"].items():
                st.markdown(
                    f"**{col}** — {info['method']}: last **{info['last_hist']:,.0f}**, "
                    f"projected **{info['last_fcst']:,.0f}** at horizon."
                )
        st.subheader("📤 Export — Forecast version rows (ALL line items)")
        st.dataframe(cache["export_df"].head(30), use_container_width=True)

        st.download_button(
            "Download Anaplan Import CSV",
            data=cache["csv_bytes"],
            file_name="anaplan_forecast_version_all_lines.csv",
            mime="text/csv",
            key="btn_download_csv"
        )

        if st.session_state.get("source_mode") == "Export from Anaplan":
            # ----- Push to Anaplan -----
            st.markdown("### 🚀 Push to Anaplan (Import Action)")
            creds = st.session_state.get("anaplan_creds") or {}
            with st.form("anaplan_import_single", clear_on_submit=False):
                st.caption(f"Using saved credentials for workspace **{creds.get('wid','?')}** / model **{creds.get('mid','?')}**.")
                email = creds["email"]; password = creds["password"]; wid = creds["wid"]; mid = creds["mid"]
                locale = creds.get("locale", "en_US")

                top = st.columns([2,2])
                with top[0]:
                    if st.form_submit_button("🔄 Refresh import actions", use_container_width=True):
                        try:
                            token = get_token(email, password)
                            st.session_state["ana_imports"] = list_imports(token, wid, mid)
                            st.success(f"Loaded {len(st.session_state['ana_imports'])} import action(s).")
                        except Exception as e:
                            st.error(f"Failed to list imports: {e}")
                with top[1]:
                    if st.form_submit_button("🔄 Refresh server files", use_container_width=True):
                        try:
                            token = get_token(email, password)
                            st.session_state["ana_files"] = list_files(token, wid, mid)
                            st.success(f"Loaded {len(st.session_state['ana_files'])} server file(s).")
                        except Exception as e:
                            st.error(f"Failed to list files: {e}")

                imports_list = st.session_state.get("ana_imports", [])
                files_list = st.session_state.get("ana_files", [])

                import_id = None
                if imports_list:
                    lbls = [f"{imp.get('name','(no name)')} — {imp.get('id')}" for imp in imports_list]
                    ids = [imp.get("id") for imp in imports_list]
                    idx = 0
                    if "ana_import_selected_id" in st.session_state and st.session_state["ana_import_selected_id"] in ids:
                        idx = ids.index(st.session_state["ana_import_selected_id"])
                    sel_imp = st.selectbox("Import Action", lbls, index=idx, key="imp_sel_lbl")
                    import_id = ids[lbls.index(sel_imp)]
                    st.session_state["ana_import_selected_id"] = import_id
                else:
                    st.info("No imports loaded yet. Click **Refresh import actions**.")

                server_file_override = st.text_input("Or paste a Server File ID (overrides dropdown)", value="", key="imp_file_override")

                server_file_id = None
                if files_list and not server_file_override.strip():
                    flabels = [f"{f.get('name','(no name)')} — {f.get('id')}" for f in files_list]
                    fids = [f.get("id") for f in files_list]
                    fsel = st.selectbox("Server File to upload into", flabels, key="file_sel_lbl")
                    server_file_id = fids[flabels.index(fsel)]
                elif server_file_override.strip():
                    server_file_id = server_file_override.strip()

                fmt_choice = st.selectbox(
                    "Time headers to send to Anaplan",
                    options=["KEEP", "MMM YY", "MMM-YY", "YY-MMM", "MMM YYYY", "YYYY-MM", "MM/YYYY"],
                    index=1,
                    help="If your import expects 'Jan 23', choose MMM YY."
                )
                with st.expander("Advanced"):
                    header_row = st.number_input("Header row (1-based)", min_value=1, value=1)
                    first_data_row = st.number_input("First data row (1-based)", min_value=1, value=2)

                src_choice = st.radio("CSV to import", ["Use generated forecast results", "Upload CSV/TSV file"], horizontal=True, key="imp_src")
                uploaded_csv = None
                if src_choice == "Upload CSV/TSV file":
                    uploaded = st.file_uploader("Select CSV/TSV file", type=["csv", "tsv"], key="imp_upl")
                    if uploaded:
                        uploaded_csv = uploaded.getvalue()

                run_btn = st.form_submit_button("Upload & Run Import", use_container_width=True)

            if run_btn:
                if not all([email, password, wid, mid, import_id]):
                    st.error("Please choose an Import action (and ensure credentials are present).")
                else:
                    ph = st.empty()
                    try:
                        original_bytes = uploaded_csv if uploaded_csv is not None else cache["csv_bytes"]
                        if not original_bytes:
                            raise ValueError("No CSV bytes to upload.")

                        csv_bytes = convert_time_headers(original_bytes, fmt_choice)
                        try:
                            df_check = pd.read_csv(io.BytesIO(csv_bytes), nrows=1, dtype=str)
                            st.caption("Converted header sample ➜ " + ", ".join(df_check.columns[:24]))
                        except Exception:
                            pass

                        token = get_token(email, password)

                        used_file_id = server_file_id
                        used_file_name = None
                        if not used_file_id:
                            used_file_id, used_file_name = resolve_file_id_for_import(token, wid, mid, import_id)
                            if not used_file_id:
                                raise ValueError("Could not determine the Server File to upload into. "
                                                 "Ensure the Import’s source is a model file or pick a Server File ID manually.")

                        fname = "anaplan_forecast_version_all_lines.csv"
                        total = len(csv_bytes)
                        chunk_size = 25 * 1024 * 1024
                        n_chunks = max(1, math.ceil(total / chunk_size))

                        url_begin = f"{BASE}/workspaces/{wid}/models/{mid}/files/{used_file_id}"
                        r = requests.post(url_begin, headers=_api_headers(token, "application/json"),
                                          json={"chunkCount": -1, "headerRow": int(header_row), "firstDataRow": int(first_data_row)},
                                          **requests_kwargs())
                        r.raise_for_status()

                        prog = st.progress(0.0, text="Uploading file…")
                        for i in range(n_chunks):
                            start = i * chunk_size
                            end = min(start + chunk_size, total)
                            ch = csv_bytes[start:end]
                            url_chunk = f"{BASE}/workspaces/{wid}/models/{mid}/files/{used_file_id}/chunks/{i}"
                            rr = requests.put(url_chunk, headers=_api_headers(token, "application/octet-stream", accept="*/*"),
                                              data=ch, **requests_kwargs())
                            rr.raise_for_status()
                            prog.progress((i + 1) / n_chunks, text=f"Uploaded chunk {i+1}/{n_chunks}")
                        prog.empty()

                        url_complete = f"{BASE}/workspaces/{wid}/models/{mid}/files/{used_file_id}/complete"
                        payload = {"id": str(used_file_id), "name": fname, "chunkCount": -1,
                                   "headerRow": int(header_row), "firstDataRow": int(first_data_row)}
                        rc = requests.post(url_complete, headers=_api_headers(token, "application/json"),
                                           json=payload, **requests_kwargs())
                        rc.raise_for_status()
                        ph.info("Uploading file and starting the import…")

                        task_id = run_import(token, wid, mid, import_id, locale=locale)
                        result_task = wait_for_import_task(token, wid, mid, import_id, task_id)
                        ok = result_task.get("result", {}).get("successful", False)
                        result_task["_used_file_id"] = used_file_id
                        result_task["_used_file_name"] = used_file_name or fname
                        if ok:
                            ph.success(f"Import completed ✅ (file {used_file_id})")
                        else:
                            ph.warning(f"Import finished with issues: {result_task}")
                    except Exception as e:
                        st.error(f"Import failed: {type(e).__name__}: {e}")

        else:
            # ----- Save to S3 if file was loaded from S3 -----
            st.markdown("### ☁️ Save forecast CSV to **S3** (overwrite same filename)")
            def _sec(k, default=""):
                try: return st.secrets.get(k, default)
                except Exception: return default
            with st.form("s3_overwrite_form", clear_on_submit=False):
                c1, c2, c3 = st.columns([1.3, 1, 1])
                with c1:
                    s3_bucket = st.text_input("S3 Bucket", value=_sec("s3_bucket", "your-bucket-name"))
                    s3_prefix = st.text_input("S3 Key prefix (folder)", value=_sec("s3_prefix", ""))
                with c2:
                    aws_region = st.text_input("AWS Region", value=_sec("aws_region", "us-east-1"))
                    file_name  = st.text_input("File name", value="anaplan_forecast_version_all_lines.csv",
                                               help="Will overwrite this exact key every time.")
                with c3:
                    st.caption("AWS creds read from secrets.toml")
                do_upload = st.form_submit_button("Upload (overwrite in S3)")

            if do_upload:
                ph = st.empty()
                try:
                    uri = upload_forecast_csv_to_s3(
                        csv_bytes=cache["csv_bytes"],
                        bucket=s3_bucket,
                        region=aws_region,
                        prefix=s3_prefix,
                        filename=file_name.strip() or "anaplan_forecast_version_all_lines.csv",
                        verify_tls=get_verify_flag()
                    )
                    ph.success(f"Uploaded and replaced: **{uri}**")
                except Exception as e:
                    ph.error(f"{type(e).__name__}: {e}")

        with st.expander("Advanced"):
            if st.button("Clear forecast results", key="btn_clear_cache"):
                clear_forecast_cache()
                st.experimental_rerun()

if __name__ == "__main__":
    main()
