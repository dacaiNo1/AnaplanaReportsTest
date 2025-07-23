#!/usr/bin/env python3
import os, ssl, certifi
# Ensure certifi’s CA bundle is used for HTTPS (including S3)
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import boto3
from io import StringIO
from datetime import datetime
from fredapi import Fred
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

st.set_page_config(page_title="P&L + Macro & Micro Trend Analysis", layout="wide")

# ----------------------------------------
# Loader: detect multiple header date formats
# ----------------------------------------
@st.cache_data
def load_and_melt(csv_file):
    df = pd.read_csv(csv_file)
    date_formats = ['%b %y','%b %Y','%B %y','%B %Y']
    period_cols = []
    for col in df.columns:
        clean = col.strip().replace('-', ' ').replace('_', ' ')
        for fmt in date_formats:
            try:
                datetime.strptime(clean, fmt)
                period_cols.append(col)
                break
            except:
                continue
    if not period_cols:
        st.error("No monthly columns found. Supported: Jan 23, Jan-2023, January 23, January 2023.")
        return None
    id_vars = [c for c in df.columns if c not in period_cols]
    m = df.melt(id_vars=id_vars, value_vars=period_cols,
                var_name="Period", value_name="Amount")
    m["PeriodClean"] = m["Period"].str.replace('-', ' ').str.replace('_', ' ')
    for fmt in date_formats:
        try:
            m["Date"] = pd.to_datetime(m["PeriodClean"], format=fmt, errors='raise')
            break
        except:
            continue
    else:
        m["Date"] = pd.to_datetime(m["PeriodClean"], errors='coerce')
    m["Amount"] = pd.to_numeric(m["Amount"], errors='coerce')
    return m.drop(columns="PeriodClean")

# ----------------------------------------
# Fetch macro/micro indicators from FRED
# ----------------------------------------
@st.cache_data
def fetch_macro_data(start: datetime, end: datetime):
    key = st.secrets.get("general", {}).get("FRED_API_KEY") or os.getenv("FRED_API_KEY")
    if not key:
        st.error("FRED_API_KEY not set.")
        return pd.DataFrame()
    fred = Fred(api_key=key)
    series_map = {
        "CPIAUCSL":"Inflation (CPI)",
        "UNRATE":"Unemployment Rate",
        "FEDFUNDS":"Fed Funds Rate",
        "GS10":"10Y Treasury Yield",
        "INDPRO":"Industrial Production",
        "M2SL":"Money Supply (M2)",
        "UMCSENT":"Consumer Sentiment",
        "PPIACO":"PPI – Finished Goods",
        "NAPM":"ISM Manufacturing PMI",
        "DCOILWTICO":"Crude Oil Price (WTI)",
        "JTSJOR":"Job Openings (JOLTS)",
        "RSAFS":"Retail Sales (Total)",
        "TCU":"Capacity Utilization"
    }
    data = {}
    for code, name in series_map.items():
        try:
            s = fred.get_series(code, start, end).resample("M").last()
        except:
            s = pd.Series(dtype=float, index=pd.date_range(start, end, freq="M"))
        data[name] = s
    return pd.DataFrame(data)

def main():
    st.title("📊 P&L + Macro & Micro Trend Analysis")

    # Step 1: File loading
    st.markdown("### Step 1: Choose File Source")
    source = st.radio("File Source:", ["Manual Upload","Load from S3"], horizontal=True)
    if source == "Manual Upload":
        uploaded = st.file_uploader("Upload P&L CSV", type="csv")
        if uploaded:
            st.session_state.buffer = uploaded
    else:
        bucket = st.text_input("S3 Bucket Name", value="hannahtest12345")
        key    = st.text_input("S3 File Key",    value="REP100_ P&L Summary (8).csv")
        if st.button("Load CSV from S3"):
            try:
                s3 = boto3.client(
                    's3',
                    region_name=st.secrets["aws_region"],
                    aws_access_key_id=st.secrets["aws_access_key"],
                    aws_secret_access_key=st.secrets["aws_secret_key"],
                    verify=False
                )
                resp = s3.get_object(Bucket=bucket, Key=key)
                content = resp['Body'].read().decode('utf-8')
                st.session_state.buffer = StringIO(content)
                st.success(f"Loaded from S3: {key}")
            except Exception as e:
                st.error(f"Failed to load from S3: {e}")

    buf = st.session_state.get("buffer", None)
    if buf is None:
        st.info("Please upload or load a CSV to proceed.")
        return

    # Step 2: Load & melt
    data = load_and_melt(buf)
    if data is None:
        return

    # Step 3: Data Filters
    st.sidebar.header("Data Filters")
    versions = sorted(data["Versions"].dropna().unique())
    sel_version = st.sidebar.selectbox("Version", versions)
    data = data[data["Versions"] == sel_version]

    lob = sorted(data["Line of Business L2 SS: Active"].dropna().unique())
    sel_lob = st.sidebar.multiselect("Line of Business", lob, default=lob)
    data = data[data["Line of Business L2 SS: Active"].isin(sel_lob)]

    legal = sorted(data["Legal Entity L2 SS: Active"].dropna().unique())
    sel_legal = st.sidebar.multiselect("Legal Entity", legal, default=legal)
    data = data[data["Legal Entity L2 SS: Active"].isin(sel_legal)]

    centers = sorted(data["Cost Center L3 SS: Active"].dropna().unique())
    sel_centers = st.sidebar.multiselect("Cost Center", centers, default=centers)
    data = data[data["Cost Center L3 SS: Active"].isin(sel_centers)]

    st.subheader("Filtered Data Preview")
    st.dataframe(data, use_container_width=True)

    # Step 4: Analysis options
    st.sidebar.header("Options")
    agg = st.sidebar.selectbox("Aggregation", ["Monthly","Quarterly","Annual"])
    chart_type = st.sidebar.selectbox("Chart Type", ["Line","Area","Bar"])
    forecast_method = st.sidebar.selectbox(
        "Forecast Method",
        ["None","Linear Trend","Rolling Average","ARIMA","Prophet","Holt-Winters"]
    )
    if forecast_method != "None":
        horizon = st.sidebar.slider("Forecast horizon (periods)",1,12,3)
    if forecast_method == "Rolling Average":
        roll_window = st.sidebar.slider("Rolling window size",1,12,3)
    if forecast_method == "ARIMA":
        p = st.sidebar.number_input("ARIMA p",0,5,1)
        d = st.sidebar.number_input("ARIMA d",0,2,1)
        q = st.sidebar.number_input("ARIMA q",0,5,0)

    run_forecast = st.sidebar.button("Run Forecast")

    # Step 5: Date Range
    min_d, max_d = data["Date"].min().date(), data["Date"].max().date()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", min_value=min_d, max_value=max_d, value=min_d)
    with c2:
        end_date   = st.date_input("End Date",   min_value=min_d, max_value=max_d, value=max_d)
    if start_date > end_date:
        st.error("Start must be on or before End.")
        return
    data = data[(data["Date"] >= pd.to_datetime(start_date)) &
                (data["Date"] <= pd.to_datetime(end_date))]

    # Step 6: Select & Pivot
    items = sorted(data["Line Items"].unique())
    sel = st.multiselect("Select Line Items", items, default=items[:3])
    if not sel:
        st.warning("Pick at least one.")
        return
    df_sel = data[data["Line Items"].isin(sel)]
    pivot = df_sel.pivot_table(
        index="Date", columns="Line Items", values="Amount", aggfunc="sum"
    ).sort_index()
    if agg != "Monthly":
        pivot = pivot.resample({"Monthly":"M","Quarterly":"Q","Annual":"A"}[agg]).sum()
    pivot = pivot.sort_index()

    # % Change
    pct = pivot.pct_change().dropna() * 100
    st.subheader(f"{agg} % Change")
    st.dataframe(pct.style.format("{:+.2f}%"))

    # P&L Chart
    st.subheader(f"{agg} {chart_type} Chart")
    if chart_type == "Line":
        fig = px.line(pivot, x=pivot.index, y=pivot.columns,
                      labels={"value":"Amount","Date":"Date"})
    elif chart_type == "Area":
        fig = px.area(pivot, x=pivot.index, y=pivot.columns,
                      labels={"value":"Amount","Date":"Date"})
    else:
        dfm = pivot.reset_index().melt(id_vars="Date",
                                       var_name="Line Items",
                                       value_name="Amount")
        fig = px.bar(dfm, x="Date", y="Amount",
                     color="Line Items", barmode="group")
    fig.update_layout(hovermode="x unified", yaxis_tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

    # Determine forecast_start as next month of end_date
    freq_str = {"Monthly":"M","Quarterly":"Q","Annual":"A"}[agg]
    offset = pd.tseries.frequencies.to_offset(freq_str)
    last_period = pivot.index.max()
    forecast_start = last_period + offset

    # Step 7: Forecast & Narratives
    if forecast_method != "None" and run_forecast:
        fc_fig = go.Figure()
        forecast_info = {}

        for col in pivot.columns:
            s_full = pivot[col].dropna()
            # truncate to historical only
            s = s_full.loc[:last_period].dropna()
            x = np.arange(len(s))

            # Linear Trend
            if forecast_method == "Linear Trend":
                m_, b_ = np.polyfit(x, s.values, 1)
                future_x = np.arange(len(s), len(s) + horizon)
                future_y = m_*future_x + b_
                dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                method = "Linear Trend"

            # Rolling Average
            elif forecast_method == "Rolling Average":
                w = min(roll_window, len(s))
                avg = s.rolling(w).mean().iloc[-1]
                future_y = np.repeat(avg, horizon)
                dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                method = f"Rolling Avg (w={w})"

            # ARIMA
            elif forecast_method == "ARIMA":
                try:
                    mod = ARIMA(s, order=(p, d, q)).fit()
                    fut = mod.forecast(steps=horizon)
                    future_y, dates = fut.values, fut.index
                except:
                    future_y = np.repeat(s.values[-1], horizon)
                    dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                method = f"ARIMA({p},{d},{q})"

            # Prophet
            elif forecast_method == "Prophet":
                dfp = s.reset_index().rename(columns={"Date":"ds", col:"y"})
                mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                mp.fit(dfp)
                fut = mp.make_future_dataframe(periods=horizon, freq=freq_str)
                pf  = mp.predict(fut).set_index("ds")["yhat"].iloc[-horizon:]
                future_y, dates = pf.values, pf.index
                method = "Prophet"

            # Holt-Winters
            else:
                sp = {"Monthly":12,"Quarterly":4,"Annual":1}[agg]
                try:
                    hw = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=sp).fit()
                    fut = hw.forecast(horizon)
                    future_y, dates = fut.values, fut.index
                except:
                    future_y = np.repeat(s.values[-1], horizon)
                    dates = pd.date_range(forecast_start, periods=horizon, freq=freq_str)
                method = "Holt-Winters"

            forecast_info[col] = {
                "method": method,
                "last_hist": float(s.values[-1]),
                "last_fcst": float(future_y[-1])
            }

            fc_fig.add_trace(go.Scatter(x=s_full.index, y=s_full.values, mode="lines", name=col))
            fc_fig.add_trace(go.Scatter(x=dates, y=future_y, mode="lines",
                                       name=f"{col} forecast", line=dict(dash="dash")))

        fc_fig.update_layout(
            title="Trend & Forecast",
            xaxis_title="Date", yaxis_title="Amount",
            hovermode="x unified", yaxis_tickformat=","
        )
        st.plotly_chart(fc_fig, use_container_width=True)

        # Narratives
        left, right = st.columns(2)
        with left:
            st.subheader("🔍 Trend Analysis & Detailed Narratives")
            for item in sel:
                s  = pivot[item].dropna()
                sv, ev = s.iloc[0], s.iloc[-1]
                pct = (ev - sv) / (abs(sv) if sv != 0 else 1) * 100
                d1 = s.index[0].strftime("%m/%d/%Y")
                d2 = s.index[-1].strftime("%m/%d/%Y")

                cors = {}
                macro = fetch_macro_data(pd.to_datetime(start_date), pd.to_datetime(end_date))
                for name in macro.columns:
                    dfc = pd.concat([s.rename(item),
                                     macro[name].reindex(pivot.index).rename(name)],
                                    axis=1, join="inner").dropna()
                    if len(dfc) > 1:
                        cors[name] = dfc[item].corr(dfc[name])
                top3 = sorted(cors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

                st.markdown(f"**Trend Analysis for {item}**")
                st.markdown(
                    f"It went from **{sv:,.0f}** on **{d1}** to **{ev:,.0f}** on **{d2}** — **{pct:+.2f}%** collapse."
                )
                for name, corr in top3:
                    sign = "+" if corr>0 else ""
                    st.markdown(
                        f"- A high {'positive' if corr>0 else 'negative'} correlation "
                        f"({sign}{corr:.2f}) means when **{name}** rose, **{item}** tended to "
                        f"{'rise' if corr>0 else 'fall'}."
                    )

        with right:
            st.subheader("🔮 Forecast Narratives")
            for col, info in forecast_info.items():
                last   = info["last_hist"]
                last_d = pivot[col].dropna().index[-1].strftime("%d/%m/%Y")
                fc     = info["last_fcst"]
                method = info["method"]

                st.markdown(f"**Forecast for {col}**")
                st.markdown(
                    f"Using **{method}**, last value was **{last:,.0f}** on **{last_d}**, "
                    f"projected to **{fc:,.0f}** in **{horizon}** periods."
                )

    # Macro & Micro Context
    macro = fetch_macro_data(pd.to_datetime(start_date), pd.to_datetime(end_date))
    if not macro.empty:
        st.subheader("📈 Macro & Micro Context")
        ms = macro.apply(lambda c: c.dropna().iloc[0] if not c.dropna().empty else np.nan, axis=0)
        me = macro.apply(lambda c: c.dropna().iloc[-1] if not c.dropna().empty else np.nan, axis=0)
        mp = (me - ms) / ms * 100
        md = mp.apply(lambda x: "up" if x>0 else "down" if x<0 else "flat")
        for name in macro.columns:
            st.markdown(
                f"• **{name}**: {md[name]}, from **{ms[name]:.2f}** to **{me[name]:.2f}** "
                f"({mp[name]:+.2f}% change)"
            )

if __name__ == "__main__":
    main()
