#!/usr/bin/env python3
import os, certifi
# Use certifi’s CA bundle for HTTPS
os.environ['SSL_CERT_FILE'] = certifi.where()

import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime
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
        st.error("No monthly columns found. Supported: Jan 23, Jan‑2023, January 23, January 2023.")
        return None
    id_vars = [c for c in df.columns if c not in period_cols]
    m = df.melt(id_vars=id_vars, value_vars=period_cols, var_name="Period", value_name="Amount")
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
    key = st.secrets.get("general",{}).get("FRED_API_KEY") or os.getenv("FRED_API_KEY")
    if not key:
        st.error("FRED_API_KEY not set.")
        return pd.DataFrame()
    fred = Fred(api_key=key)
    series_map = {
        "CPIAUCSL":"Inflation (CPI)","UNRATE":"Unemployment Rate",
        "FEDFUNDS":"Fed Funds Rate","GS10":"10Y Treasury Yield",
        "INDPRO":"Industrial Production","M2SL":"Money Supply (M2)",
        "UMCSENT":"Consumer Sentiment","PPIACO":"PPI – Finished Goods",
        "NAPM":"ISM Manufacturing PMI","DCOILWTICO":"Crude Oil Price (WTI)",
        "JTSJOR":"Job Openings (JOLTS)","RSAFS":"Retail Sales (Total)",
        "TCU":"Capacity Utilization"
    }
    data = {}
    for code,name in series_map.items():
        try:
            s = fred.get_series(code, start, end).resample("M").last()
        except:
            s = pd.Series(dtype=float, index=pd.date_range(start, end, freq="M"))
        data[name] = s
    return pd.DataFrame(data)

def main():
    st.title("📊 P&L + Macro & Micro Trend Analysis")

    # Sidebar controls
    st.sidebar.header("Options")
    agg = st.sidebar.selectbox("Aggregation", ["Monthly","Quarterly","Annual"])
    chart_type = st.sidebar.selectbox("Chart Type", ["Line","Area","Bar"])
    forecast_method = st.sidebar.selectbox("Forecast Method",
        ["None","Linear Trend","Rolling Average","ARIMA","Prophet","Holt-Winters"])
    if forecast_method!="None":
        horizon = st.sidebar.slider("Forecast horizon (periods)",1,12,3)
    if forecast_method=="Rolling Average":
        roll_window = st.sidebar.slider("Rolling window size",1,12,3)
    if forecast_method=="ARIMA":
        p = st.sidebar.number_input("ARIMA p",0,5,1)
        d = st.sidebar.number_input("ARIMA d",0,2,1)
        q = st.sidebar.number_input("ARIMA q",0,5,0)

    # Upload CSV
    uploaded = st.file_uploader("Upload P&L CSV", type="csv")
    if not uploaded:
        st.info("Please upload a CSV.")
        return
    data = load_and_melt(uploaded)
    if data is None:
        return

    # Analysis period
    min_d, max_d = data.Date.min().date(), data.Date.max().date()
    c1,c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
    with c2:
        end_date = st.date_input("End Date", max_d, min_value=min_d, max_value=max_d)
    if start_date> end_date:
        st.error("Start ≤ End.")
        return
    data = data[(data.Date>=pd.to_datetime(start_date)) & (data.Date<=pd.to_datetime(end_date))]

    # Select items
    items = sorted(data["Line Items"].unique())
    sel = st.multiselect("Select Line Items", items, default=items[:3])
    if not sel:
        st.warning("Pick at least one.")
        return
    df_sel = data[data["Line Items"].isin(sel)]
    pivot = df_sel.pivot_table(index="Date",columns="Line Items",values="Amount",aggfunc="sum").sort_index()

    # Resample
    freq = {"Monthly":"M","Quarterly":"Q","Annual":"A"}[agg]
    if agg!="Monthly":
        pivot = pivot.resample(freq).sum()
    pivot = pivot.sort_index()

    # % Change table
    pct = pivot.pct_change().dropna()*100
    st.subheader(f"{agg} % Change")
    st.dataframe(pct.style.format("{:+.2f}%"))

    # P&L chart
    st.subheader(f"{agg} {chart_type} Chart")
    if chart_type=="Line":
        fig = px.line(pivot,x=pivot.index,y=pivot.columns,labels={"value":"Amount","Date":"Date"})
    elif chart_type=="Area":
        fig = px.area(pivot,x=pivot.index,y=pivot.columns,labels={"value":"Amount","Date":"Date"})
    else:
        dfm = pivot.reset_index().melt(id_vars="Date",var_name="Line Items",value_name="Amount")
        fig = px.bar(dfm,x="Date",y="Amount",color="Line Items",barmode="group")
    fig.update_layout(hovermode="x unified",yaxis_tickformat=",")
    st.plotly_chart(fig,use_container_width=True)

    # Forecast chart + narrative
    if forecast_method!="None":
        fc_fig = go.Figure()
        forecast_info = {}
        for col in pivot.columns:
            s = pivot[col].dropna(); x=np.arange(len(s))
            # compute forecast
            if forecast_method=="Linear Trend":
                m_,b_ = np.polyfit(x,s.values,1)
                future_x=np.arange(len(s),len(s)+horizon)
                future_y=m_*future_x+b_
                dates=pd.date_range(s.index[-1]+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
                method="Linear Trend"
            elif forecast_method=="Rolling Average":
                w=min(roll_window,len(s)); avg=s.rolling(w).mean().iloc[-1]
                future_y=np.repeat(avg,horizon)
                dates=pd.date_range(s.index[-1]+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
                method=f"Rolling Avg (w={w})"
            elif forecast_method=="ARIMA":
                try:
                    mod=ARIMA(s,order=(p,d,q)).fit()
                    fut=mod.forecast(steps=horizon)
                    future_y=fut.values; dates=fut.index
                except:
                    future_y=np.repeat(s.values[-1],horizon)
                    dates=pd.date_range(s.index[-1]+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
                method=f"ARIMA({p},{d},{q})"
            elif forecast_method=="Prophet":
                dfp=s.reset_index().rename(columns={"Date":"ds",col:"y"})
                mp=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
                mp.fit(dfp)
                fut=mp.make_future_dataframe(periods=horizon,freq=freq)
                pf=mp.predict(fut).set_index("ds")["yhat"].iloc[-horizon:]
                future_y=pf.values; dates=pf.index
                method="Prophet"
            else: # Holt‑Winters
                sp={"Monthly":12,"Quarterly":4,"Annual":1}[agg]
                try:
                    hw=ExponentialSmoothing(s,trend="add",seasonal="add",seasonal_periods=sp).fit()
                    fut=hw.forecast(horizon)
                    future_y=fut.values; dates=fut.index
                except:
                    future_y=np.repeat(s.values[-1],horizon)
                    dates=pd.date_range(s.index[-1]+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
                method="Holt-Winters"
            forecast_info[col] = {"method":method,"last_hist":float(s.values[-1]),"last_fcst":float(future_y[-1])}
            fc_fig.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines",name=col))
            fc_fig.add_trace(go.Scatter(x=dates,y=future_y,mode="lines",name=f"{col} forecast",line=dict(dash="dash")))
        fc_fig.update_layout(title="Trend & Forecast",xaxis_title="Date",yaxis_title="Amount",
                             hovermode="x unified",yaxis_tickformat=",")
        st.plotly_chart(fc_fig,use_container_width=True)

        # Side-by-side: Trend Analysis and Forecast Narratives
        left, right = st.columns(2)
        with left:
            st.subheader("🔍 Trend Analysis & Detailed Narratives")
            for item in sel:
                s = pivot[item].dropna()
                if s.empty:
                    st.markdown(f"**{item}**: No data."); continue
                sv,ev=s.iloc[0],s.iloc[-1]
                pct=(ev-sv)/(abs(sv) if sv!=0 else 1)*100
                # correlations
                cors={}
                for name in fetch_macro_data(pd.to_datetime(start_date),pd.to_datetime(end_date)).columns:
                    dfc=pd.concat([s.rename(item),fetch_macro_data(pd.to_datetime(start_date),pd.to_datetime(end_date))[name].reindex(pivot.index).rename(name)],axis=1,join='inner').dropna()
                    if len(dfc)>1: cors[name]=dfc[item].corr(dfc[name])
                top3=sorted(cors.items(),key=lambda x:abs(x[1]),reverse=True)[:3]
                st.markdown(f"""
<div style="font-family:Arial,sans-serif;font-size:16px;line-height:1.5;margin-bottom:1em;">
<b>Trend Analysis for {item}</b><br>
It went from <b>${sv:,.0f}</b> to <b>${ev:,.0f}</b>—{pct:+.2f}% {'collapse' if pct<0 else 'increase'}.<br><br>
<ul>
""", unsafe_allow_html=True)
                for name,corr in top3:
                    desc = (f"A high positive correlation (+{corr:.2f}) means when <b>{name}</b> rose, <b>{item}</b> also tended to rise."
                            if corr>0 else
                            f"A high negative correlation ({corr:.2f}) means when <b>{name}</b> rose, <b>{item}</b> tended to fall.")
                    st.markdown(f"<li style=\"font-family:Arial,sans-serif;font-size:16px;\">{desc}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)

        with right:
            st.subheader("🔮 Forecast Narratives")
            for col,info in forecast_info.items():
                st.markdown(f"""
<div style="font-family:Arial,sans-serif;font-size:16px;line-height:1.5;margin-bottom:1em;">
<b>Forecast for {col}</b><br>
Using <i>{info['method']}</i>, last value was <b>${info['last_hist']:,.0f}</b>,<br>
projected to {'fall' if info['last_fcst']<info['last_hist'] else 'rise'} to <b>${info['last_fcst']:,.0f}</b> over {horizon} periods.
</div>
""", unsafe_allow_html=True)

    # Macro & Micro context at bottom
    macro = fetch_macro_data(pd.to_datetime(start_date), pd.to_datetime(end_date))
    if not macro.empty:
        st.subheader("📈 Macro & Micro Context")
        ms=macro.apply(lambda col:col[col.first_valid_index()] if col.first_valid_index() else np.nan)
        me=macro.apply(lambda col:col[col.last_valid_index()] if col.last_valid_index() else np.nan)
        mp=(me-ms)/ms*100; md=mp.apply(lambda x:"up" if x>0 else "down" if x<0 else "flat")
        for name in macro.columns:
            st.markdown(f"• **{name}**: {md[name]}, from {ms[name]:.2f} to {me[name]:.2f} ({mp[name]:+.2f}% change)")

if __name__ == "__main__":
    main()
