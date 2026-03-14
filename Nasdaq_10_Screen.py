# streamlit_app.py
# NASDAQ Quant Screener Professional Version
# Menu + Tabs + Graph Analysis

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="NASDAQ Quant Screener", layout="wide")

st.title("📈 NASDAQ Quant Screener")

# --------------------------------------------------
# Candidate Universe
# --------------------------------------------------

CANDIDATES = [
"MSFT","AAPL","NVDA","AMZN","GOOG","META",
"AVGO","TSLA","COST","NFLX","AMD","ADBE",
"CSCO","INTC","QCOM","TXN","AMGN"
]

# --------------------------------------------------
# Sidebar Menu
# --------------------------------------------------

menu = st.sidebar.selectbox(
"Menu",
[
"Dashboard",
"Market Cap Top10",
"Quant Analysis",
"Factor Analysis",
"Single Stock Analysis"
]
)

refresh = st.sidebar.button("🔄 Refresh Market Data")

# --------------------------------------------------
# Data Loader
# --------------------------------------------------

@st.cache_data(ttl=3600)

def load_data():

    rows=[]

    for t in CANDIDATES:

        tk=yf.Ticker(t)

        info=tk.info

        price=tk.history(period="1d")["Close"].iloc[-1]

        rows.append({

            "ticker":t,
            "name":info.get("shortName",""),
            "market_cap":info.get("marketCap",np.nan),
            "forward_pe":info.get("forwardPE",np.nan),
            "peg":info.get("pegRatio",np.nan),
            "roe":info.get("returnOnEquity",np.nan),
            "revenue_growth":info.get("revenueGrowth",np.nan),
            "profit_margin":info.get("profitMargins",np.nan),
            "price":price

        })

    df=pd.DataFrame(rows)

    df=df.sort_values("market_cap",ascending=False)

    return df

df=load_data()

top10=df.head(10)

# --------------------------------------------------
# Dashboard
# --------------------------------------------------

if menu=="Dashboard":

    st.subheader("Overview")

    c1,c2,c3=st.columns(3)

    c1.metric("Universe",len(CANDIDATES))
    c2.metric("Top Market Cap",top10.iloc[0]["ticker"])
    c3.metric("Last Update",datetime.now().strftime("%H:%M:%S"))

    st.dataframe(top10)

# --------------------------------------------------
# Market Cap Top10
# --------------------------------------------------

if menu=="Market Cap Top10":

    st.subheader("Top 10 NASDAQ Market Cap")

    fig=px.bar(
        top10,
        x="market_cap",
        y="ticker",
        orientation="h",
        title="Market Cap Ranking"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.dataframe(top10)

# --------------------------------------------------
# Quant Analysis Tabs
# --------------------------------------------------

if menu=="Quant Analysis":

    st.subheader("Quant Factor Analysis")

    tab1,tab2,tab3,tab4,tab5=st.tabs(
        ["Value","Quality","Growth","Stability","Momentum"]
    )

    # ---------------- Value ----------------

    with tab1:

        st.subheader("Value Factors")

        fig=px.scatter(
            top10,
            x="forward_pe",
            y="peg",
            size="market_cap",
            text="ticker",
            title="Forward PE vs PEG"
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Quality ----------------

    with tab2:

        st.subheader("Quality Factors")

        fig=px.bar(
            top10,
            x="ticker",
            y="roe",
            title="Return on Equity"
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Growth ----------------

    with tab3:

        st.subheader("Growth Factors")

        fig=px.bar(
            top10,
            x="ticker",
            y="revenue_growth",
            title="Revenue Growth"
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Stability ----------------

    with tab4:

        st.subheader("Stability")

        fig=px.bar(
            top10,
            x="ticker",
            y="profit_margin",
            title="Profit Margin"
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Momentum ----------------

    with tab5:

        st.subheader("Momentum")

        momentum=[]

        for t in top10["ticker"]:

            hist=yf.Ticker(t).history(period="6mo")

            momentum.append(hist["Close"].pct_change().mean())

        top10["momentum"]=momentum

        fig=px.bar(
            top10,
            x="ticker",
            y="momentum",
            title="6M Momentum"
        )

        st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------
# Factor Analysis
# --------------------------------------------------

if menu=="Factor Analysis":

    st.subheader("Factor Comparison")

    fig=px.scatter(
        top10,
        x="roe",
        y="revenue_growth",
        size="market_cap",
        text="ticker",
        title="Quality vs Growth"
    )

    st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------
# Single Stock Analysis
# --------------------------------------------------

if menu=="Single Stock Analysis":

    ticker=st.selectbox("Select Stock",top10["ticker"])

    hist=yf.Ticker(ticker).history(period="1y")

    fig=px.line(
        hist,
        y="Close",
        title=f"{ticker} Price Chart"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.dataframe(hist.tail())
