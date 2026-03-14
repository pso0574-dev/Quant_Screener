# streamlit_app.py
# pip install streamlit yfinance pandas numpy plotly

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Nasdaq-100 Quant Strategy Dashboard", layout="wide")

st.title("Nasdaq-100 Quant Strategy Dashboard")
st.caption("ROE / MDD based undervalued stock screening with strategy tabs")

# =========================================================
# 1) NASDAQ-100 ticker universe
# =========================================================
# For production stability, many people keep a local CSV backup.
# Here we use a maintained list directly in code for simplicity.

NASDAQ100_TICKERS = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN",
    "AMZN","ANSS","ARM","ASML","AVGO","AXON","AZN","BIIB","BKNG","CDNS",
    "CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH",
    "DASH","DDOG","DXCM","EA","EXC","FANG","FAST","FTNT","GEHC","GFS",
    "GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP","KHC",
    "KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST",
    "MRVL","MSFT","MU","NFLX","NVDA","ODFL","ON","ORLY","PANW","PAYX",
    "PCAR","PDD","PEP","PLTR","PYPL","QCOM","REGN","ROP","ROST","SBUX",
    "SNPS","TEAM","TMUS","TSLA","TTD","TTWO","TXN","VRSK","VRTX","WBD",
    "WDAY","XEL","ZS"
]

# Some data vendors may return BRK.B style issues differently, but Nasdaq-100 mostly avoids that.
universe = NASDAQ100_TICKERS

# =========================================================
# 2) Sidebar
# =========================================================
st.sidebar.header("Settings")
period = st.sidebar.selectbox("Price lookback", ["1y", "2y", "3y", "5y"], index=1)
top_n = st.sidebar.slider("Top N", 5, 30, 10)
min_roe = st.sidebar.slider("Minimum ROE (%)", 0, 40, 10)
min_mktcap_b = st.sidebar.slider("Minimum Market Cap ($B)", 0, 500, 10)
refresh = st.sidebar.button("Refresh Data")

# =========================================================
# 3) Helpers
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(tickers, period="2y"):
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(tickers):
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            roe = info.get("returnOnEquity", np.nan)
            market_cap = info.get("marketCap", np.nan)
            trailing_pe = info.get("trailingPE", np.nan)
            forward_pe = info.get("forwardPE", np.nan)
            gross_margin = info.get("grossMargins", np.nan)
            operating_margin = info.get("operatingMargins", np.nan)
            revenue_growth = info.get("revenueGrowth", np.nan)
            debt_to_equity = info.get("debtToEquity", np.nan)
            fcf = info.get("freeCashflow", np.nan)

            rows.append({
                "Ticker": t,
                "ROE": roe * 100 if pd.notna(roe) else np.nan,
                "MarketCap_B": market_cap / 1e9 if pd.notna(market_cap) else np.nan,
                "TrailingPE": trailing_pe,
                "ForwardPE": forward_pe,
                "GrossMargin": gross_margin * 100 if pd.notna(gross_margin) else np.nan,
                "OperatingMargin": operating_margin * 100 if pd.notna(operating_margin) else np.nan,
                "RevenueGrowth": revenue_growth * 100 if pd.notna(revenue_growth) else np.nan,
                "DebtToEquity": debt_to_equity,
                "FCF_B": fcf / 1e9 if pd.notna(fcf) else np.nan,
            })
        except Exception:
            rows.append({
                "Ticker": t,
                "ROE": np.nan,
                "MarketCap_B": np.nan,
                "TrailingPE": np.nan,
                "ForwardPE": np.nan,
                "GrossMargin": np.nan,
                "OperatingMargin": np.nan,
                "RevenueGrowth": np.nan,
                "DebtToEquity": np.nan,
                "FCF_B": np.nan,
            })
    return pd.DataFrame(rows)

def compute_mdd_and_momentum(price_df, tickers):
    rows = []
    for t in tickers:
        try:
            s = price_df[t]["Close"].dropna() if isinstance(price_df.columns, pd.MultiIndex) else price_df["Close"].dropna()
            if len(s) < 60:
                continue

            roll_max = s.cummax()
            dd = (s / roll_max - 1.0) * 100
            mdd = dd.min()

            ret_6m = (s.iloc[-1] / s.iloc[max(0, len(s)-126)] - 1.0) * 100 if len(s) > 126 else np.nan
            ret_3m = (s.iloc[-1] / s.iloc[max(0, len(s)-63)] - 1.0) * 100 if len(s) > 63 else np.nan

            ma50 = s.rolling(50).mean().iloc[-1]
            ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
            dist_200ma = ((s.iloc[-1] / ma200) - 1.0) * 100 if pd.notna(ma200) and ma200 != 0 else np.nan

            vol_1y = s.pct_change().dropna().std() * np.sqrt(252) * 100

            rows.append({
                "Ticker": t,
                "Price": s.iloc[-1],
                "MDD": mdd,
                "Momentum_6M": ret_6m,
                "Momentum_3M": ret_3m,
                "Dist_200MA": dist_200ma,
                "Volatility_1Y": vol_1y,
            })
        except Exception:
            pass

    return pd.DataFrame(rows)

def pct_rank_high(series):
    return series.rank(pct=True, ascending=True)

def pct_rank_low(series):
    return 1 - series.rank(pct=True, ascending=True)

def safe_fill(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

def build_scores(df):
    df = df.copy()
    df = safe_fill(df, df.columns)

    # favorable-high metrics
    df["r_ROE"] = pct_rank_high(df["ROE"])
    df["r_GrossMargin"] = pct_rank_high(df["GrossMargin"])
    df["r_OpMargin"] = pct_rank_high(df["OperatingMargin"])
    df["r_RevenueGrowth"] = pct_rank_high(df["RevenueGrowth"])
    df["r_Mom6"] = pct_rank_high(df["Momentum_6M"])
    df["r_FCF"] = pct_rank_high(df["FCF_B"])

    # favorable-low metrics
    # MDD is more attractive when more negative, e.g. -30% cheaper than -8%
    df["r_MDD"] = pct_rank_low(df["MDD"])
    df["r_Debt"] = pct_rank_low(df["DebtToEquity"])
    df["r_Vol"] = pct_rank_low(df["Volatility_1Y"])
    df["r_PE"] = pct_rank_low(df["ForwardPE"].fillna(df["TrailingPE"]))

    # Strategy scores
    df["Score_ROE_MDD"] = 0.60 * df["r_ROE"] + 0.40 * df["r_MDD"]

    df["Score_Quality_Pullback"] = (
        0.30 * df["r_ROE"] +
        0.20 * df["r_GrossMargin"] +
        0.15 * df["r_OpMargin"] +
        0.20 * df["r_MDD"] +
        0.15 * df["r_Debt"]
    )

    df["Score_Recovery_Momentum"] = (
        0.30 * df["r_ROE"] +
        0.30 * df["r_MDD"] +
        0.25 * df["r_Mom6"] +
        0.15 * df["r_RevenueGrowth"]
    )

    df["Score_LowVol_Pullback"] = (
        0.30 * df["r_ROE"] +
        0.30 * df["r_MDD"] +
        0.25 * df["r_Vol"] +
        0.15 * df["r_PE"]
    )

    return df

def top_table(df, score_col, top_n=10):
    cols = [
        "Ticker", "ROE", "MDD", "Momentum_6M", "Dist_200MA",
        "Volatility_1Y", "MarketCap_B", "ForwardPE", "TrailingPE",
        "RevenueGrowth", "GrossMargin", "DebtToEquity", score_col
    ]
    cols = [c for c in cols if c in df.columns]
    out = df.sort_values(score_col, ascending=False)[cols].head(top_n).copy()
    return out

# =========================================================
# 4) Load data
# =========================================================
if refresh:
    st.cache_data.clear()

with st.spinner("Loading Nasdaq-100 data..."):
    price_data = load_price_data(universe, period=period)
    funda = load_fundamentals(universe)
    tech = compute_mdd_and_momentum(price_data, universe)

df = funda.merge(tech, on="Ticker", how="inner")

# basic filters
df = df[
    (df["MarketCap_B"].fillna(0) >= min_mktcap_b) &
    (df["ROE"].fillna(-999) >= min_roe)
].copy()

df = build_scores(df)

st.subheader("Filtered Universe")
st.write(f"Number of stocks after filter: **{len(df)}**")

# =========================================================
# 5) Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "ROE + MDD",
    "Quality + Pullback",
    "Recovery Momentum",
    "Low Vol Pullback"
])

with tab1:
    st.markdown("""
    ### Overview
    This dashboard uses the **Nasdaq-100 universe** and compares several quant styles for finding
    **undervalued but fundamentally strong stocks**.

    **Main idea**
    - **ROE**: quality / profitability
    - **MDD**: how much the stock is down from its previous peak
    - **Momentum**: whether recovery has started
    - **Volatility**: whether price behavior is relatively stable
    """)

    best = top_table(df, "Score_ROE_MDD", top_n)
    st.dataframe(best, use_container_width=True)

    fig = px.scatter(
        df,
        x="MDD",
        y="ROE",
        size="MarketCap_B",
        color="Score_ROE_MDD",
        hover_name="Ticker",
        title="Nasdaq-100: ROE vs MDD"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    ### Strategy 1: ROE + MDD
    **Goal:** Find high-quality companies that are currently in a significant drawdown.

    **Interpretation**
    - High **ROE** = good business quality
    - Deep **MDD** = possibly discounted price

    Good for investors looking for **quality on sale**.
    """)

    out = top_table(df, "Score_ROE_MDD", top_n)
    st.dataframe(out, use_container_width=True)

    fig = px.scatter(
        df.sort_values("Score_ROE_MDD", ascending=False),
        x="MDD",
        y="ROE",
        color="Score_ROE_MDD",
        hover_name="Ticker",
        title="ROE + MDD Strategy"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    ### Strategy 2: Quality + Pullback
    **Goal:** Prefer strong businesses with healthy margins, reasonable balance sheet,
    and a meaningful pullback from prior highs.

    **Uses**
    - ROE
    - Gross Margin
    - Operating Margin
    - Debt to Equity
    - MDD

    This is a more **fundamental quality-focused** version.
    """)

    out = top_table(df, "Score_Quality_Pullback", top_n)
    st.dataframe(out, use_container_width=True)

    fig = px.bar(
        out,
        x="Ticker",
        y="Score_Quality_Pullback",
        title="Top Nasdaq-100 Stocks: Quality + Pullback"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("""
    ### Strategy 3: Recovery Momentum
    **Goal:** Find stocks that were hit hard, but are showing signs of recovery.

    **Uses**
    - ROE
    - MDD
    - 6M Momentum
    - Revenue Growth

    This helps avoid stocks that are simply cheap for a bad reason.
    """)

    out = top_table(df, "Score_Recovery_Momentum", top_n)
    st.dataframe(out, use_container_width=True)

    fig = px.scatter(
        df,
        x="MDD",
        y="Momentum_6M",
        color="Score_Recovery_Momentum",
        size="MarketCap_B",
        hover_name="Ticker",
        title="Recovery Momentum Strategy"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("""
    ### Strategy 4: Low Vol Pullback
    **Goal:** Find quality stocks in drawdown, but with relatively lower volatility.

    **Uses**
    - ROE
    - MDD
    - Volatility
    - PE

    Useful when you want a more **stable pullback strategy**.
    """)

    out = top_table(df, "Score_LowVol_Pullback", top_n)
    st.dataframe(out, use_container_width=True)

    fig = px.scatter(
        df,
        x="Volatility_1Y",
        y="MDD",
        color="Score_LowVol_Pullback",
        hover_name="Ticker",
        title="Low Volatility Pullback Strategy"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6) Raw data
# =========================================================
with st.expander("Show raw merged data"):
    st.dataframe(df.sort_values("Score_ROE_MDD", ascending=False), use_container_width=True)
