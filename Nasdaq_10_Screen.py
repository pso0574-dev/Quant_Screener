# streamlit_app.py
# NASDAQ Quant Screener
# Horizontal main tabs + MVA (Moving Average Analysis)

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="NASDAQ Quant Screener with MVA",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NASDAQ Quant Screener")
st.caption("Horizontal tabs + MVA-based price valuation analysis")

# =========================================================
# Universe
# =========================================================
CANDIDATES = [
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOG", "GOOGL", "META", "AVGO",
    "TSLA", "COST", "NFLX", "AMD", "ADBE", "CSCO", "INTC", "QCOM",
    "TXN", "AMGN", "INTU", "BKNG", "HON", "SBUX", "ADI", "MU"
]

DEFAULT_TOP_N = 10

# =========================================================
# Sidebar settings
# =========================================================
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N by Market Cap", 5, 15, DEFAULT_TOP_N, 1)
refresh_btn = st.sidebar.button("🔄 Refresh Market Data", use_container_width=True)
show_only_undervalued = st.sidebar.checkbox("Show only Undervalued Quality", value=False)

# =========================================================
# Helper functions
# =========================================================
def safe_get(d: dict, key: str, default=np.nan):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default


def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)

    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        out = pd.Series(np.full(len(s), 50.0), index=s.index)
    else:
        out = 100 * (s - s_min) / (s_max - s_min)

    return out if higher_is_better else (100 - out)


def median_fill(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    med = s.median(skipna=True)
    if pd.isna(med):
        med = 0.0
    return s.fillna(med)


@st.cache_data(ttl=1800, show_spinner=False)
def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        return hist
    except Exception:
        return pd.DataFrame()


def add_moving_averages(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return hist

    df = hist.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    df["Dev_MA20"] = safe_div_series(df["Close"] - df["MA20"], df["MA20"])
    df["Dev_MA50"] = safe_div_series(df["Close"] - df["MA50"], df["MA50"])
    df["Dev_MA200"] = safe_div_series(df["Close"] - df["MA200"], df["MA200"])

    return df


def safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def get_history_metrics(ticker: str, period: str = "1y") -> dict:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return {
                "price": np.nan,
                "ret_1m": np.nan,
                "ret_3m": np.nan,
                "ret_6m": np.nan,
                "ret_12m": np.nan,
                "vol_1y": np.nan,
                "mdd_1y": np.nan,
                "dist_from_52w_high": np.nan,
                "dist_from_200dma": np.nan,
            }

        close = hist["Close"].dropna()
        price = close.iloc[-1]

        def pct_return(days: int):
            if len(close) <= days:
                return np.nan
            base = close.iloc[-days - 1]
            if base == 0:
                return np.nan
            return price / base - 1.0

        ret_1m = pct_return(21)
        ret_3m = pct_return(63)
        ret_6m = pct_return(126)
        ret_12m = pct_return(252)

        daily_ret = close.pct_change().dropna()
        vol_1y = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 10 else np.nan

        roll_max = close.cummax()
        drawdown = close / roll_max - 1.0
        mdd_1y = drawdown.min() if not drawdown.empty else np.nan

        high_52w = close.max()
        dist_from_52w_high = price / high_52w - 1.0 if high_52w > 0 else np.nan

        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        dist_from_200dma = price / ma200 - 1.0 if pd.notna(ma200) and ma200 > 0 else np.nan

        return {
            "price": price,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "vol_1y": vol_1y,
            "mdd_1y": mdd_1y,
            "dist_from_52w_high": dist_from_52w_high,
            "dist_from_200dma": dist_from_200dma,
        }
    except Exception:
        return {
            "price": np.nan,
            "ret_1m": np.nan,
            "ret_3m": np.nan,
            "ret_6m": np.nan,
            "ret_12m": np.nan,
            "vol_1y": np.nan,
            "mdd_1y": np.nan,
            "dist_from_52w_high": np.nan,
            "dist_from_200dma": np.nan,
        }


def get_mva_metrics(ticker: str) -> dict:
    hist = get_price_history(ticker, period="2y")
    if hist.empty or len(hist) < 210:
        return {
            "ma20": np.nan,
            "ma50": np.nan,
            "ma200": np.nan,
            "dev_ma20": np.nan,
            "dev_ma50": np.nan,
            "dev_ma200": np.nan,
            "discount_52w_high": np.nan,
            "mva_score": np.nan,
            "mva_label": "Insufficient Data"
        }

    hist = add_moving_averages(hist)

    last = hist.iloc[-1]
    price = last["Close"]
    ma20 = last["MA20"]
    ma50 = last["MA50"]
    ma200 = last["MA200"]

    dev_ma20 = safe_div(price - ma20, ma20)
    dev_ma50 = safe_div(price - ma50, ma50)
    dev_ma200 = safe_div(price - ma200, ma200)

    high_52w = hist["Close"].tail(252).max()
    discount_52w_high = safe_div(price - high_52w, high_52w)

    # MVA score logic:
    # More negative deviation from MA200 and 52W high = relatively cheaper
    # Too far below MA200 can still be risky, so keep score moderate
    score = 50.0

    if pd.notna(dev_ma200):
        if dev_ma200 <= -0.20:
            score += 25
        elif dev_ma200 <= -0.10:
            score += 18
        elif dev_ma200 <= -0.05:
            score += 10
        elif dev_ma200 >= 0.20:
            score -= 20
        elif dev_ma200 >= 0.10:
            score -= 12
        elif dev_ma200 >= 0.05:
            score -= 6

    if pd.notna(discount_52w_high):
        if discount_52w_high <= -0.25:
            score += 20
        elif discount_52w_high <= -0.15:
            score += 12
        elif discount_52w_high <= -0.08:
            score += 6
        elif discount_52w_high >= -0.03:
            score -= 8

    if pd.notna(dev_ma50):
        if -0.08 <= dev_ma50 <= 0.03:
            score += 8
        elif dev_ma50 >= 0.10:
            score -= 5

    score = max(0, min(100, score))

    if score >= 70:
        label = "Potentially Undervalued"
    elif score >= 45:
        label = "Fairly Priced"
    else:
        label = "Extended / Not Cheap"

    return {
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "dev_ma20": dev_ma20,
        "dev_ma50": dev_ma50,
        "dev_ma200": dev_ma200,
        "discount_52w_high": discount_52w_high,
        "mva_score": score,
        "mva_label": label
    }


def get_fundamental_snapshot(ticker: str) -> dict:
    tk = yf.Ticker(ticker)

    info = {}
    try:
        info = tk.info
    except Exception:
        info = {}

    market_cap = np.nan
    try:
        market_cap = tk.fast_info.get("market_cap", np.nan)
    except Exception:
        pass

    if pd.isna(market_cap):
        market_cap = safe_get(info, "marketCap", np.nan)

    data = {
        "ticker": ticker,
        "short_name": safe_get(info, "shortName", ticker),
        "sector": safe_get(info, "sector", ""),
        "industry": safe_get(info, "industry", ""),
        "market_cap": market_cap,
        "trailing_pe": safe_get(info, "trailingPE", np.nan),
        "forward_pe": safe_get(info, "forwardPE", np.nan),
        "peg_ratio": safe_get(info, "pegRatio", np.nan),
        "price_to_book": safe_get(info, "priceToBook", np.nan),
        "enterprise_to_revenue": safe_get(info, "enterpriseToRevenue", np.nan),
        "ev_to_ebitda": safe_get(info, "enterpriseToEbitda", np.nan),
        "gross_margin": safe_get(info, "grossMargins", np.nan),
        "operating_margin": safe_get(info, "operatingMargins", np.nan),
        "profit_margin": safe_get(info, "profitMargins", np.nan),
        "roe": safe_get(info, "returnOnEquity", np.nan),
        "roa": safe_get(info, "returnOnAssets", np.nan),
        "revenue_growth": safe_get(info, "revenueGrowth", np.nan),
        "earnings_growth": safe_get(info, "earningsGrowth", np.nan),
        "debt_to_equity": safe_get(info, "debtToEquity", np.nan),
        "current_ratio": safe_get(info, "currentRatio", np.nan),
        "free_cashflow": safe_get(info, "freeCashflow", np.nan),
        "operating_cashflow": safe_get(info, "operatingCashflow", np.nan),
    }

    data.update(get_history_metrics(ticker))
    data.update(get_mva_metrics(ticker))
    return data


def classify_stock(row: pd.Series) -> str:
    v = row.get("value_score", np.nan)
    q = row.get("quality_score", np.nan)
    g = row.get("growth_score", np.nan)
    s = row.get("stability_score", np.nan)

    if pd.isna(v) or pd.isna(q) or pd.isna(g) or pd.isna(s):
        return "Insufficient Data"

    if v >= 70 and q >= 60 and s >= 50:
        return "Undervalued Quality"
    if q >= 75 and g >= 70:
        return "High-Quality Compounder"
    if v >= 70 and g < 45:
        return "Cheap but Slow Growth"
    if g >= 70 and v < 45:
        return "Strong Growth, Expensive"
    if s < 40:
        return "Higher Risk Profile"
    return "Balanced Mega-cap"


def run_quant_screen(top_n: int = 10) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0, text="Loading latest market data...")

    for i, ticker in enumerate(CANDIDATES):
        rows.append(get_fundamental_snapshot(ticker))
        progress.progress((i + 1) / len(CANDIDATES), text=f"Loading {ticker} ...")

    progress.empty()

    df = pd.DataFrame(rows)

    if "GOOG" in df["ticker"].values and "GOOGL" in df["ticker"].values:
        goog_mc = df.loc[df["ticker"] == "GOOG", "market_cap"].iloc[0]
        googl_mc = df.loc[df["ticker"] == "GOOGL", "market_cap"].iloc[0]
        drop_ticker = "GOOGL" if goog_mc >= googl_mc else "GOOG"
        df = df[df["ticker"] != drop_ticker].copy()

    df = df.sort_values("market_cap", ascending=False).head(top_n).reset_index(drop=True)

    df["fcf_yield"] = df.apply(lambda r: safe_div(r["free_cashflow"], r["market_cap"]), axis=1)
    df["ocf_yield"] = df.apply(lambda r: safe_div(r["operating_cashflow"], r["market_cap"]), axis=1)

    scoring_df = df.copy()
    score_cols = [
        "forward_pe", "peg_ratio", "price_to_book", "enterprise_to_revenue", "ev_to_ebitda",
        "gross_margin", "operating_margin", "profit_margin", "roe", "roa",
        "revenue_growth", "earnings_growth", "debt_to_equity", "current_ratio",
        "fcf_yield", "ocf_yield", "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "vol_1y", "mdd_1y", "dist_from_52w_high", "dist_from_200dma"
    ]
    for col in score_cols:
        scoring_df[col] = median_fill(scoring_df[col])

    value_parts = pd.DataFrame({
        "forward_pe": minmax_score(scoring_df["forward_pe"], higher_is_better=False),
        "peg_ratio": minmax_score(scoring_df["peg_ratio"], higher_is_better=False),
        "price_to_book": minmax_score(scoring_df["price_to_book"], higher_is_better=False),
        "enterprise_to_revenue": minmax_score(scoring_df["enterprise_to_revenue"], higher_is_better=False),
        "ev_to_ebitda": minmax_score(scoring_df["ev_to_ebitda"], higher_is_better=False),
        "fcf_yield": minmax_score(scoring_df["fcf_yield"], higher_is_better=True),
        "ocf_yield": minmax_score(scoring_df["ocf_yield"], higher_is_better=True),
    })
    df["value_score"] = value_parts.mean(axis=1)

    quality_parts = pd.DataFrame({
        "gross_margin": minmax_score(scoring_df["gross_margin"], higher_is_better=True),
        "operating_margin": minmax_score(scoring_df["operating_margin"], higher_is_better=True),
        "profit_margin": minmax_score(scoring_df["profit_margin"], higher_is_better=True),
        "roe": minmax_score(scoring_df["roe"], higher_is_better=True),
        "roa": minmax_score(scoring_df["roa"], higher_is_better=True),
    })
    df["quality_score"] = quality_parts.mean(axis=1)

    growth_parts = pd.DataFrame({
        "revenue_growth": minmax_score(scoring_df["revenue_growth"], higher_is_better=True),
        "earnings_growth": minmax_score(scoring_df["earnings_growth"], higher_is_better=True),
    })
    df["growth_score"] = growth_parts.mean(axis=1)

    stability_parts = pd.DataFrame({
        "debt_to_equity": minmax_score(scoring_df["debt_to_equity"], higher_is_better=False),
        "current_ratio": minmax_score(scoring_df["current_ratio"], higher_is_better=True),
        "vol_1y": minmax_score(scoring_df["vol_1y"], higher_is_better=False),
        "mdd_1y": minmax_score(scoring_df["mdd_1y"], higher_is_better=True),
    })
    df["stability_score"] = stability_parts.mean(axis=1)

    price_parts = pd.DataFrame({
        "ret_1m": minmax_score(scoring_df["ret_1m"], higher_is_better=True),
        "ret_3m": minmax_score(scoring_df["ret_3m"], higher_is_better=True),
        "ret_6m": minmax_score(scoring_df["ret_6m"], higher_is_better=True),
        "ret_12m": minmax_score(scoring_df["ret_12m"], higher_is_better=True),
        "dist_from_200dma": minmax_score(scoring_df["dist_from_200dma"], higher_is_better=True),
        "dist_from_52w_high": minmax_score(scoring_df["dist_from_52w_high"], higher_is_better=True),
    })
    df["price_score"] = price_parts.mean(axis=1)

    df["quant_score"] = (
        0.35 * df["value_score"] +
        0.25 * df["quality_score"] +
        0.20 * df["growth_score"] +
        0.10 * df["stability_score"] +
        0.10 * df["price_score"]
    )

    df["style_label"] = df.apply(classify_stock, axis=1)
    df = df.sort_values("quant_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return df


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = [
        "gross_margin", "operating_margin", "profit_margin", "roe", "roa",
        "revenue_growth", "earnings_growth",
        "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "vol_1y", "mdd_1y", "dist_from_52w_high", "dist_from_200dma",
        "fcf_yield", "ocf_yield",
        "dev_ma20", "dev_ma50", "dev_ma200", "discount_52w_high"
    ]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col] * 100

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(2)
    return out


# =========================================================
# Session state
# =========================================================
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None

# =========================================================
# Refresh
# =========================================================
if refresh_btn or st.session_state.result_df is None:
    with st.spinner("Fetching latest market data and MVA metrics..."):
        result_df = run_quant_screen(top_n=top_n)
        st.session_state.result_df = result_df
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

raw_df = st.session_state.result_df.copy()
display_df = format_display_df(raw_df)

filtered_df = display_df.copy()
if show_only_undervalued:
    filtered_df = filtered_df[filtered_df["style_label"] == "Undervalued Quality"].reset_index(drop=True)

# =========================================================
# Top summary
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe Candidates", len(CANDIDATES))
c2.metric("Top N Selected", len(raw_df))
c3.metric("Displayed Rows", len(filtered_df))
c4.metric("Last Update", st.session_state.last_update if st.session_state.last_update else "-")

# =========================================================
# Horizontal tabs on main screen
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Market Cap Top10",
    "Quant Analysis",
    "Factor Analysis",
    "Single Stock Analysis"
])

# =========================================================
# Tab 1: Dashboard
# =========================================================
with tab1:
    st.subheader("Dashboard Overview")

    top_cols = [
        "rank", "ticker", "short_name", "market_cap", "price",
        "quant_score", "style_label", "mva_score", "mva_label"
    ]
    st.dataframe(filtered_df[top_cols], use_container_width=True, hide_index=True)

    fig = px.bar(
        filtered_df.sort_values("quant_score", ascending=True),
        x="quant_score",
        y="ticker",
        orientation="h",
        hover_data=["short_name", "style_label", "mva_score", "mva_label"],
        title="Quant Score Ranking"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Tab 2: Market Cap Top10
# =========================================================
with tab2:
    st.subheader("Latest Market Cap Ranking")

    mc_view = filtered_df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    mc_view.index = mc_view.index + 1
    st.dataframe(
        mc_view[["ticker", "short_name", "market_cap", "price", "quant_score", "mva_score", "mva_label"]],
        use_container_width=True
    )

    fig_mc = px.bar(
        mc_view,
        x="market_cap",
        y="ticker",
        orientation="h",
        title="Market Cap Ranking"
    )
    fig_mc.update_layout(height=500)
    st.plotly_chart(fig_mc, use_container_width=True)

# =========================================================
# Tab 3: Quant Analysis
# =========================================================
with tab3:
    st.subheader("Quant Analysis")

    qtab1, qtab2, qtab3, qtab4, qtab5, qtab6 = st.tabs([
        "Value", "Quality", "Growth", "Stability", "Momentum", "MVA"
    ])

    with qtab1:
        fig_val = px.scatter(
            filtered_df,
            x="forward_pe",
            y="peg_ratio",
            size="market_cap",
            text="ticker",
            hover_data=["quant_score", "style_label"],
            title="Forward PE vs PEG Ratio"
        )
        fig_val.update_traces(textposition="top center")
        st.plotly_chart(fig_val, use_container_width=True)

    with qtab2:
        fig_q = px.bar(
            filtered_df,
            x="ticker",
            y="roe",
            title="ROE (%)"
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with qtab3:
        fig_g = px.bar(
            filtered_df,
            x="ticker",
            y="revenue_growth",
            title="Revenue Growth (%)"
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with qtab4:
        fig_s = px.bar(
            filtered_df,
            x="ticker",
            y="mdd_1y",
            title="1Y Max Drawdown (%)"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with qtab5:
        fig_m = px.scatter(
            filtered_df,
            x="ret_6m",
            y="ret_12m",
            size="market_cap",
            text="ticker",
            hover_data=["quant_score", "style_label"],
            title="6M Return vs 12M Return"
        )
        fig_m.update_traces(textposition="top center")
        st.plotly_chart(fig_m, use_container_width=True)

    with qtab6:
        st.markdown("### MVA-Based Price Valuation")
        st.markdown(
            """
- **Dev MA50 / MA200 < 0**: current price is below moving average  
- **More negative value** can imply a cheaper entry zone  
- **MVA Score** estimates whether price looks relatively cheap or extended  
            """
        )

        fig_mva = px.scatter(
            filtered_df,
            x="dev_ma200",
            y="discount_52w_high",
            size="market_cap",
            color="mva_score",
            text="ticker",
            hover_data=["price", "mva_score", "mva_label", "dev_ma50"],
            title="Deviation from MA200 vs Discount from 52W High"
        )
        fig_mva.update_traces(textposition="top center")
        st.plotly_chart(fig_mva, use_container_width=True)

        mva_cols = [
            "ticker", "price", "ma20", "ma50", "ma200",
            "dev_ma20", "dev_ma50", "dev_ma200",
            "discount_52w_high", "mva_score", "mva_label"
        ]
        st.dataframe(filtered_df[mva_cols], use_container_width=True, hide_index=True)

# =========================================================
# Tab 4: Factor Analysis
# =========================================================
with tab4:
    st.subheader("Factor Comparison")

    left, right = st.columns(2)

    with left:
        fig1 = px.scatter(
            filtered_df,
            x="value_score",
            y="quality_score",
            size="market_cap",
            text="ticker",
            hover_data=["quant_score", "style_label"],
            title="Value vs Quality"
        )
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        fig2 = px.scatter(
            filtered_df,
            x="growth_score",
            y="stability_score",
            size="market_cap",
            text="ticker",
            hover_data=["quant_score", "style_label"],
            title="Growth vs Stability"
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# Tab 5: Single Stock Analysis
# =========================================================
with tab5:
    st.subheader("Single Stock Analysis")

    ticker_list = display_df["ticker"].tolist() if not display_df.empty else []
    selected_ticker = st.selectbox("Select a ticker", options=ticker_list)

    if selected_ticker:
        hist = get_price_history(selected_ticker, period="2y")
        hist = add_moving_averages(hist)

        if not hist.empty:
            last_row_df = raw_df[raw_df["ticker"] == selected_ticker]
            if not last_row_df.empty:
                stock_info = last_row_df.iloc[0]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"{stock_info['price']:.2f}" if pd.notna(stock_info["price"]) else "-")
                c2.metric("MVA Score", f"{stock_info['mva_score']:.1f}" if pd.notna(stock_info["mva_score"]) else "-")
                c3.metric("MVA Label", stock_info["mva_label"])
                c4.metric("Quant Score", f"{stock_info['quant_score']:.1f}" if pd.notna(stock_info["quant_score"]) else "-")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], mode="lines", name="Close"))
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MA20"], mode="lines", name="MA20"))
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MA50"], mode="lines", name="MA50"))
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MA200"], mode="lines", name="MA200"))

            fig.update_layout(
                title=f"{selected_ticker} Price with Moving Averages",
                height=550,
                xaxis_title="Date",
                yaxis_title="Price"
            )
            st.plotly_chart(fig, use_container_width=True)

            hist["Dev_MA50_pct"] = hist["Dev_MA50"] * 100
            hist["Dev_MA200_pct"] = hist["Dev_MA200"] * 100

            fig_dev = go.Figure()
            fig_dev.add_trace(go.Scatter(x=hist["Date"], y=hist["Dev_MA50_pct"], mode="lines", name="Price vs MA50 (%)"))
            fig_dev.add_trace(go.Scatter(x=hist["Date"], y=hist["Dev_MA200_pct"], mode="lines", name="Price vs MA200 (%)"))
            fig_dev.add_hline(y=0, line_dash="dash")
            fig_dev.update_layout(
                title=f"{selected_ticker} Price Deviation from Moving Averages",
                height=450,
                xaxis_title="Date",
                yaxis_title="Deviation (%)"
            )
            st.plotly_chart(fig_dev, use_container_width=True)

            st.markdown("### MVA Interpretation")
            st.markdown(
                """
- **Below MA200** and **far from 52W high**: can indicate a relatively cheaper zone  
- **Far above MA50 / MA200**: often means the stock is extended, not cheap  
- **MVA Score is not intrinsic valuation**  
  it is a **price-location indicator** to judge whether entry timing looks expensive or discounted
                """
            )

            latest = hist.iloc[-1]
            summary_df = pd.DataFrame({
                "Metric": [
                    "Current Price",
                    "MA20",
                    "MA50",
                    "MA200",
                    "Price vs MA50 (%)",
                    "Price vs MA200 (%)"
                ],
                "Value": [
                    round(latest["Close"], 2) if pd.notna(latest["Close"]) else np.nan,
                    round(latest["MA20"], 2) if pd.notna(latest["MA20"]) else np.nan,
                    round(latest["MA50"], 2) if pd.notna(latest["MA50"]) else np.nan,
                    round(latest["MA200"], 2) if pd.notna(latest["MA200"]) else np.nan,
                    round(latest["Dev_MA50"] * 100, 2) if pd.notna(latest["Dev_MA50"]) else np.nan,
                    round(latest["Dev_MA200"] * 100, 2) if pd.notna(latest["Dev_MA200"]) else np.nan,
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Price history could not be loaded.")

# =========================================================
# Download
# =========================================================
st.markdown("---")
csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Download Current Table as CSV",
    data=csv_data,
    file_name="nasdaq_quant_mva_screener.csv",
    mime="text/csv",
    use_container_width=True
)
