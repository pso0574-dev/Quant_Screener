# streamlit_app.py
# NASDAQ Quant Screener
# Horizontal main tabs + MVA + Relative Price + MDD
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# Run:
#   streamlit run streamlit_app.py

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="NASDAQ Quant Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NASDAQ Quant Screener")
st.caption("Horizontal tabs + MVA + Relative Price + MDD analysis")

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
# Sidebar
# =========================================================
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N by Market Cap", 5, 15, DEFAULT_TOP_N, 1)
refresh_btn = st.sidebar.button("🔄 Refresh Market Data", use_container_width=True)
show_only_undervalued = st.sidebar.checkbox("Show only Undervalued Quality", value=False)
chart_period = st.sidebar.selectbox("Chart Period", ["6mo", "1y", "2y"], index=1)

# =========================================================
# Helpers
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


def safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)

    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        out = pd.Series(np.full(len(s), 50.0), index=s.index)
    else:
        out = 100.0 * (s - s_min) / (s_max - s_min)

    return out if higher_is_better else (100.0 - out)


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
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
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

    roll_max = df["Close"].cummax()
    df["Drawdown"] = safe_div_series(df["Close"], roll_max) - 1.0

    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_multi_price_history(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    try:
        data = yf.download(
            list(tickers),
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True
        )

        if data.empty:
            return pd.DataFrame()

        frames = []

        # Multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    sub = data[ticker].copy().reset_index()
                    if "Close" not in sub.columns:
                        continue
                    sub = sub[["Date", "Close"]].copy()
                    sub["Ticker"] = ticker
                    sub["Date"] = pd.to_datetime(sub["Date"]).dt.tz_localize(None)
                    frames.append(sub)
                except Exception:
                    continue
        else:
            # Single ticker fallback
            sub = data.reset_index()[["Date", "Close"]].copy()
            sub["Ticker"] = tickers[0]
            sub["Date"] = pd.to_datetime(sub["Date"]).dt.tz_localize(None)
            frames.append(sub)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        return out

    except Exception:
        return pd.DataFrame()


def build_relative_price_df(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return price_df

    frames = []
    for ticker, grp in price_df.groupby("Ticker"):
        g = grp.sort_values("Date").copy()
        first_close = g["Close"].dropna()
        if first_close.empty:
            continue
        base = first_close.iloc[0]
        if pd.isna(base) or base == 0:
            continue

        g["RelativePrice"] = g["Close"] / base * 100.0
        g["RollingMax"] = g["Close"].cummax()
        g["Drawdown"] = g["Close"] / g["RollingMax"] - 1.0
        frames.append(g)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


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
        if close.empty:
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

        price = float(close.iloc[-1])

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
            out[col] = out[col] * 100.0

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
    with st.spinner("Fetching latest market data and factor metrics..."):
        result_df = run_quant_screen(top_n=top_n)
        st.session_state.result_df = result_df
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

raw_df = st.session_state.result_df.copy()
display_df = format_display_df(raw_df)

filtered_raw_df = raw_df.copy()
filtered_display_df = display_df.copy()

if show_only_undervalued:
    mask = filtered_raw_df["style_label"] == "Undervalued Quality"
    filtered_raw_df = filtered_raw_df[mask].reset_index(drop=True)
    filtered_display_df = filtered_display_df[mask].reset_index(drop=True)

# =========================================================
# Summary
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe Candidates", len(CANDIDATES))
c2.metric("Top N Selected", len(raw_df))
c3.metric("Displayed Rows", len(filtered_display_df))
c4.metric("Last Update", st.session_state.last_update if st.session_state.last_update else "-")

# =========================================================
# Main Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard",
    "Market Cap Top10",
    "Quant Analysis",
    "Factor Analysis",
    "Relative Price",
    "MDD Analysis",
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
    st.dataframe(filtered_display_df[top_cols], use_container_width=True, hide_index=True)

    chart_df = filtered_display_df.dropna(subset=["quant_score"]).copy()
    if chart_df.empty:
        st.warning("No valid data available for dashboard chart.")
    else:
        fig = px.bar(
            chart_df.sort_values("quant_score", ascending=True),
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

    mc_view = filtered_display_df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    mc_view.index = mc_view.index + 1

    st.dataframe(
        mc_view[["ticker", "short_name", "market_cap", "price", "quant_score", "mva_score", "mva_label"]],
        use_container_width=True
    )

    if mc_view.empty:
        st.warning("No valid market cap data available.")
    else:
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
        st.markdown("### Value Analysis")

        value_df = filtered_raw_df.copy()
        needed_cols = ["ticker", "market_cap", "forward_pe", "peg_ratio", "quant_score", "style_label"]
        for col in needed_cols:
            if col not in value_df.columns:
                value_df[col] = np.nan

        value_df = value_df.replace([np.inf, -np.inf], np.nan)
        value_df = value_df.dropna(subset=["forward_pe", "peg_ratio", "market_cap"])

        if value_df.empty:
            st.warning("No valid data available for Value chart. (forward_pe / peg_ratio missing)")
        else:
            fig_val = px.scatter(
                value_df,
                x="forward_pe",
                y="peg_ratio",
                size="market_cap",
                text="ticker",
                hover_data=["quant_score", "style_label"],
                title="Forward PE vs PEG Ratio"
            )
            fig_val.update_traces(textposition="top center")
            fig_val.update_layout(height=500)
            st.plotly_chart(fig_val, use_container_width=True)

            value_table = format_display_df(value_df)
            st.dataframe(
                value_table[["ticker", "forward_pe", "peg_ratio", "quant_score", "style_label"]],
                use_container_width=True,
                hide_index=True
            )

    with qtab2:
        st.markdown("### Quality Analysis")

        quality_df = filtered_raw_df.copy()
        quality_df = quality_df.replace([np.inf, -np.inf], np.nan)
        quality_df = quality_df.dropna(subset=["roe"])

        if quality_df.empty:
            st.warning("No valid data available for Quality chart.")
        else:
            quality_plot = format_display_df(quality_df)
            fig_q = px.bar(
                quality_plot,
                x="ticker",
                y="roe",
                title="ROE (%)"
            )
            fig_q.update_layout(height=500)
            st.plotly_chart(fig_q, use_container_width=True)

    with qtab3:
        st.markdown("### Growth Analysis")

        growth_df = filtered_raw_df.copy()
        growth_df = growth_df.replace([np.inf, -np.inf], np.nan)
        growth_df = growth_df.dropna(subset=["revenue_growth"])

        if growth_df.empty:
            st.warning("No valid data available for Growth chart.")
        else:
            growth_plot = format_display_df(growth_df)
            fig_g = px.bar(
                growth_plot,
                x="ticker",
                y="revenue_growth",
                title="Revenue Growth (%)"
            )
            fig_g.update_layout(height=500)
            st.plotly_chart(fig_g, use_container_width=True)

    with qtab4:
        st.markdown("### Stability Analysis")

        stability_df = filtered_raw_df.copy()
        stability_df = stability_df.replace([np.inf, -np.inf], np.nan)
        stability_df = stability_df.dropna(subset=["mdd_1y"])

        if stability_df.empty:
            st.warning("No valid data available for Stability chart.")
        else:
            stability_plot = format_display_df(stability_df)
            fig_s = px.bar(
                stability_plot,
                x="ticker",
                y="mdd_1y",
                title="1Y Max Drawdown (%)"
            )
            fig_s.update_layout(height=500)
            st.plotly_chart(fig_s, use_container_width=True)

    with qtab5:
        st.markdown("### Momentum Analysis")

        momentum_df = filtered_raw_df.copy()
        needed_cols = ["ticker", "market_cap", "ret_6m", "ret_12m", "quant_score", "style_label"]
        for col in needed_cols:
            if col not in momentum_df.columns:
                momentum_df[col] = np.nan

        momentum_df = momentum_df.replace([np.inf, -np.inf], np.nan)
        momentum_df = momentum_df.dropna(subset=["ret_6m", "ret_12m", "market_cap"])

        if momentum_df.empty:
            st.warning("No valid data available for Momentum chart. (ret_6m / ret_12m missing)")
        else:
            momentum_plot = format_display_df(momentum_df)
            fig_m = px.scatter(
                momentum_plot,
                x="ret_6m",
                y="ret_12m",
                size="market_cap",
                text="ticker",
                hover_data=["quant_score", "style_label"],
                title="6M Return vs 12M Return (%)"
            )
            fig_m.update_traces(textposition="top center")
            fig_m.update_layout(height=500)
            st.plotly_chart(fig_m, use_container_width=True)

            st.dataframe(
                momentum_plot[["ticker", "ret_6m", "ret_12m", "quant_score", "style_label"]],
                use_container_width=True,
                hide_index=True
            )

    with qtab6:
        st.markdown("### MVA-Based Price Valuation")

        mva_df = filtered_raw_df.copy()
        mva_df = mva_df.replace([np.inf, -np.inf], np.nan)
        mva_df = mva_df.dropna(subset=["dev_ma200", "discount_52w_high", "market_cap", "mva_score"])

        if mva_df.empty:
            st.warning("No valid data available for MVA chart.")
        else:
            mva_plot = format_display_df(mva_df)
            fig_mva = px.scatter(
                mva_plot,
                x="dev_ma200",
                y="discount_52w_high",
                size="market_cap",
                color="mva_score",
                text="ticker",
                hover_data=["price", "mva_score", "mva_label", "dev_ma50"],
                title="Deviation from MA200 vs Discount from 52W High (%)"
            )
            fig_mva.update_traces(textposition="top center")
            fig_mva.update_layout(height=550)
            st.plotly_chart(fig_mva, use_container_width=True)

            mva_cols = [
                "ticker", "price", "ma20", "ma50", "ma200",
                "dev_ma20", "dev_ma50", "dev_ma200",
                "discount_52w_high", "mva_score", "mva_label"
            ]
            st.dataframe(mva_plot[mva_cols], use_container_width=True, hide_index=True)

# =========================================================
# Tab 4: Factor Analysis
# =========================================================
with tab4:
    st.subheader("Factor Comparison")

    factor_df = filtered_raw_df.copy().replace([np.inf, -np.inf], np.nan)

    left, right = st.columns(2)

    with left:
        df1 = factor_df.dropna(subset=["value_score", "quality_score", "market_cap"])
        if df1.empty:
            st.warning("No valid data for Value vs Quality chart.")
        else:
            fig1 = px.scatter(
                df1,
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
        df2 = factor_df.dropna(subset=["growth_score", "stability_score", "market_cap"])
        if df2.empty:
            st.warning("No valid data for Growth vs Stability chart.")
        else:
            fig2 = px.scatter(
                df2,
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
# Tab 5: Relative Price
# =========================================================
with tab5:
    st.subheader("Relative Price Comparison")

    tickers_for_chart = tuple(filtered_raw_df["ticker"].tolist())

    if len(tickers_for_chart) == 0:
        st.warning("No tickers available for relative price comparison.")
    else:
        multi_price = get_multi_price_history(tickers_for_chart, period=chart_period)
        rel_df = build_relative_price_df(multi_price)

        if rel_df.empty:
            st.warning("Could not load multi-stock price data.")
        else:
            fig_rel = px.line(
                rel_df,
                x="Date",
                y="RelativePrice",
                color="Ticker",
                title=f"Normalized Price Comparison ({chart_period}, start=100)"
            )
            fig_rel.update_layout(height=600, yaxis_title="Relative Price (Start = 100)")
            st.plotly_chart(fig_rel, use_container_width=True)

            last_rel = (
                rel_df.sort_values("Date")
                .groupby("Ticker", as_index=False)
                .tail(1)[["Ticker", "RelativePrice", "Drawdown"]]
                .copy()
            )
            last_rel["Drawdown"] = (last_rel["Drawdown"] * 100).round(2)
            last_rel["RelativePrice"] = last_rel["RelativePrice"].round(2)

            st.markdown("### Current Relative Position")
            st.dataframe(
                last_rel.sort_values("RelativePrice"),
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                """
- **Relative Price = 100** means same as starting point  
- **Below 100** means the stock is below its own starting-period level  
- This helps identify which top market-cap stocks have fallen more, relatively
                """
            )

# =========================================================
# Tab 6: MDD Analysis
# =========================================================
with tab6:
    st.subheader("MDD Analysis")

    mdd_tab1, mdd_tab2 = st.tabs(["Cross-Section MDD", "Drawdown Curve"])

    with mdd_tab1:
        mdd_df = filtered_raw_df.copy().replace([np.inf, -np.inf], np.nan)
        mdd_df = mdd_df.dropna(subset=["mdd_1y"])

        if mdd_df.empty:
            st.warning("No valid MDD data available.")
        else:
            mdd_plot = format_display_df(mdd_df)
            fig_mdd = px.bar(
                mdd_plot.sort_values("mdd_1y"),
                x="mdd_1y",
                y="ticker",
                orientation="h",
                title="1Y Maximum Drawdown Comparison (%)"
            )
            fig_mdd.update_layout(height=550, xaxis_title="MDD (%)")
            st.plotly_chart(fig_mdd, use_container_width=True)

            st.dataframe(
                mdd_plot[["ticker", "price", "ret_6m", "ret_12m", "mdd_1y", "quant_score", "mva_score"]],
                use_container_width=True,
                hide_index=True
            )

    with mdd_tab2:
        ticker_list = filtered_raw_df["ticker"].tolist()
        if not ticker_list:
            st.warning("No ticker available.")
        else:
            selected_dd_ticker = st.selectbox(
                "Select a ticker for drawdown curve",
                options=ticker_list,
                key="drawdown_curve_ticker"
            )

            dd_hist = get_price_history(selected_dd_ticker, period="2y")
            dd_hist = add_moving_averages(dd_hist)

            if dd_hist.empty or "Drawdown" not in dd_hist.columns:
                st.warning("Could not load drawdown curve.")
            else:
                dd_plot = dd_hist.copy()
                dd_plot["DrawdownPct"] = dd_plot["Drawdown"] * 100.0

                fig_dd = px.line(
                    dd_plot,
                    x="Date",
                    y="DrawdownPct",
                    title=f"{selected_dd_ticker} Drawdown Curve (%)"
                )
                fig_dd.update_layout(height=500, yaxis_title="Drawdown (%)")
                st.plotly_chart(fig_dd, use_container_width=True)

                st.dataframe(
                    pd.DataFrame({
                        "Metric": ["Current Drawdown (%)", "Worst Drawdown (%)"],
                        "Value": [
                            round(dd_plot["DrawdownPct"].iloc[-1], 2),
                            round(dd_plot["DrawdownPct"].min(), 2)
                        ]
                    }),
                    use_container_width=True,
                    hide_index=True
                )

# =========================================================
# Tab 7: Single Stock Analysis
# =========================================================
with tab7:
    st.subheader("Single Stock Analysis")

    ticker_list = filtered_raw_df["ticker"].tolist() if not filtered_raw_df.empty else []
    selected_ticker = st.selectbox("Select a ticker", options=ticker_list, key="single_stock_ticker")

    if selected_ticker:
        hist = get_price_history(selected_ticker, period="2y")
        hist = add_moving_averages(hist)

        if not hist.empty:
            last_row_df = filtered_raw_df[filtered_raw_df["ticker"] == selected_ticker]
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

            hist["DrawdownPct"] = hist["Drawdown"] * 100.0
            fig_dd_single = px.line(
                hist,
                x="Date",
                y="DrawdownPct",
                title=f"{selected_ticker} Drawdown Curve (%)"
            )
            fig_dd_single.update_layout(height=420, yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd_single, use_container_width=True)

            latest = hist.iloc[-1]
            summary_df = pd.DataFrame({
                "Metric": [
                    "Current Price",
                    "MA20",
                    "MA50",
                    "MA200",
                    "Price vs MA50 (%)",
                    "Price vs MA200 (%)",
                    "Current Drawdown (%)",
                    "Worst Drawdown (%)"
                ],
                "Value": [
                    round(latest["Close"], 2) if pd.notna(latest["Close"]) else np.nan,
                    round(latest["MA20"], 2) if pd.notna(latest["MA20"]) else np.nan,
                    round(latest["MA50"], 2) if pd.notna(latest["MA50"]) else np.nan,
                    round(latest["MA200"], 2) if pd.notna(latest["MA200"]) else np.nan,
                    round(latest["Dev_MA50"] * 100, 2) if pd.notna(latest["Dev_MA50"]) else np.nan,
                    round(latest["Dev_MA200"] * 100, 2) if pd.notna(latest["Dev_MA200"]) else np.nan,
                    round(latest["Drawdown"] * 100, 2) if pd.notna(latest["Drawdown"]) else np.nan,
                    round(hist["Drawdown"].min() * 100, 2) if pd.notna(hist["Drawdown"].min()) else np.nan,
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Price history could not be loaded.")

# =========================================================
# Download
# =========================================================
st.markdown("---")
csv_data = filtered_display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Download Current Table as CSV",
    data=csv_data,
    file_name="nasdaq_quant_mva_mdd_screener.csv",
    mime="text/csv",
    use_container_width=True
)
