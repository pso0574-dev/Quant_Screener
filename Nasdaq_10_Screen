# streamlit_app.py
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NASDAQ Top 10 Quant Screener", page_icon="📈", layout="wide")

st.title("📈 NASDAQ Top 10 Quant Screener")
st.caption("Find undervalued high-quality mega-cap stocks")

# =========================================================
# Candidate universe
# =========================================================
CANDIDATES = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOG",
    "META", "AVGO", "TSLA", "COST", "NFLX",
    "AMD", "PEP", "ADBE", "CSCO", "INTC", "QCOM", "TXN"
]


# =========================================================
# Helper functions
# =========================================================
def safe_get(d, key, default=np.nan):
    try:
        val = d.get(key, default)
        if val is None:
            return default
        return val
    except Exception:
        return default


def minmax_score(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)

    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        out = pd.Series(np.full(len(s), 50.0), index=s.index)
    else:
        out = 100 * (s - s_min) / (s_max - s_min)

    return out if higher_is_better else (100 - out)


def get_history_metrics(ticker, period="1y"):
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return {
                "price": np.nan,
                "ret_6m": np.nan,
                "ret_12m": np.nan,
                "vol_1y": np.nan,
                "mdd_1y": np.nan,
                "dist_from_52w_high": np.nan,
                "dist_from_200dma": np.nan
            }

        close = hist["Close"].dropna()
        price = close.iloc[-1]

        def pct_return(days):
            if len(close) <= days:
                return np.nan
            return price / close.iloc[-days - 1] - 1

        ret_6m = pct_return(126)
        ret_12m = pct_return(252)

        daily_ret = close.pct_change().dropna()
        vol_1y = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 10 else np.nan

        roll_max = close.cummax()
        drawdown = close / roll_max - 1
        mdd_1y = drawdown.min() if len(drawdown) > 0 else np.nan

        high_52w = close.max()
        dist_from_52w_high = price / high_52w - 1 if high_52w > 0 else np.nan

        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        dist_from_200dma = price / ma200 - 1 if pd.notna(ma200) and ma200 > 0 else np.nan

        return {
            "price": price,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "vol_1y": vol_1y,
            "mdd_1y": mdd_1y,
            "dist_from_52w_high": dist_from_52w_high,
            "dist_from_200dma": dist_from_200dma
        }
    except Exception:
        return {
            "price": np.nan,
            "ret_6m": np.nan,
            "ret_12m": np.nan,
            "vol_1y": np.nan,
            "mdd_1y": np.nan,
            "dist_from_52w_high": np.nan,
            "dist_from_200dma": np.nan
        }


def get_fundamental_snapshot(ticker):
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
        "market_cap": market_cap,
        "trailing_pe": safe_get(info, "trailingPE", np.nan),
        "forward_pe": safe_get(info, "forwardPE", np.nan),
        "peg_ratio": safe_get(info, "pegRatio", np.nan),
        "price_to_book": safe_get(info, "priceToBook", np.nan),
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
        "short_name": safe_get(info, "shortName", ticker),
    }

    data.update(get_history_metrics(ticker))
    return data


def label_stock(row):
    v = row["value_score"]
    q = row["quality_score"]
    g = row["growth_score"]
    s = row["stability_score"]

    if pd.isna(row["quant_score"]):
        return "Insufficient data"
    if v >= 70 and q >= 60:
        return "Undervalued Quality"
    if q >= 70 and g >= 70:
        return "High-Quality Compounder"
    if v >= 70 and g < 45:
        return "Cheap but Weak Growth"
    if g >= 70 and v < 45:
        return "Great Business, Expensive"
    if s < 40:
        return "Higher Risk Profile"
    return "Balanced Mega-cap"


def run_quant_screen(top_n=10):
    rows = []
    progress = st.progress(0, text="Loading market data...")

    for i, t in enumerate(CANDIDATES):
        rows.append(get_fundamental_snapshot(t))
        progress.progress((i + 1) / len(CANDIDATES), text=f"Loading {t} ...")

    progress.empty()

    df = pd.DataFrame(rows)

    # Market cap descending -> Top N
    df = df.sort_values("market_cap", ascending=False).head(top_n).reset_index(drop=True)

    # Derived ratios
    df["fcf_yield"] = df["free_cashflow"] / df["market_cap"]
    df["ocf_yield"] = df["operating_cashflow"] / df["market_cap"]

    # Scores
    value_parts = pd.DataFrame({
        "forward_pe": minmax_score(df["forward_pe"], higher_is_better=False),
        "peg_ratio": minmax_score(df["peg_ratio"], higher_is_better=False),
        "price_to_book": minmax_score(df["price_to_book"], higher_is_better=False),
        "ev_to_ebitda": minmax_score(df["ev_to_ebitda"], higher_is_better=False),
        "fcf_yield": minmax_score(df["fcf_yield"], higher_is_better=True),
        "ocf_yield": minmax_score(df["ocf_yield"], higher_is_better=True),
    })
    df["value_score"] = value_parts.mean(axis=1, skipna=True)

    quality_parts = pd.DataFrame({
        "gross_margin": minmax_score(df["gross_margin"], higher_is_better=True),
        "operating_margin": minmax_score(df["operating_margin"], higher_is_better=True),
        "profit_margin": minmax_score(df["profit_margin"], higher_is_better=True),
        "roe": minmax_score(df["roe"], higher_is_better=True),
        "roa": minmax_score(df["roa"], higher_is_better=True),
    })
    df["quality_score"] = quality_parts.mean(axis=1, skipna=True)

    growth_parts = pd.DataFrame({
        "revenue_growth": minmax_score(df["revenue_growth"], higher_is_better=True),
        "earnings_growth": minmax_score(df["earnings_growth"], higher_is_better=True),
    })
    df["growth_score"] = growth_parts.mean(axis=1, skipna=True)

    stability_parts = pd.DataFrame({
        "debt_to_equity": minmax_score(df["debt_to_equity"], higher_is_better=False),
        "current_ratio": minmax_score(df["current_ratio"], higher_is_better=True),
        "vol_1y": minmax_score(df["vol_1y"], higher_is_better=False),
        "mdd_1y": minmax_score(df["mdd_1y"], higher_is_better=True),
    })
    df["stability_score"] = stability_parts.mean(axis=1, skipna=True)

    price_parts = pd.DataFrame({
        "ret_6m": minmax_score(df["ret_6m"], higher_is_better=True),
        "ret_12m": minmax_score(df["ret_12m"], higher_is_better=True),
        "dist_from_200dma": minmax_score(df["dist_from_200dma"], higher_is_better=True),
        "dist_from_52w_high": minmax_score(df["dist_from_52w_high"], higher_is_better=True),
    })
    df["price_score"] = price_parts.mean(axis=1, skipna=True)

    df["quant_score"] = (
        0.35 * df["value_score"] +
        0.25 * df["quality_score"] +
        0.20 * df["growth_score"] +
        0.10 * df["stability_score"] +
        0.10 * df["price_score"]
    )

    df["style_label"] = df.apply(label_stock, axis=1)
    df = df.sort_values("quant_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return df


def format_output(df):
    out = df.copy()

    pct_cols = [
        "gross_margin", "operating_margin", "profit_margin", "roe", "roa",
        "revenue_growth", "earnings_growth", "ret_6m", "ret_12m", "mdd_1y",
        "dist_from_52w_high", "dist_from_200dma", "vol_1y", "fcf_yield", "ocf_yield"
    ]
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c] * 100

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(2)
    return out


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N by Market Cap", min_value=5, max_value=15, value=10, step=1)

refresh = st.sidebar.button("🔄 실시간 시총 상위 종목 조회", use_container_width=True)

if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None

# =========================================================
# Main action
# =========================================================
if refresh or st.session_state.result_df is None:
    with st.spinner("Fetching latest market cap and quant data..."):
        result = run_quant_screen(top_n=top_n)
        st.session_state.result_df = result
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df = st.session_state.result_df
out = format_output(df)

# =========================================================
# Metrics
# =========================================================
col1, col2, col3 = st.columns(3)
col1.metric("Universe Candidates", len(CANDIDATES))
col2.metric("Top N Selected", len(out))
col3.metric("Last Update", st.session_state.last_update)

st.markdown("### Top Ranked Stocks")

show_cols = [
    "rank", "ticker", "short_name", "market_cap", "price",
    "forward_pe", "peg_ratio", "price_to_book",
    "revenue_growth", "earnings_growth",
    "roe", "operating_margin",
    "mdd_1y", "quant_score", "style_label"
]

st.dataframe(out[show_cols], use_container_width=True)

st.markdown("### Market Cap Top 10")
marketcap_view = out.sort_values("market_cap", ascending=False)[
    ["ticker", "short_name", "market_cap", "price"]
].reset_index(drop=True)
marketcap_view.index = marketcap_view.index + 1
st.dataframe(marketcap_view, use_container_width=True)

st.markdown("### Full Quant Data")
st.dataframe(out, use_container_width=True)

csv = out.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="nasdaq_top10_quant.csv",
    mime="text/csv",
    use_container_width=True
)
