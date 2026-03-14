# streamlit_app.py
# NASDAQ-100 Quant Screener
# Focus: "Cheap now" based on Drawdown/MDD + ROE + valuation
# Run:
#   pip install streamlit yfinance pandas numpy plotly
#   streamlit run streamlit_app.py

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="NASDAQ Quant Screener",
    page_icon="📉",
    layout="wide"
)

st.title("📉 NASDAQ Quant Screener")
st.caption("Undervaluation view using Drawdown / MDD + ROE + basic valuation")

# =========================================================
# Constants
# =========================================================
# Stable built-in Nasdaq-100 universe to avoid fragile web scraping
NASDAQ100_TICKERS = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN",
    "AMZN","ANSS","APP","ARM","ASML","AVGO","AXON","AZN","BIIB","BKNG",
    "CDNS","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS",
    "CTSH","DASH","DDOG","DXCM","EA","EXC","FANG","FAST","FTNT","GEHC",
    "GFS","GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP",
    "KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META",
    "MNST","MRVL","MSFT","MSTR","MU","NFLX","NVDA","NXPI","ODFL","ON",
    "ORLY","PANW","PAYX","PCAR","PDD","PEP","PLTR","PYPL","QCOM","REGN",
    "ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TTWO","TXN",
    "VRSK","VRTX","WBD","WDAY","XEL","ZS"
]

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AVGO", "NFLX", "TSLA", "AMD"]

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")

universe_mode = st.sidebar.selectbox(
    "Universe",
    ["NASDAQ-100", "Custom List"],
    index=0
)

if universe_mode == "NASDAQ-100":
    universe = NASDAQ100_TICKERS
else:
    custom = st.sidebar.text_area(
        "Custom tickers (comma-separated)",
        value="AAPL,MSFT,GOOGL,META,AMZN,NVDA,AVGO,NFLX,TSLA,AMD"
    )
    universe = [x.strip().upper() for x in custom.split(",") if x.strip()]

history_years = st.sidebar.slider("Price history window (years)", 1, 10, 5)
min_roe = st.sidebar.slider("Minimum ROE (%)", -20, 60, 10)
min_discount = st.sidebar.slider("Minimum discount from ATH (%)", 0, 80, 20)
max_pe = st.sidebar.slider("Maximum PE", 5, 100, 40)
top_n = st.sidebar.slider("Top results", 5, 50, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("Scoring Weights")

w_roe = st.sidebar.slider("ROE weight", 0.0, 1.0, 0.35, 0.05)
w_discount = st.sidebar.slider("Discount/MDD weight", 0.0, 1.0, 0.35, 0.05)
w_pe = st.sidebar.slider("PE weight", 0.0, 1.0, 0.20, 0.05)
w_growth = st.sidebar.slider("Revenue growth weight", 0.0, 1.0, 0.10, 0.05)

total_w = w_roe + w_discount + w_pe + w_growth
if total_w == 0:
    st.sidebar.warning("At least one weight must be greater than 0.")
    st.stop()

w_roe /= total_w
w_discount /= total_w
w_pe /= total_w
w_growth /= total_w

refresh = st.sidebar.button("🔄 Refresh data")

# =========================================================
# Helpers
# =========================================================
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def pct_or_nan(x):
    x = safe_float(x)
    if np.isnan(x):
        return np.nan
    return x * 100.0

def label_cheapness(discount_pct: float, roe_pct: float) -> str:
    if np.isnan(discount_pct):
        return "N/A"
    if discount_pct >= 40 and roe_pct >= 15:
        return "Very Cheap"
    if discount_pct >= 30 and roe_pct >= 12:
        return "Cheap"
    if discount_pct >= 20 and roe_pct >= 10:
        return "Watchlist"
    if discount_pct >= 10:
        return "Normal Pullback"
    return "Near High"

def normalize_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.copy()
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(valid) <= 1:
        out.loc[valid.index] = 50.0
        return out

    ranks = valid.rank(ascending=not higher_is_better, method="average")
    scores = 100.0 * (len(valid) - ranks) / (len(valid) - 1)
    out.loc[valid.index] = scores
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history(tickers: list[str], period_years: int) -> pd.DataFrame:
    period = f"{period_years}y"
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False
    )
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def load_snapshot_data(tickers: list[str]) -> dict:
    result = {}
    for i, t in enumerate(tickers):
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            fi = tk.fast_info if hasattr(tk, "fast_info") else {}
            result[t] = {
                "shortName": info.get("shortName", t),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "marketCap": safe_float(info.get("marketCap")),
                "trailingPE": safe_float(info.get("trailingPE")),
                "forwardPE": safe_float(info.get("forwardPE")),
                "priceToBook": safe_float(info.get("priceToBook")),
                "roe_pct": pct_or_nan(info.get("returnOnEquity")),
                "roa_pct": pct_or_nan(info.get("returnOnAssets")),
                "grossMargins_pct": pct_or_nan(info.get("grossMargins")),
                "operatingMargins_pct": pct_or_nan(info.get("operatingMargins")),
                "profitMargins_pct": pct_or_nan(info.get("profitMargins")),
                "revenueGrowth_pct": pct_or_nan(info.get("revenueGrowth")),
                "debtToEquity": safe_float(info.get("debtToEquity")),
                "currentPrice": safe_float(info.get("currentPrice", fi.get("lastPrice", np.nan))),
            }
        except Exception:
            result[t] = {
                "shortName": t,
                "sector": "",
                "industry": "",
                "marketCap": np.nan,
                "trailingPE": np.nan,
                "forwardPE": np.nan,
                "priceToBook": np.nan,
                "roe_pct": np.nan,
                "roa_pct": np.nan,
                "grossMargins_pct": np.nan,
                "operatingMargins_pct": np.nan,
                "profitMargins_pct": np.nan,
                "revenueGrowth_pct": np.nan,
                "debtToEquity": np.nan,
                "currentPrice": np.nan,
            }
        time.sleep(0.05)
    return result

def get_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            df = data[ticker]
            for c in ["Close", "Adj Close"]:
                if c in df.columns:
                    s = df[c].dropna()
                    if len(s) > 0:
                        return s
    else:
        for c in ["Close", "Adj Close"]:
            if c in data.columns:
                s = data[c].dropna()
                if len(s) > 0:
                    return s
    return pd.Series(dtype=float)

def compute_price_metrics(price_s: pd.Series) -> dict:
    if price_s.empty:
        return {
            "current": np.nan,
            "ath": np.nan,
            "discount_from_ath_pct": np.nan,
            "rolling_mdd_pct": np.nan,
            "mva50_gap_pct": np.nan,
            "mva200_gap_pct": np.nan,
            "ytd_return_pct": np.nan,
            "ret_1y_pct": np.nan,
        }

    current = float(price_s.iloc[-1])
    ath = float(price_s.max())
    discount = (ath - current) / ath * 100 if ath > 0 else np.nan

    running_max = price_s.cummax()
    drawdown = (price_s / running_max - 1.0) * 100.0
    rolling_mdd = float(drawdown.min())

    ma50 = price_s.rolling(50).mean().iloc[-1]
    ma200 = price_s.rolling(200).mean().iloc[-1]
    mva50_gap = (current / ma50 - 1.0) * 100 if pd.notna(ma50) and ma50 != 0 else np.nan
    mva200_gap = (current / ma200 - 1.0) * 100 if pd.notna(ma200) and ma200 != 0 else np.nan

    year_start = pd.Timestamp(datetime(price_s.index[-1].year, 1, 1))
    ytd_base = price_s[price_s.index >= year_start]
    ytd_return = (current / ytd_base.iloc[0] - 1.0) * 100 if len(ytd_base) > 0 else np.nan

    if len(price_s) >= 252:
        ret_1y = (current / price_s.iloc[-252] - 1.0) * 100
    else:
        ret_1y = np.nan

    return {
        "current": current,
        "ath": ath,
        "discount_from_ath_pct": discount,
        "rolling_mdd_pct": rolling_mdd,
        "mva50_gap_pct": mva50_gap,
        "mva200_gap_pct": mva200_gap,
        "ytd_return_pct": ytd_return,
        "ret_1y_pct": ret_1y,
    }

def human_num(x):
    if pd.isna(x):
        return "N/A"
    x = float(x)
    absx = abs(x)
    if absx >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.2f}T"
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    return f"{x:,.0f}"

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.1f}%"

def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

def build_screener_df(tickers: list[str], price_data: pd.DataFrame, snap: dict) -> pd.DataFrame:
    rows = []
    for t in tickers:
        s = get_close_series(price_data, t)
        p = compute_price_metrics(s)
        info = snap.get(t, {})
        row = {
            "Ticker": t,
            "Name": info.get("shortName", t),
            "Sector": info.get("sector", ""),
            "Industry": info.get("industry", ""),
            "MarketCap": info.get("marketCap", np.nan),
            "Price": p["current"],
            "ATH": p["ath"],
            "DiscountFromATH_pct": p["discount_from_ath_pct"],
            "RollingMDD_pct": p["rolling_mdd_pct"],
            "ROE_pct": info.get("roe_pct", np.nan),
            "ROA_pct": info.get("roa_pct", np.nan),
            "GrossMargin_pct": info.get("grossMargins_pct", np.nan),
            "OperatingMargin_pct": info.get("operatingMargins_pct", np.nan),
            "ProfitMargin_pct": info.get("profitMargins_pct", np.nan),
            "RevenueGrowth_pct": info.get("revenueGrowth_pct", np.nan),
            "TrailingPE": info.get("trailingPE", np.nan),
            "ForwardPE": info.get("forwardPE", np.nan),
            "PBR": info.get("priceToBook", np.nan),
            "DebtToEquity": info.get("debtToEquity", np.nan),
            "MVA50Gap_pct": p["mva50_gap_pct"],
            "MVA200Gap_pct": p["mva200_gap_pct"],
            "YTD_pct": p["ytd_return_pct"],
            "Ret1Y_pct": p["ret_1y_pct"],
        }
        row["CheapnessLabel"] = label_cheapness(row["DiscountFromATH_pct"], row["ROE_pct"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Scoring
    df["ROE_score"] = normalize_rank(df["ROE_pct"], higher_is_better=True)
    df["Discount_score"] = normalize_rank(df["DiscountFromATH_pct"], higher_is_better=True)

    # Lower PE is better, but only valid positive PE should be scored
    pe_for_score = df["TrailingPE"].where(df["TrailingPE"] > 0, np.nan)
    df["PE_score"] = normalize_rank(pe_for_score, higher_is_better=False)

    df["Growth_score"] = normalize_rank(df["RevenueGrowth_pct"], higher_is_better=True)

    df["QuantScore"] = (
        w_roe * df["ROE_score"].fillna(0)
        + w_discount * df["Discount_score"].fillna(0)
        + w_pe * df["PE_score"].fillna(0)
        + w_growth * df["Growth_score"].fillna(0)
    )

    # Basic filter flag
    df["PassFilter"] = (
        (df["ROE_pct"].fillna(-999) >= min_roe) &
        (df["DiscountFromATH_pct"].fillna(-999) >= min_discount) &
        (
            df["TrailingPE"].isna() |
            ((df["TrailingPE"] > 0) & (df["TrailingPE"] <= max_pe))
        )
    )

    df = df.sort_values(["PassFilter", "QuantScore", "ROE_pct"], ascending=[False, False, False]).reset_index(drop=True)
    return df

def draw_relative_chart(price_data: pd.DataFrame, tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        s = get_close_series(price_data, t)
        if s.empty:
            continue
        rebased = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=rebased.index,
            y=rebased.values,
            mode="lines",
            name=t
        ))
    fig.update_layout(
        title="Relative Price Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Price",
        height=520,
        legend_title="Ticker",
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig

def draw_drawdown_chart(price_data: pd.DataFrame, tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        s = get_close_series(price_data, t)
        if s.empty:
            continue
        dd = (s / s.cummax() - 1.0) * 100.0
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name=t
        ))
    fig.update_layout(
        title="Drawdown vs Previous Peak (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        height=520,
        legend_title="Ticker",
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig

# =========================================================
# Refresh control
# =========================================================
if refresh:
    st.cache_data.clear()

# =========================================================
# Data loading
# =========================================================
if not universe:
    st.warning("Please provide at least one ticker.")
    st.stop()

with st.spinner("Loading price and fundamental data..."):
    price_data = load_price_history(universe, history_years)
    snap = load_snapshot_data(universe)
    df = build_screener_df(universe, price_data, snap)

last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated: {last_updated}")

# =========================================================
# Summary metrics
# =========================================================
passed = df[df["PassFilter"]].copy()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Universe Size", f"{len(df)}")
col2.metric("Passed Filter", f"{len(passed)}")
col3.metric("Best Quant Score", fmt_num(df["QuantScore"].max()))
col4.metric("Avg Discount (Passed)", fmt_pct(passed["DiscountFromATH_pct"].mean() if len(passed) else np.nan))

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["Screener", "Charts", "Methodology"])

with tab1:
    st.subheader("Top Quant Ideas")
    show_pass_only = st.toggle("Show only filtered stocks", value=True)

    work = passed.copy() if show_pass_only else df.copy()
    work = work.head(top_n).copy()

    display_cols = [
        "Ticker", "Name", "Sector", "MarketCap", "Price", "ATH",
        "DiscountFromATH_pct", "RollingMDD_pct", "ROE_pct", "RevenueGrowth_pct",
        "TrailingPE", "ForwardPE", "PBR", "DebtToEquity", "MVA50Gap_pct",
        "MVA200Gap_pct", "QuantScore", "CheapnessLabel", "PassFilter"
    ]

    shown = work[display_cols].copy()
    shown["MarketCap"] = shown["MarketCap"].apply(human_num)

    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "ATH": st.column_config.NumberColumn(format="%.2f"),
            "DiscountFromATH_pct": st.column_config.NumberColumn("Discount from ATH %", format="%.1f"),
            "RollingMDD_pct": st.column_config.NumberColumn("Window MDD %", format="%.1f"),
            "ROE_pct": st.column_config.NumberColumn("ROE %", format="%.1f"),
            "RevenueGrowth_pct": st.column_config.NumberColumn("Revenue Growth %", format="%.1f"),
            "TrailingPE": st.column_config.NumberColumn("Trailing PE", format="%.2f"),
            "ForwardPE": st.column_config.NumberColumn("Forward PE", format="%.2f"),
            "PBR": st.column_config.NumberColumn("P/B", format="%.2f"),
            "DebtToEquity": st.column_config.NumberColumn("Debt/Equity", format="%.2f"),
            "MVA50Gap_pct": st.column_config.NumberColumn("vs MA50 %", format="%.1f"),
            "MVA200Gap_pct": st.column_config.NumberColumn("vs MA200 %", format="%.1f"),
            "QuantScore": st.column_config.ProgressColumn(
                "Quant Score", min_value=0, max_value=100, format="%.1f"
            ),
        }
    )

    st.markdown("### Quick interpretation")
    if len(work) > 0:
        best = work.iloc[0]
        st.info(
            f"Top idea now: **{best['Ticker']}** | "
            f"Discount from ATH: **{fmt_pct(best['DiscountFromATH_pct'])}** | "
            f"ROE: **{fmt_pct(best['ROE_pct'])}** | "
            f"PE: **{fmt_num(best['TrailingPE'])}** | "
            f"Label: **{best['CheapnessLabel']}**"
        )

    csv_bytes = shown.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download screener CSV",
        data=csv_bytes,
        file_name="nasdaq_quant_screener.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Visual Analysis")

    chart_candidates = (passed if len(passed) > 0 else df).head(10)["Ticker"].tolist()
    selected = st.multiselect(
        "Select tickers for chart",
        options=df["Ticker"].tolist(),
        default=chart_candidates if chart_candidates else DEFAULT_TICKERS[:5]
    )

    if selected:
        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(draw_relative_chart(price_data, selected), use_container_width=True)

        with c2:
            st.plotly_chart(draw_drawdown_chart(price_data, selected), use_container_width=True)

        selected_one = st.selectbox("Single ticker detail", selected, index=0)
        s = get_close_series(price_data, selected_one)

        if not s.empty:
            ma50 = s.rolling(50).mean()
            ma200 = s.rolling(200).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Price"))
            fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode="lines", name="MA50"))
            fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, mode="lines", name="MA200"))
            fig.update_layout(
                title=f"{selected_one} Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                height=520,
                margin=dict(l=30, r=30, t=60, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

            row = df[df["Ticker"] == selected_one].iloc[0]
            st.markdown(
                f"""
**{selected_one} summary**
- Cheapness label: **{row['CheapnessLabel']}**
- Discount from ATH: **{fmt_pct(row['DiscountFromATH_pct'])}**
- Rolling MDD ({history_years}Y window): **{fmt_pct(row['RollingMDD_pct'])}**
- ROE: **{fmt_pct(row['ROE_pct'])}**
- Trailing PE: **{fmt_num(row['TrailingPE'])}**
- Revenue growth: **{fmt_pct(row['RevenueGrowth_pct'])}**
- Price vs MA50: **{fmt_pct(row['MVA50Gap_pct'])}**
- Price vs MA200: **{fmt_pct(row['MVA200Gap_pct'])}**
- Quant Score: **{fmt_num(row['QuantScore'])}**
                """
            )
    else:
        st.warning("Please select at least one ticker.")

with tab3:
    st.subheader("Methodology")

    st.markdown(
        """
### 1. What this screener means by "cheap now"
This app does **not** define cheapness only by low PE.

It combines:
- **ROE**: stronger business quality
- **Discount from ATH**: how far current price is below peak
- **Rolling MDD**: worst drawdown inside the selected history window
- **PE**: valuation sanity check
- **Revenue growth**: optional growth support

### 2. Main formulas
- **Discount from ATH %** = `(ATH - Current Price) / ATH × 100`
- **Drawdown %** = `(Price / Running Peak - 1) × 100`
- **Rolling MDD %** = minimum drawdown over the selected window

### 3. Interpretation
- High ROE + large discount from ATH + reasonable PE
  → potentially attractive
- Large drawdown but low/negative ROE
  → may be a value trap
- High ROE but near ATH
  → great company, but not necessarily cheap now

### 4. Suggested practical filter
A simple rule many investors like:
- ROE ≥ 10~15%
- Discount from ATH ≥ 20~30%
- PE not excessively high
- Revenue growth still positive

### 5. Notes
- Some fundamentals can be missing or delayed.
- PE can be meaningless for firms with unstable or negative earnings.
- This is a ranking tool, not an automatic buy signal.
        """
    )
