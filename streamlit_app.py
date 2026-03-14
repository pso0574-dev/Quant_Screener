# -*- coding: utf-8 -*-
# SPY / QQQ / SCHD Quant Screener
# Run:
#   streamlit run streamlit_app.py

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SPY QQQ SCHD Quant Screener",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# SETTINGS
# =========================================================
ETF_UNIVERSE = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "SCHD": "Schwab U.S. Dividend Equity ETF"
}

DEFAULT_START = "2018-01-01"

st.title("📊 SPY / QQQ / SCHD Quant Screener")
st.caption("Compare momentum, trend, volatility, drawdown, and relative strength")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Settings")

start_date = st.sidebar.date_input(
    "Start date",
    value=pd.to_datetime(DEFAULT_START)
)

benchmark = st.sidebar.selectbox(
    "Relative benchmark",
    options=["SPY", "QQQ", "SCHD"],
    index=0
)

profile = st.sidebar.selectbox(
    "Scoring profile",
    options=["Balanced", "Aggressive", "Defensive", "Long-Term"],
    index=0
)

show_normalized = st.sidebar.checkbox("Show normalized price chart", value=True)
show_drawdown = st.sidebar.checkbox("Show drawdown chart", value=True)

# Default weights by profile
if profile == "Balanced":
    default_weights = {
        "mom_1m": 0.10,
        "mom_3m": 0.15,
        "mom_6m": 0.20,
        "mom_12m": 0.20,
        "trend": 0.15,
        "sharpe": 0.10,
        "low_vol": 0.05,
        "low_mdd": 0.05
    }
elif profile == "Aggressive":
    default_weights = {
        "mom_1m": 0.15,
        "mom_3m": 0.20,
        "mom_6m": 0.20,
        "mom_12m": 0.20,
        "trend": 0.15,
        "sharpe": 0.05,
        "low_vol": 0.03,
        "low_mdd": 0.02
    }
elif profile == "Defensive":
    default_weights = {
        "mom_1m": 0.05,
        "mom_3m": 0.10,
        "mom_6m": 0.15,
        "mom_12m": 0.15,
        "trend": 0.15,
        "sharpe": 0.15,
        "low_vol": 0.15,
        "low_mdd": 0.10
    }
else:  # Long-Term
    default_weights = {
        "mom_1m": 0.03,
        "mom_3m": 0.07,
        "mom_6m": 0.20,
        "mom_12m": 0.30,
        "trend": 0.20,
        "sharpe": 0.10,
        "low_vol": 0.05,
        "low_mdd": 0.05
    }

st.sidebar.markdown("### Factor Weights")
weights = {}
for k, v in default_weights.items():
    weights[k] = st.sidebar.slider(
        k, 0.0, 1.0, float(v), 0.01
    )

weight_sum = sum(weights.values())
if weight_sum == 0:
    st.error("At least one factor weight must be greater than 0.")
    st.stop()

weights = {k: v / weight_sum for k, v in weights.items()}

# =========================================================
# DATA LOADER
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start):
    data = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )
    return data

tickers = list(ETF_UNIVERSE.keys())
raw = load_data(tickers, str(start_date))

if raw.empty:
    st.error("Failed to download data.")
    st.stop()

# =========================================================
# HELPERS
# =========================================================
def get_close_volume(data, ticker):
    close = data[ticker]["Close"].dropna()
    volume = data[ticker]["Volume"].dropna()
    return close, volume

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def pct_return(series, days):
    if len(series) <= days:
        return np.nan
    return series.iloc[-1] / series.iloc[-days - 1] - 1

def annualized_vol(returns):
    return returns.std() * np.sqrt(252)

def sharpe_proxy(returns):
    vol = annualized_vol(returns)
    if pd.isna(vol) or vol == 0:
        return np.nan
    return returns.mean() * 252 / vol

def max_drawdown(series):
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd.min()

def drawdown_series(series):
    roll_max = series.cummax()
    return series / roll_max - 1.0

def zscore_rank(s):
    if s.nunique() <= 1:
        return pd.Series(50.0, index=s.index)
    z = (s - s.mean()) / s.std(ddof=0)
    score = 50 + 15 * z
    return score.clip(0, 100)

# =========================================================
# FACTOR CALCULATION
# =========================================================
rows = []
price_dict = {}
drawdown_dict = {}

for ticker in tickers:
    close, volume = get_close_volume(raw, ticker)

    if len(close) < 220:
        continue

    ret = close.pct_change().dropna()

    price_dict[ticker] = close
    drawdown_dict[ticker] = drawdown_series(close)

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last_price = close.iloc[-1]

    trend_raw = (
        int(last_price > sma50) +
        int(last_price > sma200) +
        int(sma50 > sma200)
    ) / 3.0

    row = {
        "Ticker": ticker,
        "Name": ETF_UNIVERSE[ticker],
        "Last": last_price,
        "Return 1M": pct_return(close, 21),
        "Return 3M": pct_return(close, 63),
        "Return 6M": pct_return(close, 126),
        "Return 12M": pct_return(close, 252),
        "Vol 20D": ret.rolling(20).std().iloc[-1] * np.sqrt(252),
        "Vol 60D": ret.rolling(60).std().iloc[-1] * np.sqrt(252),
        "MDD Full": max_drawdown(close),
        "MDD 1Y": max_drawdown(close.tail(252)),
        "RSI14": calc_rsi(close, 14).iloc[-1],
        "SMA50": sma50,
        "SMA200": sma200,
        "Trend Raw": trend_raw,
        "Sharpe 1Y": sharpe_proxy(ret.tail(252)),
        "Avg Dollar Vol 20D": (close * volume).rolling(20).mean().iloc[-1]
    }
    rows.append(row)

factors = pd.DataFrame(rows).set_index("Ticker")

if factors.empty:
    st.error("Not enough data to calculate factors.")
    st.stop()

# Relative strength vs selected benchmark
bench = price_dict[benchmark]
rel_3m = {}
rel_6m = {}
rel_12m = {}

for ticker in factors.index:
    aligned = pd.concat([price_dict[ticker], bench], axis=1, join="inner").dropna()
    aligned.columns = ["asset", "bench"]

    def rel_ret(days):
        if len(aligned) <= days:
            return np.nan
        asset_r = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-days - 1] - 1
        bench_r = aligned["bench"].iloc[-1] / aligned["bench"].iloc[-days - 1] - 1
        return asset_r - bench_r

    rel_3m[ticker] = rel_ret(63)
    rel_6m[ticker] = rel_ret(126)
    rel_12m[ticker] = rel_ret(252)

factors["Rel 3M"] = pd.Series(rel_3m)
factors["Rel 6M"] = pd.Series(rel_6m)
factors["Rel 12M"] = pd.Series(rel_12m)

# =========================================================
# SCORING
# =========================================================
score_df = factors.copy()

score_df["Score Mom 1M"] = zscore_rank(score_df["Return 1M"])
score_df["Score Mom 3M"] = zscore_rank(score_df["Return 3M"])
score_df["Score Mom 6M"] = zscore_rank(score_df["Return 6M"])
score_df["Score Mom 12M"] = zscore_rank(score_df["Return 12M"])
score_df["Score Trend"] = score_df["Trend Raw"] * 100
score_df["Score Sharpe"] = zscore_rank(score_df["Sharpe 1Y"])
score_df["Score Low Vol"] = zscore_rank(-score_df["Vol 60D"])
score_df["Score Low MDD"] = zscore_rank(-score_df["MDD 1Y"].abs())

score_df["Composite Score"] = (
    weights["mom_1m"] * score_df["Score Mom 1M"] +
    weights["mom_3m"] * score_df["Score Mom 3M"] +
    weights["mom_6m"] * score_df["Score Mom 6M"] +
    weights["mom_12m"] * score_df["Score Mom 12M"] +
    weights["trend"] * score_df["Score Trend"] +
    weights["sharpe"] * score_df["Score Sharpe"] +
    weights["low_vol"] * score_df["Score Low Vol"] +
    weights["low_mdd"] * score_df["Score Low MDD"]
)

score_df["Rank"] = score_df["Composite Score"].rank(
    ascending=False, method="dense"
).astype(int)

# =========================================================
# DISPLAY FORMAT
# =========================================================
table_cols = [
    "Rank", "Name", "Last",
    "Return 1M", "Return 3M", "Return 6M", "Return 12M",
    "Rel 3M", "Rel 6M", "Rel 12M",
    "Vol 20D", "Vol 60D",
    "MDD 1Y", "MDD Full",
    "RSI14", "Sharpe 1Y", "Composite Score"
]

table_df = score_df[table_cols].sort_values("Rank").copy()

def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"

def fmt_num(x):
    return f"{x:,.2f}" if pd.notna(x) else "-"

styled_df = table_df.copy()
pct_cols = [
    "Return 1M", "Return 3M", "Return 6M", "Return 12M",
    "Rel 3M", "Rel 6M", "Rel 12M",
    "Vol 20D", "Vol 60D", "MDD 1Y", "MDD Full"
]
num_cols = ["Last", "RSI14", "Sharpe 1Y", "Composite Score"]

for c in pct_cols:
    styled_df[c] = styled_df[c].map(fmt_pct)
for c in num_cols:
    styled_df[c] = styled_df[c].map(fmt_num)

# =========================================================
# TOP SUMMARY
# =========================================================
top_ticker = table_df.index[0]
top_name = ETF_UNIVERSE[top_ticker]

colA, colB, colC, colD = st.columns(4)
colA.metric("Top Ranked", top_ticker)
colB.metric("Benchmark", benchmark)
colC.metric("Profile", profile)
colD.metric("Universe Size", len(table_df))

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Screener",
    "📈 Price",
    "📉 Drawdown",
    "🧪 Factor Heatmap",
    "📘 Interpretation"
])

with tab1:
    st.subheader("Ranking Table")
    st.dataframe(styled_df, use_container_width=True)
    st.success(f"Current top-ranked ETF: {top_ticker} - {top_name}")

    c1, c2 = st.columns(2)

    with c1:
        bar_df = table_df.reset_index()[["Ticker", "Composite Score"]]
        fig_bar = px.bar(
            bar_df,
            x="Ticker",
            y="Composite Score",
            title="Composite Score"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        scatter_df = score_df.reset_index()
        fig_scatter = px.scatter(
            scatter_df,
            x="Vol 60D",
            y="Return 12M",
            text="Ticker",
            size="Composite Score",
            hover_data=["Name", "Sharpe 1Y", "MDD 1Y"],
            title="12M Return vs 60D Volatility"
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Normalized Price Comparison")
    if show_normalized:
        base100 = pd.DataFrame()
        for ticker, series in price_dict.items():
            s = series.dropna()
            base100[ticker] = s / s.iloc[0] * 100

        fig_line = go.Figure()
        for ticker in base100.columns:
            fig_line.add_trace(go.Scatter(
                x=base100.index,
                y=base100[ticker],
                mode="lines",
                name=ticker
            ))

        fig_line.update_layout(
            title="Growth of 100",
            xaxis_title="Date",
            yaxis_title="Indexed Price"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        returns_compare = pd.DataFrame({
            "Ticker": list(factors.index),
            "1M": factors["Return 1M"].values,
            "3M": factors["Return 3M"].values,
            "6M": factors["Return 6M"].values,
            "12M": factors["Return 12M"].values
        }).set_index("Ticker")

        st.dataframe(
            returns_compare.style.format("{:.2%}"),
            use_container_width=True
        )

with tab3:
    st.subheader("Drawdown Comparison")
    if show_drawdown:
        dd_df = pd.DataFrame(drawdown_dict)

        fig_dd = go.Figure()
        for ticker in dd_df.columns:
            fig_dd.add_trace(go.Scatter(
                x=dd_df.index,
                y=dd_df[ticker],
                mode="lines",
                name=ticker
            ))

        fig_dd.update_layout(
            title="Historical Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        dd_summary = factors[["MDD 1Y", "MDD Full"]].copy()
        st.dataframe(dd_summary.style.format("{:.2%}"), use_container_width=True)

with tab4:
    st.subheader("Factor Heatmap")

    heat_cols = [
        "Score Mom 1M", "Score Mom 3M", "Score Mom 6M", "Score Mom 12M",
        "Score Trend", "Score Sharpe", "Score Low Vol", "Score Low MDD",
        "Composite Score"
    ]
    heat_df = score_df[heat_cols].sort_values("Composite Score", ascending=False)

    fig_heat = px.imshow(
        heat_df,
        text_auto=".1f",
        aspect="auto",
        title="Factor Scores"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab5:
    st.subheader("How to Read This")

    st.markdown(f"""
### ETF Character Differences
- **SPY**: broad US large-cap exposure, core market beta
- **QQQ**: stronger growth/tech tilt, usually higher upside and higher volatility
- **SCHD**: dividend + quality tilt, often stronger downside defense than QQQ

### Quant Interpretation
- **High 1M / 3M / 6M / 12M return**  
  = stronger momentum

- **Price > SMA50 > SMA200**  
  = strong trend structure

- **Low Vol 60D**  
  = lower recent risk

- **Low MDD 1Y**  
  = better downside control

- **High Sharpe 1Y**  
  = better return per unit of risk

- **Positive Relative Return vs {benchmark}**  
  = outperforming selected benchmark

### Practical Use
- When **QQQ** ranks first:
  growth / risk-on regime 가능성

- When **SPY** ranks first:
  broad market leadership, more balanced regime 가능성

- When **SCHD** ranks first:
  defensive / income / quality preference 강화 가능성

### Important
This is a **screening and regime-reading tool**, not a guaranteed buy/sell signal.
Use it together with:
- position sizing
- rebalancing rules
- macro/liquidity context
- drawdown control
""")

st.markdown("---")
st.caption(
    f"Universe: {', '.join(tickers)} | Benchmark: {benchmark} | "
    f"Top ranked: {top_ticker} | Data source: Yahoo Finance via yfinance"
)
