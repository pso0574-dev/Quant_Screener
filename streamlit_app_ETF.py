# streamlit_app.py
# QQQ / SPY / SCHD Quant Analysis Dashboard

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="ETF Quant Analyzer",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# Settings
# =========================================================
DEFAULT_TICKERS = ["QQQ", "SPY", "SCHD"]
RISK_FREE_RATE = 0.02  # 2% assumption for Sharpe

# =========================================================
# Helpers
# =========================================================
@st.cache_data
def load_data(tickers, start_date, end_date):
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            prices = df["Close"].copy()
        else:
            # fallback: try Adj Close or first level
            first_level = df.columns.get_level_values(0).unique().tolist()
            chosen = "Adj Close" if "Adj Close" in first_level else first_level[0]
            prices = df[chosen].copy()
    else:
        prices = df.copy()

    prices = prices.dropna(how="all")
    return prices

def compute_metrics(price_df):
    daily_ret = price_df.pct_change().dropna()
    total_days = (price_df.index[-1] - price_df.index[0]).days
    years = max(total_days / 365.25, 1e-9)

    metrics = {}

    for ticker in price_df.columns:
        px = price_df[ticker].dropna()
        ret = daily_ret[ticker].dropna()

        if len(px) < 30 or len(ret) < 20:
            continue

        cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = np.nan if ann_vol == 0 else (ret.mean() * 252 - RISK_FREE_RATE) / ann_vol

        running_max = px.cummax()
        drawdown = px / running_max - 1
        mdd = drawdown.min()

        ma50 = px.rolling(50).mean().iloc[-1]
        ma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else np.nan
        current = px.iloc[-1]

        mom_1m = px.iloc[-1] / px.iloc[-22] - 1 if len(px) >= 22 else np.nan
        mom_3m = px.iloc[-1] / px.iloc[-63] - 1 if len(px) >= 63 else np.nan
        mom_6m = px.iloc[-1] / px.iloc[-126] - 1 if len(px) >= 126 else np.nan
        mom_12m = px.iloc[-1] / px.iloc[-252] - 1 if len(px) >= 252 else np.nan

        metrics[ticker] = {
            "Last Price": current,
            "CAGR": cagr,
            "Volatility": ann_vol,
            "Sharpe": sharpe,
            "MDD": mdd,
            "Above MA50 (%)": (current / ma50 - 1) if pd.notna(ma50) else np.nan,
            "Above MA200 (%)": (current / ma200 - 1) if pd.notna(ma200) else np.nan,
            "Momentum 1M": mom_1m,
            "Momentum 3M": mom_3m,
            "Momentum 6M": mom_6m,
            "Momentum 12M": mom_12m,
        }

    return pd.DataFrame(metrics).T

def normalize_series(s, higher_is_better=True):
    s = s.astype(float)
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=s.index)

    min_v, max_v = valid.min(), valid.max()
    if np.isclose(max_v, min_v):
        return pd.Series(50.0, index=s.index)

    scaled = (s - min_v) / (max_v - min_v) * 100
    if not higher_is_better:
        scaled = 100 - scaled
    return scaled

def compute_quant_score(metrics_df):
    score_df = pd.DataFrame(index=metrics_df.index)

    score_df["Momentum 3M Score"] = normalize_series(metrics_df["Momentum 3M"], True)
    score_df["Momentum 6M Score"] = normalize_series(metrics_df["Momentum 6M"], True)
    score_df["Momentum 12M Score"] = normalize_series(metrics_df["Momentum 12M"], True)
    score_df["Sharpe Score"] = normalize_series(metrics_df["Sharpe"], True)
    score_df["Volatility Score"] = normalize_series(metrics_df["Volatility"], False)
    score_df["MDD Score"] = normalize_series(metrics_df["MDD"], True)  # less negative is better
    score_df["Trend Score"] = normalize_series(metrics_df["Above MA200 (%)"], True)

    score_df["Quant Score"] = (
        score_df["Momentum 3M Score"] * 0.15 +
        score_df["Momentum 6M Score"] * 0.20 +
        score_df["Momentum 12M Score"] * 0.20 +
        score_df["Sharpe Score"] * 0.20 +
        score_df["Volatility Score"] * 0.10 +
        score_df["MDD Score"] * 0.10 +
        score_df["Trend Score"] * 0.05
    )

    return score_df.round(2)

def format_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"

def format_num(x):
    return f"{x:.2f}" if pd.notna(x) else "-"

def generate_signal(row):
    signals = []

    if pd.notna(row["Above MA200 (%)"]):
        if row["Above MA200 (%)"] > 0:
            signals.append("Bullish Trend")
        else:
            signals.append("Below MA200")

    if pd.notna(row["Momentum 6M"]):
        if row["Momentum 6M"] > 0:
            signals.append("Positive 6M Momentum")
        else:
            signals.append("Weak 6M Momentum")

    if pd.notna(row["Sharpe"]):
        if row["Sharpe"] > 0.8:
            signals.append("Strong Risk-Adjusted Return")
        elif row["Sharpe"] < 0.3:
            signals.append("Low Risk-Adjusted Return")

    return " | ".join(signals)

# =========================================================
# Sidebar
# =========================================================
st.title("📊 ETF Quant Analyzer")
st.caption("QQQ / SPY / SCHD based quantitative comparison dashboard")

st.sidebar.header("Settings")

end_date = datetime.today().date()
start_date = st.sidebar.date_input(
    "Start Date",
    value=end_date - timedelta(days=365 * 5)
)
end_date = st.sidebar.date_input(
    "End Date",
    value=end_date
)

tickers_input = st.sidebar.text_input(
    "Tickers",
    value="QQQ,SPY,SCHD"
)

tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

if len(tickers) == 0:
    st.warning("Please enter at least one ticker.")
    st.stop()

# =========================================================
# Data Load
# =========================================================
prices = load_data(tickers, start_date, end_date)

if prices.empty:
    st.error("No price data loaded. Please check ticker symbols or date range.")
    st.stop()

# Ensure only valid columns
prices = prices.dropna(axis=1, how="all")

if prices.shape[1] == 0:
    st.error("No valid ticker data available.")
    st.stop()

returns = prices.pct_change().dropna()
cum_returns = (1 + returns).cumprod()

metrics_df = compute_metrics(prices)
score_df = compute_quant_score(metrics_df)
summary_df = metrics_df.join(score_df["Quant Score"]).sort_values("Quant Score", ascending=False)
summary_df["Signal"] = summary_df.apply(generate_signal, axis=1)

# =========================================================
# Top Summary
# =========================================================
st.subheader("1. Quant Ranking")

rank_table = summary_df.copy()

display_rank = pd.DataFrame(index=rank_table.index)
display_rank["Last Price"] = rank_table["Last Price"].map(format_num)
display_rank["CAGR"] = rank_table["CAGR"].map(format_pct)
display_rank["Volatility"] = rank_table["Volatility"].map(format_pct)
display_rank["Sharpe"] = rank_table["Sharpe"].map(format_num)
display_rank["MDD"] = rank_table["MDD"].map(format_pct)
display_rank["Momentum 3M"] = rank_table["Momentum 3M"].map(format_pct)
display_rank["Momentum 6M"] = rank_table["Momentum 6M"].map(format_pct)
display_rank["Momentum 12M"] = rank_table["Momentum 12M"].map(format_pct)
display_rank["Above MA200"] = rank_table["Above MA200 (%)"].map(format_pct)
display_rank["Quant Score"] = rank_table["Quant Score"].map(format_num)
display_rank["Signal"] = rank_table["Signal"]

st.dataframe(display_rank, use_container_width=True)

winner = summary_df.index[0]
st.success(f"Top ranked ETF by current Quant Score: **{winner}**")

# =========================================================
# Price Chart
# =========================================================
st.subheader("2. Price Chart")

fig_price = go.Figure()
for col in prices.columns:
    fig_price.add_trace(go.Scatter(
        x=prices.index,
        y=prices[col],
        mode="lines",
        name=col
    ))

fig_price.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Ticker"
)
st.plotly_chart(fig_price, use_container_width=True)

# =========================================================
# Cumulative Return Chart
# =========================================================
st.subheader("3. Cumulative Return")

fig_cum = go.Figure()
for col in cum_returns.columns:
    fig_cum.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns[col],
        mode="lines",
        name=col
    ))

fig_cum.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Growth of $1",
    legend_title="Ticker"
)
st.plotly_chart(fig_cum, use_container_width=True)

# =========================================================
# Drawdown Chart
# =========================================================
st.subheader("4. Drawdown Analysis")

fig_dd = go.Figure()
for col in prices.columns:
    rolling_max = prices[col].cummax()
    dd = prices[col] / rolling_max - 1

    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd,
        mode="lines",
        name=col
    ))

fig_dd.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Drawdown",
    legend_title="Ticker"
)
st.plotly_chart(fig_dd, use_container_width=True)

# =========================================================
# Momentum Comparison
# =========================================================
st.subheader("5. Momentum Comparison")

momentum_cols = ["Momentum 1M", "Momentum 3M", "Momentum 6M", "Momentum 12M"]
mom_display = metrics_df[momentum_cols].copy()

for c in momentum_cols:
    mom_display[c] = mom_display[c].map(format_pct)

st.dataframe(mom_display, use_container_width=True)

# =========================================================
# Moving Average / Trend
# =========================================================
st.subheader("6. Trend Filter (MA50 / MA200)")

trend_display = pd.DataFrame(index=metrics_df.index)
trend_display["Above MA50"] = metrics_df["Above MA50 (%)"].map(format_pct)
trend_display["Above MA200"] = metrics_df["Above MA200 (%)"].map(format_pct)

st.dataframe(trend_display, use_container_width=True)

# =========================================================
# Score Breakdown
# =========================================================
st.subheader("7. Quant Score Breakdown")

score_display = score_df.copy()
for c in score_display.columns:
    score_display[c] = score_display[c].map(format_num)

st.dataframe(score_display, use_container_width=True)

# =========================================================
# Interpretation
# =========================================================
st.subheader("8. Interpretation Guide")

st.markdown("""
**How to read this dashboard**
- **CAGR**: Long-term annualized growth rate
- **Volatility**: Annualized risk level
- **Sharpe**: Risk-adjusted return (higher is generally better)
- **MDD**: Maximum drawdown (less negative is better)
- **Momentum 3M/6M/12M**: Medium/long trend strength
- **Above MA200**: Long-term trend filter

**Basic Quant logic**
- Prefer ETFs with:
  - higher momentum
  - higher Sharpe
  - shallower drawdown
  - lower volatility
  - price above MA200
""")

# =========================================================
# Latest Snapshot
# =========================================================
st.subheader("9. Latest Snapshot")

for ticker in summary_df.index:
    row = summary_df.loc[ticker]
    with st.container():
        st.markdown(f"### {ticker}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quant Score", format_num(row["Quant Score"]))
        c2.metric("CAGR", format_pct(row["CAGR"]))
        c3.metric("Sharpe", format_num(row["Sharpe"]))
        c4.metric("MDD", format_pct(row["MDD"]))
        st.write(f"Signal: {row['Signal']}")
        st.divider()
