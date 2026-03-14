# streamlit_app.py
# ETF Top 10 AUM Quant Screener
# Universe snapshot based on top ETFs by AUM:
# VOO, IVV, SPY, VTI, QQQ, VEA, VUG, GLD, IEFA, VTV

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ETF Top 10 Quant Screener",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Config
# --------------------------------------------------
ETF_UNIVERSE = {
    "VOO": "Vanguard S&P 500 ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "VTI": "Vanguard Total Stock Market ETF",
    "QQQ": "Invesco QQQ Trust",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VUG": "Vanguard Growth ETF",
    "GLD": "SPDR Gold Shares",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "VTV": "Vanguard Value ETF",
}

DEFAULT_START = "2020-01-01"

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.title("📊 ETF Top 10 Quant Screener")
st.caption("Top-10 ETF universe snapshot by AUM, with live market data screening")

st.sidebar.header("Settings")

start_date = st.sidebar.date_input(
    "Start date",
    value=pd.to_datetime(DEFAULT_START)
)

benchmark = st.sidebar.selectbox(
    "Benchmark for relative score",
    options=["SPY", "VOO", "VTI", "QQQ"],
    index=0
)

lookback_choice = st.sidebar.selectbox(
    "Ranking window focus",
    options=["Balanced", "Short-Term", "Long-Term", "Low-Volatility"],
    index=0
)

top_n = st.sidebar.slider("Show top N", 3, 10, 10)

st.sidebar.markdown("### Factor Weights")

if lookback_choice == "Balanced":
    default_weights = {
        "mom_1m": 0.10,
        "mom_3m": 0.15,
        "mom_6m": 0.20,
        "mom_12m": 0.20,
        "trend": 0.15,
        "sharpe": 0.10,
        "low_vol": 0.05,
        "low_mdd": 0.03,
        "liquidity": 0.02,
    }
elif lookback_choice == "Short-Term":
    default_weights = {
        "mom_1m": 0.20,
        "mom_3m": 0.25,
        "mom_6m": 0.15,
        "mom_12m": 0.05,
        "trend": 0.15,
        "sharpe": 0.10,
        "low_vol": 0.05,
        "low_mdd": 0.03,
        "liquidity": 0.02,
    }
elif lookback_choice == "Long-Term":
    default_weights = {
        "mom_1m": 0.03,
        "mom_3m": 0.07,
        "mom_6m": 0.20,
        "mom_12m": 0.30,
        "trend": 0.20,
        "sharpe": 0.10,
        "low_vol": 0.05,
        "low_mdd": 0.03,
        "liquidity": 0.02,
    }
else:  # Low-Volatility
    default_weights = {
        "mom_1m": 0.05,
        "mom_3m": 0.10,
        "mom_6m": 0.15,
        "mom_12m": 0.15,
        "trend": 0.10,
        "sharpe": 0.15,
        "low_vol": 0.20,
        "low_mdd": 0.08,
        "liquidity": 0.02,
    }

weights = {}
for k, v in default_weights.items():
    weights[k] = st.sidebar.slider(
        f"{k}",
        min_value=0.0,
        max_value=1.0,
        value=float(v),
        step=0.01
    )

weight_sum = sum(weights.values())
if weight_sum == 0:
    st.sidebar.error("At least one weight must be > 0")
    st.stop()

weights = {k: v / weight_sum for k, v in weights.items()}

# --------------------------------------------------
# Data
# --------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(tickers, start):
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )
    return df

tickers = list(ETF_UNIVERSE.keys())
raw = load_data(tickers, str(start_date))

if raw.empty:
    st.error("No data downloaded. Please retry.")
    st.stop()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_close_volume(data, ticker):
    if len(tickers) == 1:
        close = data["Close"].copy()
        volume = data["Volume"].copy()
    else:
        close = data[ticker]["Close"].copy()
        volume = data[ticker]["Volume"].copy()
    return close.dropna(), volume.dropna()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def annualized_return(series, periods_per_year=252):
    if len(series) < 2:
        return np.nan
    total_return = series.iloc[-1] / series.iloc[0]
    years = len(series) / periods_per_year
    if years <= 0:
        return np.nan
    return total_return ** (1 / years) - 1

def annualized_vol(ret, periods_per_year=252):
    return ret.std() * np.sqrt(periods_per_year)

def sharpe_proxy(ret):
    vol = annualized_vol(ret)
    if pd.isna(vol) or vol == 0:
        return np.nan
    return ret.mean() * 252 / vol

def max_drawdown(series):
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd.min()

def zscore_rank(s):
    if s.nunique() <= 1:
        return pd.Series(50.0, index=s.index)
    z = (s - s.mean()) / s.std(ddof=0)
    # map roughly into 0~100
    score = 50 + 15 * z
    return score.clip(0, 100)

def pct_return(series, days):
    if len(series) <= days:
        return np.nan
    return series.iloc[-1] / series.iloc[-days - 1] - 1

# --------------------------------------------------
# Factor Calculation
# --------------------------------------------------
rows = []
price_dict = {}

for ticker in tickers:
    close, volume = get_close_volume(raw, ticker)

    if len(close) < 220:
        continue

    ret = close.pct_change().dropna()
    price_dict[ticker] = close

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last_price = close.iloc[-1]

    trend_score_raw = (
        int(last_price > sma50) +
        int(last_price > sma200) +
        int(sma50 > sma200)
    ) / 3.0

    dollar_volume_20 = (close * volume).rolling(20).mean().iloc[-1]

    rsi14 = calc_rsi(close, 14).iloc[-1]
    vol20 = ret.rolling(20).std().iloc[-1] * np.sqrt(252)
    vol60 = ret.rolling(60).std().iloc[-1] * np.sqrt(252)
    mdd_1y = max_drawdown(close.tail(252))
    sharpe_1y = sharpe_proxy(ret.tail(252))

    row = {
        "Ticker": ticker,
        "Name": ETF_UNIVERSE[ticker],
        "Last": last_price,
        "Return 1M": pct_return(close, 21),
        "Return 3M": pct_return(close, 63),
        "Return 6M": pct_return(close, 126),
        "Return 12M": pct_return(close, 252),
        "Vol 20D": vol20,
        "Vol 60D": vol60,
        "MDD 1Y": mdd_1y,
        "SMA50": sma50,
        "SMA200": sma200,
        "Trend Raw": trend_score_raw,
        "RSI14": rsi14,
        "Sharpe 1Y": sharpe_1y,
        "Dollar Vol 20D": dollar_volume_20,
    }
    rows.append(row)

factors = pd.DataFrame(rows).set_index("Ticker")

if factors.empty:
    st.error("Not enough history to calculate factors.")
    st.stop()

# Benchmark relative return
if benchmark in price_dict:
    bench_close = price_dict[benchmark]
    rel_scores = {}
    for ticker in factors.index:
        aligned = pd.concat(
            [price_dict[ticker], bench_close],
            axis=1,
            join="inner"
        ).dropna()
        aligned.columns = ["asset", "bench"]
        rel_3m = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-64] - 1 if len(aligned) > 64 else np.nan
        bench_3m = aligned["bench"].iloc[-1] / aligned["bench"].iloc[-64] - 1 if len(aligned) > 64 else np.nan
        rel_scores[ticker] = rel_3m - bench_3m if pd.notna(rel_3m) and pd.notna(bench_3m) else np.nan
    factors["Rel 3M vs Benchmark"] = pd.Series(rel_scores)
else:
    factors["Rel 3M vs Benchmark"] = np.nan

# --------------------------------------------------
# Normalize into scores
# --------------------------------------------------
score_df = factors.copy()

score_df["Score Mom 1M"] = zscore_rank(score_df["Return 1M"])
score_df["Score Mom 3M"] = zscore_rank(score_df["Return 3M"])
score_df["Score Mom 6M"] = zscore_rank(score_df["Return 6M"])
score_df["Score Mom 12M"] = zscore_rank(score_df["Return 12M"])
score_df["Score Trend"] = score_df["Trend Raw"] * 100
score_df["Score Sharpe"] = zscore_rank(score_df["Sharpe 1Y"])
score_df["Score Low Vol"] = zscore_rank(-score_df["Vol 60D"])
score_df["Score Low MDD"] = zscore_rank(-score_df["MDD 1Y"].abs())
score_df["Score Liquidity"] = zscore_rank(score_df["Dollar Vol 20D"])

score_df["Composite Score"] = (
    weights["mom_1m"] * score_df["Score Mom 1M"] +
    weights["mom_3m"] * score_df["Score Mom 3M"] +
    weights["mom_6m"] * score_df["Score Mom 6M"] +
    weights["mom_12m"] * score_df["Score Mom 12M"] +
    weights["trend"] * score_df["Score Trend"] +
    weights["sharpe"] * score_df["Score Sharpe"] +
    weights["low_vol"] * score_df["Score Low Vol"] +
    weights["low_mdd"] * score_df["Score Low MDD"] +
    weights["liquidity"] * score_df["Score Liquidity"]
)

score_df["Rank"] = score_df["Composite Score"].rank(ascending=False, method="dense").astype(int)

display_cols = [
    "Rank", "Name", "Last",
    "Return 1M", "Return 3M", "Return 6M", "Return 12M",
    "Vol 20D", "Vol 60D", "MDD 1Y",
    "RSI14", "Sharpe 1Y", "Rel 3M vs Benchmark",
    "Composite Score"
]

table_df = score_df[display_cols].sort_values("Rank").head(top_n).copy()

# --------------------------------------------------
# Formatting
# --------------------------------------------------
def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"

def fmt_num(x):
    return f"{x:,.2f}" if pd.notna(x) else "-"

styled_df = table_df.copy()
for c in ["Return 1M", "Return 3M", "Return 6M", "Return 12M", "Vol 20D", "Vol 60D", "MDD 1Y", "Rel 3M vs Benchmark"]:
    styled_df[c] = styled_df[c].map(fmt_pct)

for c in ["Last", "RSI14", "Sharpe 1Y", "Composite Score"]:
    styled_df[c] = styled_df[c].map(fmt_num)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Screener",
    "📈 Relative Performance",
    "🧪 Factor Heatmap",
    "📘 Method"
])

with tab1:
    st.subheader("ETF Ranking Table")
    st.dataframe(styled_df, use_container_width=True)

    top_pick = table_df.index[0]
    st.success(f"Top ranked ETF: {top_pick} - {ETF_UNIVERSE[top_pick]}")

    col1, col2 = st.columns(2)

    with col1:
        bar_df = table_df.reset_index()[["Ticker", "Composite Score"]]
        fig_bar = px.bar(
            bar_df,
            x="Ticker",
            y="Composite Score",
            title="Composite Score Ranking"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
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
    base100 = pd.DataFrame()

    for ticker, series in price_dict.items():
        s = series.dropna()
        if not s.empty:
            base100[ticker] = s / s.iloc[0] * 100

    fig_line = go.Figure()
    for ticker in table_df.index:
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

with tab3:
    st.subheader("Factor Score Heatmap")

    heat_cols = [
        "Score Mom 1M", "Score Mom 3M", "Score Mom 6M", "Score Mom 12M",
        "Score Trend", "Score Sharpe", "Score Low Vol",
        "Score Low MDD", "Score Liquidity", "Composite Score"
    ]
    heat_df = score_df[heat_cols].sort_values("Composite Score", ascending=False).head(top_n)

    fig_heat = px.imshow(
        heat_df,
        text_auto=".1f",
        aspect="auto",
        title="Factor Score Heatmap"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.subheader("Methodology")
    st.markdown("""
**Universe**
- Fixed top-10 ETF universe by size snapshot

**Core factors**
- Momentum: 1M, 3M, 6M, 12M returns
- Trend: price vs SMA50/SMA200
- Risk: 20D / 60D annualized volatility
- Drawdown: trailing 1Y max drawdown
- Risk-adjusted return: trailing 1Y Sharpe proxy
- Liquidity proxy: 20D average dollar volume

**Interpretation**
- Higher momentum, stronger trend, higher Sharpe, stronger liquidity = better
- Lower volatility and smaller drawdown = better
- Final rank is a weighted composite score

**Important note**
- This is a screening tool, not an automatic buy signal.
- Use it together with macro view, valuation, and portfolio allocation rules.
    """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    f"Universe: {', '.join(tickers)} | Benchmark: {benchmark} | "
    f"Rows shown: {top_n} | Last data point may depend on market session timing."
)
