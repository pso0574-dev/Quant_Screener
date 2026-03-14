# streamlit_app.py
# Nasdaq-100 Quant Screener
# Streamlit Cloud friendly version
# - No pd.read_html()
# - Static Nasdaq-100 universe
# - Tab-based strategies
# - Evidence charts + stock tables
# - English only

import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Nasdaq-100 Quant Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Nasdaq-100 Quant Screener")
st.caption("Momentum · Quality · Growth · Value · Defensive · Multi-Factor")


# =========================================================
# 1) STATIC NASDAQ-100 UNIVERSE
# =========================================================
@st.cache_data
def get_nasdaq100_universe():
    data = [
        ("AAPL", "Apple"),
        ("ABNB", "Airbnb"),
        ("ADBE", "Adobe"),
        ("ADI", "Analog Devices"),
        ("ADP", "Automatic Data Processing"),
        ("ADSK", "Autodesk"),
        ("AEP", "American Electric Power"),
        ("AMAT", "Applied Materials"),
        ("AMD", "Advanced Micro Devices"),
        ("AMGN", "Amgen"),
        ("AMZN", "Amazon"),
        ("ANSS", "Ansys"),
        ("APP", "AppLovin"),
        ("ARM", "Arm Holdings"),
        ("ASML", "ASML"),
        ("AZN", "AstraZeneca"),
        ("BIIB", "Biogen"),
        ("BKNG", "Booking Holdings"),
        ("BKR", "Baker Hughes"),
        ("CCEP", "Coca-Cola Europacific Partners"),
        ("CDNS", "Cadence Design Systems"),
        ("CDW", "CDW"),
        ("CEG", "Constellation Energy"),
        ("CHTR", "Charter Communications"),
        ("CMCSA", "Comcast"),
        ("COST", "Costco"),
        ("CPRT", "Copart"),
        ("CRWD", "CrowdStrike"),
        ("CSCO", "Cisco"),
        ("CSGP", "CoStar Group"),
        ("CSX", "CSX"),
        ("CTAS", "Cintas"),
        ("CTSH", "Cognizant"),
        ("DASH", "DoorDash"),
        ("DDOG", "Datadog"),
        ("DXCM", "DexCom"),
        ("EA", "Electronic Arts"),
        ("EXC", "Exelon"),
        ("FANG", "Diamondback Energy"),
        ("FAST", "Fastenal"),
        ("FTNT", "Fortinet"),
        ("GEHC", "GE HealthCare"),
        ("GFS", "GlobalFoundries"),
        ("GILD", "Gilead Sciences"),
        ("GOOG", "Alphabet Class C"),
        ("GOOGL", "Alphabet Class A"),
        ("HON", "Honeywell"),
        ("IDXX", "IDEXX Laboratories"),
        ("INTC", "Intel"),
        ("INTU", "Intuit"),
        ("ISRG", "Intuitive Surgical"),
        ("KDP", "Keurig Dr Pepper"),
        ("KHC", "Kraft Heinz"),
        ("KLAC", "KLA"),
        ("LIN", "Linde"),
        ("LRCX", "Lam Research"),
        ("LULU", "Lululemon"),
        ("MAR", "Marriott"),
        ("MCHP", "Microchip Technology"),
        ("MDB", "MongoDB"),
        ("MDLZ", "Mondelez"),
        ("MELI", "MercadoLibre"),
        ("META", "Meta Platforms"),
        ("MNST", "Monster Beverage"),
        ("MRVL", "Marvell"),
        ("MSFT", "Microsoft"),
        ("MSTR", "MicroStrategy"),
        ("MU", "Micron"),
        ("NFLX", "Netflix"),
        ("NVDA", "NVIDIA"),
        ("NXPI", "NXP Semiconductors"),
        ("ODFL", "Old Dominion Freight Line"),
        ("ON", "ON Semiconductor"),
        ("ORLY", "O'Reilly Automotive"),
        ("PANW", "Palo Alto Networks"),
        ("PAYX", "Paychex"),
        ("PCAR", "PACCAR"),
        ("PDD", "PDD Holdings"),
        ("PEP", "PepsiCo"),
        ("PLTR", "Palantir"),
        ("PYPL", "PayPal"),
        ("QCOM", "Qualcomm"),
        ("REGN", "Regeneron"),
        ("ROP", "Roper Technologies"),
        ("ROST", "Ross Stores"),
        ("SBUX", "Starbucks"),
        ("SNPS", "Synopsys"),
        ("TEAM", "Atlassian"),
        ("TMUS", "T-Mobile"),
        ("TSLA", "Tesla"),
        ("TTD", "Trade Desk"),
        ("TTWO", "Take-Two Interactive"),
        ("TXN", "Texas Instruments"),
        ("VRSK", "Verisk"),
        ("VRTX", "Vertex Pharmaceuticals"),
        ("WBD", "Warner Bros. Discovery"),
        ("WDAY", "Workday"),
        ("XEL", "Xcel Energy"),
        ("ZS", "Zscaler"),
    ]
    return pd.DataFrame(data, columns=["Ticker", "Name"])


# =========================================================
# 2) HELPERS
# =========================================================
def safe_div(a, b):
    try:
        if b is None or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def pct_rank(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.sum() == 0:
        return out
    out.loc[valid] = s.loc[valid].rank(pct=True, ascending=not higher_is_better)
    return out


def winsorize_series(series, lower=0.02, upper=0.98):
    s = pd.to_numeric(series, errors="coerce").copy()
    if s.notna().sum() < 5:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}"


def fmt_money(x):
    if pd.isna(x):
        return "N/A"
    if abs(x) >= 1e12:
        return f"${x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"


# =========================================================
# 3) DATA DOWNLOAD
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(tickers, period="2y"):
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        return data
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def download_benchmark(period="2y"):
    try:
        qqq = yf.download(
            tickers="QQQ",
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if qqq.empty:
            return pd.DataFrame()
        qqq = qqq.dropna().copy()
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = qqq.columns.get_level_values(0)
        return qqq
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def download_fundamentals(tickers):
    rows = []

    for ticker in tickers:
        row = {"Ticker": ticker}
        try:
            info = yf.Ticker(ticker).info

            row.update(
                {
                    "MarketCap": info.get("marketCap", np.nan),
                    "EnterpriseValue": info.get("enterpriseValue", np.nan),
                    "TrailingPE": info.get("trailingPE", np.nan),
                    "ForwardPE": info.get("forwardPE", np.nan),
                    "PegRatio": info.get("pegRatio", np.nan),
                    "PriceToBook": info.get("priceToBook", np.nan),
                    "PriceToSales": info.get("priceToSalesTrailing12Months", np.nan),
                    "EnterpriseToRevenue": info.get("enterpriseToRevenue", np.nan),
                    "EnterpriseToEbitda": info.get("enterpriseToEbitda", np.nan),
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "OperatingMargin": info.get("operatingMargins", np.nan),
                    "GrossMargin": info.get("grossMargins", np.nan),
                    "ProfitMargin": info.get("profitMargins", np.nan),
                    "RevenueGrowth": info.get("revenueGrowth", np.nan),
                    "EpsGrowth": info.get("earningsGrowth", np.nan),
                    "QuarterlyRevenueGrowth": info.get("quarterlyRevenueGrowth", np.nan),
                    "DebtToEquity": info.get("debtToEquity", np.nan),
                    "CurrentRatio": info.get("currentRatio", np.nan),
                    "QuickRatio": info.get("quickRatio", np.nan),
                    "Beta": info.get("beta", np.nan),
                    "DividendYield": info.get("dividendYield", np.nan),
                    "Sector": info.get("sector", "Unknown"),
                    "Industry": info.get("industry", "Unknown"),
                }
            )
        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# 4) PRICE METRICS
# =========================================================
def extract_single_ticker_frame(price_data, ticker):
    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            if ticker not in price_data.columns.get_level_values(0):
                return pd.DataFrame()
            df = price_data[ticker].copy()
        else:
            df = price_data.copy()
        df = df.dropna(how="all")
        return df
    except Exception:
        return pd.DataFrame()


def compute_metrics(price_data, tickers, benchmark_df):
    rows = []

    qqq_close = None
    if not benchmark_df.empty and "Close" in benchmark_df.columns:
        qqq_close = benchmark_df["Close"].dropna()

    for ticker in tickers:
        try:
            df = extract_single_ticker_frame(price_data, ticker)
            if df.empty or "Close" not in df.columns:
                continue

            df = df.dropna().copy()
            close = df["Close"].astype(float).dropna()
            volume = df["Volume"].astype(float).dropna() if "Volume" in df.columns else pd.Series(dtype=float)

            if len(close) < 210:
                continue

            price = close.iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma100 = close.rolling(100).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]

            ret_1m = safe_div(price, close.iloc[-22]) - 1 if len(close) > 22 else np.nan
            ret_3m = safe_div(price, close.iloc[-63]) - 1 if len(close) > 63 else np.nan
            ret_6m = safe_div(price, close.iloc[-126]) - 1 if len(close) > 126 else np.nan
            ret_12m = safe_div(price, close.iloc[-252]) - 1 if len(close) > 252 else np.nan

            high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
            low_52w = close.tail(252).min() if len(close) >= 252 else close.min()
            dist_52w_high = safe_div(price, high_52w) - 1
            dist_52w_low = safe_div(price, low_52w) - 1

            daily_ret = close.pct_change().dropna()
            vol_1y = daily_ret.tail(252).std() * np.sqrt(252) if len(daily_ret) >= 20 else np.nan

            running_max_1y = close.tail(252).cummax()
            dd_1y_series = close.tail(252) / running_max_1y - 1
            mdd_1y = dd_1y_series.min() if len(dd_1y_series) > 0 else np.nan

            running_max_all = close.cummax()
            dd_all_series = close / running_max_all - 1
            mdd_all = dd_all_series.min() if len(dd_all_series) > 0 else np.nan

            avg_dollar_volume_20d = (close * volume).tail(20).mean() if len(volume) > 0 else np.nan
            avg_dollar_volume_63d = (close * volume).tail(63).mean() if len(volume) > 0 else np.nan

            momentum_trend = int(price > ma50 and ma50 > ma200)
            above_ma200 = int(price > ma200)
            above_ma50 = int(price > ma50)

            rs_vs_qqq_6m = np.nan
            rs_vs_qqq_12m = np.nan
            if qqq_close is not None and len(qqq_close) > 252:
                qqq_ret_6m = safe_div(qqq_close.iloc[-1], qqq_close.iloc[-126]) - 1
                qqq_ret_12m = safe_div(qqq_close.iloc[-1], qqq_close.iloc[-252]) - 1
                rs_vs_qqq_6m = ret_6m - qqq_ret_6m if not pd.isna(ret_6m) and not pd.isna(qqq_ret_6m) else np.nan
                rs_vs_qqq_12m = ret_12m - qqq_ret_12m if not pd.isna(ret_12m) and not pd.isna(qqq_ret_12m) else np.nan

            atr_proxy = daily_ret.tail(20).abs().mean() * math.sqrt(252) if len(daily_ret) >= 20 else np.nan

            rows.append(
                {
                    "Ticker": ticker,
                    "Price": price,
                    "MA20": ma20,
                    "MA50": ma50,
                    "MA100": ma100,
                    "MA200": ma200,
                    "Ret_1M": ret_1m,
                    "Ret_3M": ret_3m,
                    "Ret_6M": ret_6m,
                    "Ret_12M": ret_12m,
                    "High_52W": high_52w,
                    "Low_52W": low_52w,
                    "Dist_52W_High": dist_52w_high,
                    "Dist_52W_Low": dist_52w_low,
                    "Volatility_1Y": vol_1y,
                    "MDD_1Y": mdd_1y,
                    "MDD_All": mdd_all,
                    "Avg_Dollar_Volume_20D": avg_dollar_volume_20d,
                    "Avg_Dollar_Volume_63D": avg_dollar_volume_63d,
                    "Trend_OK": momentum_trend,
                    "Above_MA50": above_ma50,
                    "Above_MA200": above_ma200,
                    "RS_vs_QQQ_6M": rs_vs_qqq_6m,
                    "RS_vs_QQQ_12M": rs_vs_qqq_12m,
                    "ATR_Proxy": atr_proxy,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


# =========================================================
# 5) SCORING
# =========================================================
def build_scores(df):
    x = df.copy()

    score_cols = [
        "Ret_6M", "Ret_12M", "Dist_52W_High", "RS_vs_QQQ_6M", "RS_vs_QQQ_12M",
        "ROE", "ROA", "OperatingMargin", "GrossMargin", "ProfitMargin",
        "RevenueGrowth", "EpsGrowth", "QuarterlyRevenueGrowth",
        "ForwardPE", "PegRatio", "PriceToBook", "PriceToSales", "EnterpriseToEbitda",
        "Volatility_1Y", "MDD_1Y", "Beta", "DividendYield", "DebtToEquity"
    ]
    for col in score_cols:
        if col in x.columns:
            x[col] = winsorize_series(x[col])

    # Momentum
    x["Score_Mom_12M"] = pct_rank(x["Ret_12M"], higher_is_better=True)
    x["Score_Mom_6M"] = pct_rank(x["Ret_6M"], higher_is_better=True)
    x["Score_RS_12M"] = pct_rank(x["RS_vs_QQQ_12M"], higher_is_better=True)
    x["Score_RS_6M"] = pct_rank(x["RS_vs_QQQ_6M"], higher_is_better=True)
    x["Score_HighDist"] = pct_rank(x["Dist_52W_High"], higher_is_better=True)
    x["Score_Trend"] = (
        0.6 * x["Trend_OK"].fillna(0) + 0.2 * x["Above_MA50"].fillna(0) + 0.2 * x["Above_MA200"].fillna(0)
    )

    x["MomentumScore"] = (
        0.25 * x["Score_Mom_12M"].fillna(0) +
        0.20 * x["Score_Mom_6M"].fillna(0) +
        0.15 * x["Score_RS_12M"].fillna(0) +
        0.10 * x["Score_RS_6M"].fillna(0) +
        0.10 * x["Score_HighDist"].fillna(0) +
        0.20 * x["Score_Trend"].fillna(0)
    )

    # Quality
    x["Score_ROE"] = pct_rank(x["ROE"], higher_is_better=True)
    x["Score_ROA"] = pct_rank(x["ROA"], higher_is_better=True)
    x["Score_OpMargin"] = pct_rank(x["OperatingMargin"], higher_is_better=True)
    x["Score_GrossMargin"] = pct_rank(x["GrossMargin"], higher_is_better=True)
    x["Score_ProfitMargin"] = pct_rank(x["ProfitMargin"], higher_is_better=True)
    x["Score_Debt"] = pct_rank(-x["DebtToEquity"], higher_is_better=True)
    x["Score_CurrentRatio"] = pct_rank(x["CurrentRatio"], higher_is_better=True)

    x["QualityScore"] = (
        0.25 * x["Score_ROE"].fillna(0) +
        0.10 * x["Score_ROA"].fillna(0) +
        0.20 * x["Score_OpMargin"].fillna(0) +
        0.15 * x["Score_GrossMargin"].fillna(0) +
        0.10 * x["Score_ProfitMargin"].fillna(0) +
        0.15 * x["Score_Debt"].fillna(0) +
        0.05 * x["Score_CurrentRatio"].fillna(0)
    )

    # Growth
    x["Score_RevGrowth"] = pct_rank(x["RevenueGrowth"], higher_is_better=True)
    x["Score_EpsGrowth"] = pct_rank(x["EpsGrowth"], higher_is_better=True)
    x["Score_QRevGrowth"] = pct_rank(x["QuarterlyRevenueGrowth"], higher_is_better=True)

    x["GrowthScore"] = (
        0.45 * x["Score_RevGrowth"].fillna(0) +
        0.35 * x["Score_EpsGrowth"].fillna(0) +
        0.20 * x["Score_QRevGrowth"].fillna(0)
    )

    # Value / reasonable growth
    x["Score_ForwardPE"] = pct_rank(-x["ForwardPE"], higher_is_better=True)
    x["Score_PEG"] = pct_rank(-x["PegRatio"], higher_is_better=True)
    x["Score_PB"] = pct_rank(-x["PriceToBook"], higher_is_better=True)
    x["Score_PS"] = pct_rank(-x["PriceToSales"], higher_is_better=True)
    x["Score_EVEBITDA"] = pct_rank(-x["EnterpriseToEbitda"], higher_is_better=True)

    x["ValueScore"] = (
        0.35 * x["Score_ForwardPE"].fillna(0) +
        0.30 * x["Score_PEG"].fillna(0) +
        0.15 * x["Score_PB"].fillna(0) +
        0.10 * x["Score_PS"].fillna(0) +
        0.10 * x["Score_EVEBITDA"].fillna(0)
    )

    # Defensive
    x["Score_Vol"] = pct_rank(-x["Volatility_1Y"], higher_is_better=True)
    x["Score_MDD"] = pct_rank(x["MDD_1Y"], higher_is_better=True)
    x["Score_Beta"] = pct_rank(-x["Beta"], higher_is_better=True)
    x["Score_Div"] = pct_rank(x["DividendYield"], higher_is_better=True)
    x["Score_AboveMA200"] = x["Above_MA200"].fillna(0)

    x["DefensiveScore"] = (
        0.35 * x["Score_Vol"].fillna(0) +
        0.30 * x["Score_MDD"].fillna(0) +
        0.20 * x["Score_Beta"].fillna(0) +
        0.10 * x["Score_Div"].fillna(0) +
        0.05 * x["Score_AboveMA200"].fillna(0)
    )

    # Total multi-factor
    x["TotalScore"] = (
        0.35 * x["MomentumScore"].fillna(0) +
        0.30 * x["QualityScore"].fillna(0) +
        0.20 * x["GrowthScore"].fillna(0) +
        0.15 * x["ValueScore"].fillna(0)
    )

    x["Rank"] = x["TotalScore"].rank(ascending=False, method="min").astype("Int64")

    return x


# =========================================================
# 6) CHART HELPERS
# =========================================================
def bar_chart(df, x_col, y_col, title, color_col=None):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        title=title,
        text_auto=".2f" if y_col.endswith("Score") else False,
    )
    fig.update_layout(height=450, xaxis_title="", yaxis_title=y_col)
    return fig


def scatter_chart(df, x_col, y_col, title, size_col=None, hover_cols=None):
    if hover_cols is None:
        hover_cols = ["Ticker", "Name"]
    valid_cols = [c for c in hover_cols if c in df.columns]
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col if size_col in df.columns else None,
        hover_data=valid_cols,
        title=title,
    )
    fig.update_layout(height=450)
    return fig


def heatmap_scores(df, tickers_ordered):
    view = df[df["Ticker"].isin(tickers_ordered)].copy()
    view = view.set_index("Ticker").loc[tickers_ordered]

    heat = view[
        ["MomentumScore", "QualityScore", "GrowthScore", "ValueScore", "DefensiveScore", "TotalScore"]
    ].T

    fig = px.imshow(heat, aspect="auto", title="Factor Heatmap")
    fig.update_layout(height=450)
    return fig


def plot_selected_stock(price_data, ticker):
    df = extract_single_ticker_frame(price_data, ticker)
    if df.empty or "Close" not in df.columns:
        return go.Figure()

    close = df["Close"].dropna().copy()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, mode="lines", name="MA200"))
    fig.update_layout(
        title=f"{ticker} Price Trend",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return fig


# =========================================================
# 7) LOAD DATA
# =========================================================
with st.spinner("Loading universe and market data..."):
    universe = get_nasdaq100_universe()
    tickers = universe["Ticker"].tolist()

    price_data = download_price_data(tickers, period="2y")
    benchmark_df = download_benchmark(period="2y")
    fundamentals = download_fundamentals(tickers)

if price_data.empty:
    st.error("Failed to download price data from Yahoo Finance.")
    st.stop()

with st.spinner("Calculating metrics and factor scores..."):
    tech = compute_metrics(price_data, tickers, benchmark_df)
    if tech.empty:
        st.error("Price data was downloaded, but metric calculation returned no valid rows.")
        st.stop()

    df = tech.merge(fundamentals, on="Ticker", how="left").merge(universe, on="Ticker", how="left")
    df = build_scores(df)

# =========================================================
# 8) SIDEBAR
# =========================================================
st.sidebar.header("Settings")

top_n = st.sidebar.slider("Top N", min_value=5, max_value=30, value=15, step=1)

min_dollar_volume = st.sidebar.number_input(
    "Min Avg Dollar Volume (63D)",
    min_value=0,
    value=20_000_000,
    step=5_000_000,
)

only_above_ma200 = st.sidebar.checkbox("Only stocks above MA200", value=False)

sector_options = ["All"] + sorted([str(x) for x in df["Sector"].dropna().unique()])
selected_sector = st.sidebar.selectbox("Sector", sector_options, index=0)

selected_detail_ticker = st.sidebar.selectbox("Detail chart ticker", sorted(df["Ticker"].tolist()))

st.sidebar.markdown("---")
st.sidebar.write("**Scoring weights**")
st.sidebar.write("Multi-Factor = 35% Momentum + 30% Quality + 20% Growth + 15% Value")

last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.sidebar.markdown("---")
st.sidebar.caption(f"Last refreshed: {last_updated}")

filtered = df.copy()
filtered = filtered[filtered["Avg_Dollar_Volume_63D"].fillna(0) >= min_dollar_volume]

if only_above_ma200:
    filtered = filtered[filtered["Above_MA200"] == 1]

if selected_sector != "All":
    filtered = filtered[filtered["Sector"] == selected_sector]

if filtered.empty:
    st.warning("No stocks match the current filters.")
    st.stop()

# =========================================================
# 9) TOP SUMMARY
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe Size", len(filtered))
c2.metric("Top Multi-Factor", filtered.sort_values("TotalScore", ascending=False).iloc[0]["Ticker"])
c3.metric("Top Momentum", filtered.sort_values("MomentumScore", ascending=False).iloc[0]["Ticker"])
c4.metric("Top Quality", filtered.sort_values("QualityScore", ascending=False).iloc[0]["Ticker"])

# =========================================================
# 10) TABS
# =========================================================
tabs = st.tabs(
    [
        "Overview",
        "Momentum",
        "Quality",
        "Growth",
        "Value",
        "Defensive",
        "Multi-Factor",
        "Detail Chart",
    ]
)

# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Universe Overview")
    st.write("Broad view of Nasdaq-100 candidates after liquidity and trend filters.")

    col1, col2 = st.columns(2)

    overview_top = filtered.sort_values("Ret_12M", ascending=False).head(top_n)
    with col1:
        st.plotly_chart(
            bar_chart(overview_top, "Ticker", "Ret_12M", f"Top {top_n} by 12M Return"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "Ret_6M",
                "Ret_12M",
                "6M vs 12M Return",
                size_col="MarketCap",
                hover_cols=["Ticker", "Name", "Sector"],
            ),
            use_container_width=True,
        )

    sector_df = (
        filtered.groupby("Sector", dropna=False)["Ticker"]
        .count()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    st.plotly_chart(
        px.pie(sector_df, names="Sector", values="Count", title="Sector Distribution"),
        use_container_width=True,
    )

    show_cols = [
        "Ticker", "Name", "Sector", "Price", "Ret_1M", "Ret_3M", "Ret_6M", "Ret_12M",
        "MA50", "MA200", "Avg_Dollar_Volume_63D", "MarketCap", "Rank"
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]

    display_df = filtered.sort_values("TotalScore", ascending=False)[show_cols].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# TAB 2: MOMENTUM
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Momentum Strategy")
    st.write(
        "Logic: favor strong 6M/12M trend, relative strength vs QQQ, price near 52-week high, and price above key moving averages."
    )

    mom = filtered.sort_values("MomentumScore", ascending=False).head(top_n).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(mom, "Ticker", "MomentumScore", f"Top {top_n} Momentum Ranking"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "Ret_6M",
                "Ret_12M",
                "Momentum Evidence: 6M vs 12M",
                size_col="MomentumScore",
                hover_cols=["Ticker", "Name", "RS_vs_QQQ_12M"],
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            bar_chart(
                mom.sort_values("Dist_52W_High", ascending=False),
                "Ticker",
                "Dist_52W_High",
                "Distance from 52-Week High",
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "RS_vs_QQQ_6M",
                "RS_vs_QQQ_12M",
                "Relative Strength vs QQQ",
                size_col="MomentumScore",
                hover_cols=["Ticker", "Name"],
            ),
            use_container_width=True,
        )

    mom_cols = [
        "Ticker", "Name", "Sector", "Price", "Ret_6M", "Ret_12M",
        "RS_vs_QQQ_6M", "RS_vs_QQQ_12M", "Dist_52W_High",
        "Trend_OK", "Above_MA200", "MomentumScore"
    ]
    st.dataframe(mom[mom_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 3: QUALITY
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Quality Strategy")
    st.write(
        "Logic: favor high ROE/ROA, strong margins, and cleaner balance sheet characteristics."
    )

    q = filtered.sort_values("QualityScore", ascending=False).head(top_n).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(q, "Ticker", "ROE", f"Top {top_n} by ROE"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "ROE",
                "OperatingMargin",
                "ROE vs Operating Margin",
                size_col="QualityScore",
                hover_cols=["Ticker", "Name", "DebtToEquity"],
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "GrossMargin",
                "ProfitMargin",
                "Gross Margin vs Profit Margin",
                size_col="QualityScore",
                hover_cols=["Ticker", "Name"],
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            bar_chart(
                q.sort_values("DebtToEquity", ascending=True),
                "Ticker",
                "DebtToEquity",
                "Debt to Equity",
            ),
            use_container_width=True,
        )

    q_cols = [
        "Ticker", "Name", "Sector", "ROE", "ROA",
        "OperatingMargin", "GrossMargin", "ProfitMargin",
        "DebtToEquity", "CurrentRatio", "QualityScore"
    ]
    st.dataframe(q[q_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 4: GROWTH
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Growth Strategy")
    st.write(
        "Logic: favor revenue growth, earnings growth, and recent quarterly revenue acceleration."
    )

    g = filtered.sort_values("GrowthScore", ascending=False).head(top_n).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(g, "Ticker", "RevenueGrowth", f"Top {top_n} Revenue Growth"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "RevenueGrowth",
                "EpsGrowth",
                "Revenue Growth vs EPS Growth",
                size_col="GrowthScore",
                hover_cols=["Ticker", "Name", "QuarterlyRevenueGrowth"],
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            bar_chart(g, "Ticker", "QuarterlyRevenueGrowth", "Quarterly Revenue Growth"),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "RevenueGrowth",
                "Ret_12M",
                "Growth vs 12M Return",
                size_col="GrowthScore",
                hover_cols=["Ticker", "Name"],
            ),
            use_container_width=True,
        )

    g_cols = [
        "Ticker", "Name", "Sector",
        "RevenueGrowth", "EpsGrowth", "QuarterlyRevenueGrowth", "GrowthScore"
    ]
    st.dataframe(g[g_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 5: VALUE
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Value / Reasonable Growth")
    st.write(
        "Logic: within Nasdaq-100, pure deep value is rare. This tab looks for comparatively cheaper growth names using Forward PE, PEG, P/B, P/S, and EV/EBITDA."
    )

    v = filtered.sort_values("ValueScore", ascending=False).head(top_n).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(v, "Ticker", "ForwardPE", f"Top {top_n} by Value Score"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "ForwardPE",
                "EpsGrowth",
                "Forward PE vs EPS Growth",
                size_col="ValueScore",
                hover_cols=["Ticker", "Name", "PegRatio"],
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "PriceToSales",
                "OperatingMargin",
                "Price/Sales vs Operating Margin",
                size_col="ValueScore",
                hover_cols=["Ticker", "Name"],
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            bar_chart(v, "Ticker", "PegRatio", "PEG Ratio"),
            use_container_width=True,
        )

    v_cols = [
        "Ticker", "Name", "Sector", "ForwardPE", "TrailingPE",
        "PegRatio", "PriceToBook", "PriceToSales", "EnterpriseToEbitda", "ValueScore"
    ]
    st.dataframe(v[v_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 6: DEFENSIVE
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("Defensive Strategy")
    st.write(
        "Logic: favor lower volatility, smaller drawdown, lower beta, and stable price behavior."
    )

    d = filtered.sort_values("DefensiveScore", ascending=False).head(top_n).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(d, "Ticker", "Volatility_1Y", f"Top {top_n} Defensive Ranking"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "Volatility_1Y",
                "Ret_12M",
                "Return vs Volatility",
                size_col="DefensiveScore",
                hover_cols=["Ticker", "Name", "Beta"],
            ),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            bar_chart(
                d.sort_values("MDD_1Y", ascending=False),
                "Ticker",
                "MDD_1Y",
                "1Y Max Drawdown",
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "Beta",
                "MDD_1Y",
                "Beta vs Max Drawdown",
                size_col="DefensiveScore",
                hover_cols=["Ticker", "Name"],
            ),
            use_container_width=True,
        )

    d_cols = [
        "Ticker", "Name", "Sector", "Volatility_1Y",
        "MDD_1Y", "Beta", "DividendYield", "Above_MA200", "DefensiveScore"
    ]
    st.dataframe(d[d_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 7: MULTI-FACTOR
# ---------------------------------------------------------
with tabs[6]:
    st.subheader("Multi-Factor Ranking")
    st.write(
        "Practical blend: 35% Momentum + 30% Quality + 20% Growth + 15% Value."
    )

    mf = filtered.sort_values("TotalScore", ascending=False).head(top_n).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            bar_chart(mf, "Ticker", "TotalScore", f"Top {top_n} Multi-Factor"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            heatmap_scores(mf, mf["Ticker"].tolist()),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "MomentumScore",
                "QualityScore",
                "Momentum vs Quality",
                size_col="TotalScore",
                hover_cols=["Ticker", "Name", "GrowthScore", "ValueScore"],
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            scatter_chart(
                filtered,
                "GrowthScore",
                "ValueScore",
                "Growth vs Value",
                size_col="TotalScore",
                hover_cols=["Ticker", "Name"],
            ),
            use_container_width=True,
        )

    mf_cols = [
        "Rank", "Ticker", "Name", "Sector",
        "MomentumScore", "QualityScore", "GrowthScore",
        "ValueScore", "DefensiveScore", "TotalScore"
    ]
    st.dataframe(mf[mf_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 8: DETAIL CHART
# ---------------------------------------------------------
with tabs[7]:
    st.subheader("Selected Stock Detail")
    st.write("Price trend with MA50 and MA200.")

    detail_row = filtered[filtered["Ticker"] == selected_detail_ticker].copy()

    if detail_row.empty:
        st.info("Selected ticker is filtered out. Change filters or choose another ticker.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        row = detail_row.iloc[0]

        c1.metric("Ticker", row["Ticker"])
        c2.metric("Price", fmt_num(row["Price"], 2))
        c3.metric("12M Return", fmt_pct(row["Ret_12M"]))
        c4.metric("Multi-Factor Rank", str(row["Rank"]))

        st.plotly_chart(plot_selected_stock(price_data, selected_detail_ticker), use_container_width=True)

        detail_cols = [
            "Ticker", "Name", "Sector", "Industry",
            "Price", "Ret_1M", "Ret_3M", "Ret_6M", "Ret_12M",
            "RS_vs_QQQ_6M", "RS_vs_QQQ_12M",
            "ROE", "OperatingMargin", "GrossMargin",
            "RevenueGrowth", "EpsGrowth",
            "ForwardPE", "PegRatio",
            "Volatility_1Y", "MDD_1Y", "Beta",
            "MomentumScore", "QualityScore", "GrowthScore", "ValueScore", "TotalScore"
        ]
        detail_cols = [c for c in detail_cols if c in detail_row.columns]
        st.dataframe(detail_row[detail_cols], use_container_width=True, hide_index=True)

# =========================================================
# 11) FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Notes: This app uses a static Nasdaq-100 list and Yahoo Finance data. "
    "Fundamental fields may be missing for some tickers, and scores are relative within the current filtered universe."
)
