# streamlit_app.py
# ------------------------------------------------------------
# Nasdaq-100 Quant Strategy Dashboard
# ROE / MDD based undervalued stock screening
# + strategy tabs
# + Top 10 MVA analysis
# + MVA charts listed sequentially
# + simple company description + quant interpretation per stock
# + Sector filter
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# Run:
#   streamlit run streamlit_app.py
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Nasdaq-100 Quant Strategy Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Nasdaq-100 Quant Strategy Dashboard")
st.caption(
    "ROE / MDD based undervalued stock screening with strategy tabs + sequential Top 10 MVA analysis"
)


# ============================================================
# Nasdaq-100 universe
# ============================================================
NASDAQ100_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "CDNS", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSX", "CTAS",
    "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC",
    "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP",
    "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META",
    "MNST", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "ODFL", "ON", "ORLY", "PANW",
    "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST",
    "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX",
    "WBD", "WDAY", "XEL", "ZS"
]


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

period = st.sidebar.selectbox(
    "Price lookback",
    ["1y", "2y", "3y", "5y"],
    index=1
)

top_n = st.sidebar.slider(
    "Top N",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

min_roe = st.sidebar.slider(
    "Minimum ROE (%)",
    min_value=0,
    max_value=40,
    value=10,
    step=1
)

min_mktcap_b = st.sidebar.slider(
    "Minimum Market Cap ($B)",
    min_value=0,
    max_value=500,
    value=10,
    step=5
)

refresh = st.sidebar.button("Refresh Data")

if refresh:
    st.cache_data.clear()


# ============================================================
# Data loading helpers
# ============================================================
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

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)

            short_name = ticker
            sector = ""
            industry = ""
            business_summary = ""
            roe = np.nan
            market_cap = np.nan
            trailing_pe = np.nan
            forward_pe = np.nan
            gross_margin = np.nan
            operating_margin = np.nan
            revenue_growth = np.nan
            debt_to_equity = np.nan
            free_cashflow = np.nan

            try:
                info = tk.info
            except Exception:
                info = {}

            if isinstance(info, dict) and len(info) > 0:
                short_name = info.get("shortName", ticker)
                sector = info.get("sector", "")
                industry = info.get("industry", "")
                business_summary = info.get("longBusinessSummary", "")

                roe = info.get("returnOnEquity", np.nan)
                market_cap = info.get("marketCap", np.nan)
                trailing_pe = info.get("trailingPE", np.nan)
                forward_pe = info.get("forwardPE", np.nan)
                gross_margin = info.get("grossMargins", np.nan)
                operating_margin = info.get("operatingMargins", np.nan)
                revenue_growth = info.get("revenueGrowth", np.nan)
                debt_to_equity = info.get("debtToEquity", np.nan)
                free_cashflow = info.get("freeCashflow", np.nan)

            rows.append({
                "Ticker": ticker,
                "ShortName": short_name,
                "Sector": sector,
                "Industry": industry,
                "BusinessSummary": business_summary,
                "ROE": roe * 100 if pd.notna(roe) else np.nan,
                "MarketCap_B": market_cap / 1e9 if pd.notna(market_cap) else np.nan,
                "TrailingPE": trailing_pe,
                "ForwardPE": forward_pe,
                "GrossMargin": gross_margin * 100 if pd.notna(gross_margin) else np.nan,
                "OperatingMargin": operating_margin * 100 if pd.notna(operating_margin) else np.nan,
                "RevenueGrowth": revenue_growth * 100 if pd.notna(revenue_growth) else np.nan,
                "DebtToEquity": debt_to_equity,
                "FCF_B": free_cashflow / 1e9 if pd.notna(free_cashflow) else np.nan,
            })

        except Exception:
            rows.append({
                "Ticker": ticker,
                "ShortName": ticker,
                "Sector": "",
                "Industry": "",
                "BusinessSummary": "",
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


def get_close_series(price_data, ticker):
    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            if ticker in price_data.columns.get_level_values(0):
                s = price_data[ticker]["Close"].dropna()
            else:
                s = pd.Series(dtype=float)
        else:
            if "Close" in price_data.columns:
                s = price_data["Close"].dropna()
            else:
                s = pd.Series(dtype=float)

        return s.astype(float)

    except Exception:
        return pd.Series(dtype=float)


def compute_technical_features(price_data, tickers):
    rows = []

    for ticker in tickers:
        s = get_close_series(price_data, ticker)

        if len(s) < 60:
            continue

        try:
            rolling_peak = s.cummax()
            drawdown = (s / rolling_peak - 1.0) * 100
            mdd = drawdown.min()

            ret_3m = np.nan
            ret_6m = np.nan

            if len(s) > 63:
                ret_3m = (s.iloc[-1] / s.iloc[-63] - 1.0) * 100

            if len(s) > 126:
                ret_6m = (s.iloc[-1] / s.iloc[-126] - 1.0) * 100

            ma50 = s.rolling(50).mean().iloc[-1] if len(s) >= 50 else np.nan
            ma100 = s.rolling(100).mean().iloc[-1] if len(s) >= 100 else np.nan
            ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan

            dist_50ma = ((s.iloc[-1] / ma50) - 1.0) * 100 if pd.notna(ma50) and ma50 != 0 else np.nan
            dist_100ma = ((s.iloc[-1] / ma100) - 1.0) * 100 if pd.notna(ma100) and ma100 != 0 else np.nan
            dist_200ma = ((s.iloc[-1] / ma200) - 1.0) * 100 if pd.notna(ma200) and ma200 != 0 else np.nan

            vol_1y = s.pct_change().dropna().std() * np.sqrt(252) * 100

            rows.append({
                "Ticker": ticker,
                "Price": s.iloc[-1],
                "MDD": mdd,
                "Momentum_3M": ret_3m,
                "Momentum_6M": ret_6m,
                "MA50": ma50,
                "MA100": ma100,
                "MA200": ma200,
                "Dist_50MA": dist_50ma,
                "Dist_100MA": dist_100ma,
                "Dist_200MA": dist_200ma,
                "Volatility_1Y": vol_1y,
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# Ranking / scoring helpers
# ============================================================
def rank_high(series):
    return series.rank(pct=True, ascending=True)


def rank_low(series):
    return 1 - series.rank(pct=True, ascending=True)


def build_scores(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    df["r_ROE"] = rank_high(df["ROE"])
    df["r_GrossMargin"] = rank_high(df["GrossMargin"])
    df["r_OpMargin"] = rank_high(df["OperatingMargin"])
    df["r_RevenueGrowth"] = rank_high(df["RevenueGrowth"])
    df["r_Mom6"] = rank_high(df["Momentum_6M"])
    df["r_FCF"] = rank_high(df["FCF_B"])

    df["r_MDD"] = rank_low(df["MDD"])
    df["r_Debt"] = rank_low(df["DebtToEquity"])
    df["r_Vol"] = rank_low(df["Volatility_1Y"])
    df["r_PE"] = rank_low(df["ForwardPE"].fillna(df["TrailingPE"]))

    df["Score_ROE_MDD"] = (
        0.60 * df["r_ROE"] +
        0.40 * df["r_MDD"]
    )

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
        "Ticker",
        "ShortName",
        "Sector",
        "Industry",
        "BusinessSummary",
        "ROE",
        "MDD",
        "Momentum_3M",
        "Momentum_6M",
        "Price",
        "MA50",
        "MA100",
        "MA200",
        "Dist_50MA",
        "Dist_100MA",
        "Dist_200MA",
        "Volatility_1Y",
        "MarketCap_B",
        "ForwardPE",
        "TrailingPE",
        "RevenueGrowth",
        "GrossMargin",
        "OperatingMargin",
        "DebtToEquity",
        "FCF_B",
        score_col,
    ]
    cols = [c for c in cols if c in df.columns]

    out = (
        df.sort_values(score_col, ascending=False)[cols]
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    return out


# ============================================================
# Formatting helpers
# ============================================================
def styled_table(df):
    format_dict = {}

    for col in df.columns:
        if col in ["Ticker", "ShortName", "Sector", "Industry", "BusinessSummary"]:
            continue
        elif col in ["MarketCap_B", "FCF_B", "Price", "MA50", "MA100", "MA200"]:
            format_dict[col] = "{:.2f}"
        elif "Score_" in col:
            format_dict[col] = "{:.3f}"
        else:
            format_dict[col] = "{:.2f}"

    return df.style.format(format_dict)


def safe_text(val, default="-"):
    if val is None:
        return default
    if isinstance(val, float) and pd.isna(val):
        return default
    text = str(val).strip()
    return text if text else default


def summarize_business(text, max_len=220):
    if not isinstance(text, str) or not text.strip():
        return "Business description not available."
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


# ============================================================
# Interpretation helpers
# ============================================================
def interpret_mva(dist50, dist100, dist200):
    if pd.notna(dist50) and pd.notna(dist100) and pd.notna(dist200):
        if dist50 > 0 and dist100 > 0 and dist200 > 0:
            return "Price is above MA50, MA100, and MA200, which suggests a relatively strong trend."
        if dist50 > 0 and dist100 > 0 and dist200 < 0:
            return "Short- and mid-term recovery is visible, but the stock is still below its long-term trend."
        if dist50 > 0 and dist200 < 0:
            return "Short-term recovery is visible, but the stock remains below MA200."
        if dist50 < 0 and dist100 < 0 and dist200 < 0:
            return "Price is below all major moving averages, which suggests a weak technical trend."
        if dist200 < 0:
            return "The stock remains below MA200, so it may still be in a discounted long-term zone."
    return "Technical positioning is mixed across moving averages."


def interpret_quant_style(row, score_col):
    roe = row.get("ROE", np.nan)
    mdd = row.get("MDD", np.nan)
    mom6 = row.get("Momentum_6M", np.nan)
    vol = row.get("Volatility_1Y", np.nan)
    rev = row.get("RevenueGrowth", np.nan)
    debt = row.get("DebtToEquity", np.nan)

    comments = []

    if pd.notna(roe):
        if roe >= 20:
            comments.append("high profitability")
        elif roe >= 10:
            comments.append("decent profitability")
        else:
            comments.append("weaker profitability")

    if pd.notna(mdd):
        if mdd <= -30:
            comments.append("deep drawdown")
        elif mdd <= -15:
            comments.append("meaningful pullback")
        else:
            comments.append("limited drawdown")

    if score_col == "Score_Recovery_Momentum":
        if pd.notna(mom6):
            if mom6 > 10:
                comments.append("clear recovery momentum")
            elif mom6 > 0:
                comments.append("mild recovery momentum")
            else:
                comments.append("weak recent momentum")

        if pd.notna(rev):
            if rev > 10:
                comments.append("solid revenue growth")
            elif rev < 0:
                comments.append("negative revenue growth")

    if score_col == "Score_LowVol_Pullback":
        if pd.notna(vol):
            if vol < 25:
                comments.append("relatively stable volatility")
            else:
                comments.append("still volatile")

    if score_col == "Score_Quality_Pullback":
        if pd.notna(debt):
            if debt < 50:
                comments.append("manageable leverage")
            elif debt > 150:
                comments.append("higher leverage")

    if not comments:
        return "Quant interpretation is limited due to missing data."

    return " / ".join(comments).capitalize() + "."


def show_stock_description(row, score_col):
    ticker = safe_text(row.get("Ticker"))
    short_name = safe_text(row.get("ShortName"), ticker)
    sector = safe_text(row.get("Sector"))
    industry = safe_text(row.get("Industry"))
    summary = summarize_business(row.get("BusinessSummary", ""))
    quant_comment = interpret_quant_style(row, score_col)
    mva_comment = interpret_mva(
        row.get("Dist_50MA", np.nan),
        row.get("Dist_100MA", np.nan),
        row.get("Dist_200MA", np.nan),
    )

    st.markdown(f"**{ticker} — {short_name}**")
    st.caption(f"Sector: {sector} | Industry: {industry}")
    st.write(summary)
    st.info(f"Quant view: {quant_comment} {mva_comment}")


# ============================================================
# Chart helpers
# ============================================================
def plot_mva_chart(price_data, ticker, chart_key):
    s = get_close_series(price_data, ticker)

    if len(s) < 30:
        st.warning(f"No sufficient price history for {ticker}")
        return

    dfp = pd.DataFrame({"Close": s})
    dfp["MA50"] = dfp["Close"].rolling(50).mean()
    dfp["MA100"] = dfp["Close"].rolling(100).mean()
    dfp["MA200"] = dfp["Close"].rolling(200).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dfp.index,
        y=dfp["Close"],
        mode="lines",
        name=f"{ticker} Price"
    ))
    fig.add_trace(go.Scatter(
        x=dfp.index,
        y=dfp["MA50"],
        mode="lines",
        name="MA50"
    ))
    fig.add_trace(go.Scatter(
        x=dfp.index,
        y=dfp["MA100"],
        mode="lines",
        name="MA100"
    ))
    fig.add_trace(go.Scatter(
        x=dfp.index,
        y=dfp["MA200"],
        mode="lines",
        name="MA200"
    ))

    fig.update_layout(
        title=f"{ticker} Price vs Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        height=420,
        legend_orientation="h",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def plot_top10_mva_distance_bar(top_df, title, chart_key):
    needed_cols = ["Ticker", "Dist_50MA", "Dist_100MA", "Dist_200MA"]
    available_cols = [c for c in needed_cols if c in top_df.columns]

    if len(available_cols) < 4:
        st.warning("Not enough MVA data to display the bar chart.")
        return

    temp = top_df[needed_cols].copy()

    temp = temp.melt(
        id_vars="Ticker",
        var_name="Metric",
        value_name="Value"
    )

    fig = px.bar(
        temp,
        x="Ticker",
        y="Value",
        color="Metric",
        barmode="group",
        title=title
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def plot_score_bar(out, score_col, strategy_title, chart_key):
    fig = px.bar(
        out.sort_values(score_col, ascending=False),
        x="Ticker",
        y=score_col,
        title=f"Top {len(out)} - {strategy_title} Score Ranking"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def plot_roe_mdd_scatter(out, score_col, strategy_title, chart_key):
    fig = px.scatter(
        out,
        x="MDD",
        y="ROE",
        size="MarketCap_B" if "MarketCap_B" in out.columns else None,
        color=score_col,
        hover_name="Ticker",
        title=f"Top {len(out)} - {strategy_title}: ROE vs MDD"
    )

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def plot_overview_scatter(df, chart_key):
    fig = px.scatter(
        df,
        x="MDD",
        y="ROE",
        size="MarketCap_B" if "MarketCap_B" in df.columns else None,
        color="Score_ROE_MDD",
        hover_name="Ticker",
        title="Nasdaq-100: ROE vs MDD"
    )

    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def show_mva_metrics(row):
    c1, c2, c3 = st.columns(3)

    d50 = row.get("Dist_50MA", np.nan)
    d100 = row.get("Dist_100MA", np.nan)
    d200 = row.get("Dist_200MA", np.nan)

    c1.metric("Dist from MA50", "-" if pd.isna(d50) else f"{d50:.2f}%")
    c2.metric("Dist from MA100", "-" if pd.isna(d100) else f"{d100:.2f}%")
    c3.metric("Dist from MA200", "-" if pd.isna(d200) else f"{d200:.2f}%")


def show_all_mva_charts(price_data, out, score_col, section_prefix):
    st.markdown(f"#### Top {len(out)} Individual MVA Charts")

    for idx, ticker in enumerate(out["Ticker"].tolist(), start=1):
        row = out[out["Ticker"] == ticker].iloc[0]

        st.markdown(f"##### {idx}. {ticker}")
        show_stock_description(row, score_col)
        show_mva_metrics(row)

        plot_mva_chart(
            price_data=price_data,
            ticker=ticker,
            chart_key=f"{section_prefix}_{score_col}_{ticker}_mva_chart"
        )

        if idx < len(out):
            st.markdown("---")


# ============================================================
# Strategy section renderer
# ============================================================
def show_strategy_section(df, price_data, score_col, strategy_title, strategy_desc, top_n):
    st.markdown(f"### {strategy_title}")
    st.markdown(strategy_desc)

    out = top_table(df, score_col, top_n)

    if out.empty:
        st.warning("No stocks matched the current filters.")
        return

    st.dataframe(styled_table(out), use_container_width=True)

    plot_score_bar(
        out=out,
        score_col=score_col,
        strategy_title=strategy_title,
        chart_key=f"{score_col}_score_bar"
    )

    plot_roe_mdd_scatter(
        out=out,
        score_col=score_col,
        strategy_title=strategy_title,
        chart_key=f"{score_col}_roe_mdd_scatter"
    )

    st.markdown(f"#### Top {len(out)} MVA Distance Analysis")
    plot_top10_mva_distance_bar(
        top_df=out,
        title=f"{strategy_title}: Price Distance from MA50 / MA100 / MA200",
        chart_key=f"{score_col}_mva_distance_bar"
    )

    show_all_mva_charts(
        price_data=price_data,
        out=out,
        score_col=score_col,
        section_prefix="strategy"
    )


# ============================================================
# Load data
# ============================================================
with st.spinner("Loading Nasdaq-100 market data..."):
    price_data = load_price_data(NASDAQ100_TICKERS, period=period)
    fundamentals_df = load_fundamentals(NASDAQ100_TICKERS)
    technical_df = compute_technical_features(price_data, NASDAQ100_TICKERS)

# ============================================================
# Sector selector (after fundamentals loaded)
# ============================================================
sector_list = sorted(
    [s for s in fundamentals_df["Sector"].dropna().unique().tolist() if str(s).strip()]
)

selected_sectors = st.sidebar.multiselect(
    "Business Sector",
    options=sector_list,
    default=[],
    help="Leave empty to analyze all sectors."
)

df = fundamentals_df.merge(technical_df, on="Ticker", how="inner")

df = df[
    (df["MarketCap_B"].fillna(0) >= min_mktcap_b) &
    (df["ROE"].fillna(-999) >= min_roe)
].copy()

if selected_sectors:
    df = df[df["Sector"].isin(selected_sectors)].copy()

df = build_scores(df)

st.subheader("Filtered Universe")
if selected_sectors:
    st.write(
        f"Number of stocks after filters: **{len(df)}**  |  Selected sector(s): **{', '.join(selected_sectors)}**"
    )
else:
    st.write(f"Number of stocks after filters: **{len(df)}**  |  Selected sector(s): **All**")

if df.empty:
    st.error("No stocks matched the current filter settings. Try lowering the ROE / market cap threshold or changing the sector filter.")
    st.stop()


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "ROE + MDD",
    "Quality + Pullback",
    "Recovery Momentum",
    "Low Vol Pullback",
])


with tab1:
    st.markdown("""
### Overview
This dashboard uses the **Nasdaq-100 universe** and compares several quant styles for finding
**undervalued but fundamentally strong stocks**.

**Core interpretation**
- **ROE**: profitability / business quality
- **MDD**: how far the stock is below its previous peak
- **Momentum**: whether recovery has started
- **MVA**: where the current price sits relative to MA50 / MA100 / MA200
""")

    best = top_table(df, "Score_ROE_MDD", top_n)

    st.markdown("#### Top Picks by ROE + MDD")
    st.dataframe(styled_table(best), use_container_width=True)

    plot_overview_scatter(
        df=df,
        chart_key="overview_roe_mdd_scatter"
    )

    st.markdown("#### Overview MVA Distance of Top Picks")
    plot_top10_mva_distance_bar(
        top_df=best,
        title="Overview Top Picks: Price Distance from MA50 / MA100 / MA200",
        chart_key="overview_mva_distance_bar"
    )

    show_all_mva_charts(
        price_data=price_data,
        out=best,
        score_col="Score_ROE_MDD",
        section_prefix="overview"
    )


with tab2:
    show_strategy_section(
        df=df,
        price_data=price_data,
        score_col="Score_ROE_MDD",
        strategy_title="Strategy 1: ROE + MDD",
        strategy_desc="""
**Goal:** Find high-quality companies that are currently in a meaningful drawdown.

**Interpretation**
- High **ROE** = strong business quality
- Deep **MDD** = potentially discounted price
- **MVA** helps check whether the stock is still below medium- and long-term averages

This is the most direct **quality on sale** strategy.
""",
        top_n=top_n
    )


with tab3:
    show_strategy_section(
        df=df,
        price_data=price_data,
        score_col="Score_Quality_Pullback",
        strategy_title="Strategy 2: Quality + Pullback",
        strategy_desc="""
**Goal:** Prefer strong businesses with healthy margins, better balance-sheet quality,
and a meaningful pullback from prior highs.

**Factors**
- ROE
- Gross Margin
- Operating Margin
- Debt to Equity
- MDD

This is a more **fundamental quality-focused pullback** strategy.
""",
        top_n=top_n
    )


with tab4:
    show_strategy_section(
        df=df,
        price_data=price_data,
        score_col="Score_Recovery_Momentum",
        strategy_title="Strategy 3: Recovery Momentum",
        strategy_desc="""
**Goal:** Find stocks that were hit hard but are starting to recover.

**Factors**
- ROE
- MDD
- 6M Momentum
- Revenue Growth

This helps avoid stocks that are cheap for structurally weak reasons.
""",
        top_n=top_n
    )


with tab5:
    show_strategy_section(
        df=df,
        price_data=price_data,
        score_col="Score_LowVol_Pullback",
        strategy_title="Strategy 4: Low Vol Pullback",
        strategy_desc="""
**Goal:** Find quality stocks in drawdown, but with relatively lower volatility.

**Factors**
- ROE
- MDD
- Volatility
- PE

Useful when you want a more **stable pullback strategy**.
""",
        top_n=top_n
    )


# ============================================================
# Raw data
# ============================================================
with st.expander("Show raw merged data"):
    st.dataframe(
        styled_table(
            df.sort_values("Score_ROE_MDD", ascending=False).reset_index(drop=True)
        ),
        use_container_width=True
    )
