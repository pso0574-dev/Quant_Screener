# streamlit_app.py
# ---------------------------------------------------------
# NASDAQ Growth Screener
# Filters:
#   - Revenue growth > threshold
#   - ROIC rising
#   - MDD > threshold
#   - Large TAM sectors
#
# Run:
#   pip install streamlit yfinance pandas numpy plotly
#   streamlit run streamlit_app.py
# ---------------------------------------------------------

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
    page_title="NASDAQ Growth Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NASDAQ Growth Screener")
st.caption("Revenue Growth + ROIC Trend + MDD + TAM Sector filter")

# =========================================================
# Universe
# Fixed Nasdaq-100 style universe for stability
# =========================================================
NASDAQ_UNIVERSE = sorted(list(set([
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA",
    "COST", "NFLX", "ASML", "PEP", "AMD", "ADBE", "CSCO", "TMUS", "INTU",
    "QCOM", "AMGN", "TXN", "HON", "AMAT", "ISRG", "BKNG", "VRTX", "PDD",
    "ADI", "MU", "PANW", "LRCX", "KLAC", "SNPS", "CDNS", "CRWD", "FTNT",
    "ADP", "PYPL", "ABNB", "MAR", "MELI", "ORLY", "REGN", "MDLZ", "CMCSA",
    "CSX", "ROP", "NXPI", "MRVL", "ADSK", "WDAY", "IDXX", "KDP", "PCAR",
    "PAYX", "CHTR", "AEP", "FAST", "DDOG", "TEAM", "ROST", "EA", "ODFL",
    "CPRT", "MNST", "CTAS", "VRSK", "XEL", "CTSH", "LULU", "MCHP", "EXC",
    "DXCM", "FANG", "BKR", "GEHC", "ON", "BIIB", "GILD", "KHC", "SBUX",
    "ZS", "ANSS", "DLTR", "ILMN", "TTWO", "MDB", "WBD", "APP", "PLTR",
    "SNOW", "NET", "SHOP", "ARM"
])))

# =========================================================
# TAM sector mapping
# User can adjust / expand later
# =========================================================
TAM_BUCKETS = {
    "AI / Semiconductors": [
        "NVDA", "AMD", "AVGO", "MRVL", "ARM", "AMAT", "LRCX", "KLAC", "MCHP",
        "QCOM", "TXN", "MU", "ASML", "ADI", "NXPI", "ON"
    ],
    "Cloud / Data / Software": [
        "MSFT", "SNOW", "MDB", "DDOG", "TEAM", "ADBE", "INTU", "ADSK",
        "WDAY", "CDNS", "SNPS", "ANSS", "PLTR"
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "FTNT", "ZS", "NET"
    ],
    "Internet / Platform / Ads": [
        "META", "GOOGL", "GOOG", "AMZN", "NFLX", "APP", "ABNB", "MELI", "SHOP"
    ],
    "Automation / Enterprise Efficiency": [
        "ROP", "VRSK", "CTAS", "ADP", "PAYX", "ORLY", "CPRT"
    ],
    "Healthcare Innovation": [
        "ISRG", "VRTX", "REGN", "DXCM", "GEHC", "IDXX", "BIIB", "GILD", "ILMN"
    ],
    "Consumer Scale Platforms": [
        "COST", "PEP", "MDLZ", "SBUX", "LULU", "KDP", "KHC", "ROST", "DLTR"
    ]
}

TAM_ORDER = list(TAM_BUCKETS.keys())

# Reverse map
TICKER_TO_TAM = {}
for tam_name, tickers in TAM_BUCKETS.items():
    for t in tickers:
        TICKER_TO_TAM[t] = tam_name

# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Settings")

selected_universe = st.sidebar.multiselect(
    "Universe",
    options=NASDAQ_UNIVERSE,
    default=NASDAQ_UNIVERSE[:]
)

tam_filter = st.sidebar.multiselect(
    "TAM Sector Filter",
    options=TAM_ORDER,
    default=TAM_ORDER
)

rev_growth_threshold = st.sidebar.slider(
    "Minimum Revenue Growth YoY (%)",
    min_value=0,
    max_value=100,
    value=25,
    step=5
)

mdd_threshold = st.sidebar.slider(
    "Minimum Drawdown from Peak (%)",
    min_value=0,
    max_value=90,
    value=30,
    step=5
)

min_market_cap_b = st.sidebar.slider(
    "Minimum Market Cap (Billion USD)",
    min_value=0,
    max_value=500,
    value=5,
    step=5
)

lookback_years = st.sidebar.slider(
    "Price Lookback (Years)",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

top_n = st.sidebar.slider(
    "Top candidates to display",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)

show_all = st.sidebar.checkbox("Show all screened stocks", value=False)

refresh = st.sidebar.button("🔄 Refresh data")

# =========================================================
# Cache helpers
# =========================================================
TTL_SEC = 60 * 60 * 6  # 6 hours

if refresh:
    st.cache_data.clear()
    st.success("Cache cleared. Data will be refreshed.")

# =========================================================
# Utility functions
# =========================================================
def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def calc_mdd_from_close(close_series: pd.Series):
    if close_series is None or close_series.empty:
        return np.nan, np.nan, np.nan
    running_max = close_series.cummax()
    dd = close_series / running_max - 1.0
    mdd = dd.min()
    current_dd = dd.iloc[-1]
    peak_price = running_max.iloc[-1]
    return float(mdd * 100), float(current_dd * 100), float(peak_price)

def pick_latest_valid_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    valid_cols = [c for c in df.columns if pd.notna(df[c]).sum() > 0]
    if not valid_cols:
        return None
    return valid_cols[0]

def get_revenue_growth_yoy(quarterly_income_stmt: pd.DataFrame):
    """
    Uses quarterly total revenue if possible.
    Need at least 5 quarters to compare latest quarter vs same quarter last year.
    """
    if quarterly_income_stmt is None or quarterly_income_stmt.empty:
        return np.nan, []

    possible_rows = [
        "Total Revenue",
        "Revenue",
        "Operating Revenue"
    ]

    revenue_row = None
    for r in possible_rows:
        if r in quarterly_income_stmt.index:
            revenue_row = r
            break

    if revenue_row is None:
        return np.nan, []

    s = quarterly_income_stmt.loc[revenue_row].dropna()
    if len(s) < 5:
        return np.nan, list(s.values)

    # yfinance usually returns latest quarter first
    latest = safe_float(s.iloc[0])
    prev_year_same_q = safe_float(s.iloc[4])

    if np.isnan(latest) or np.isnan(prev_year_same_q) or prev_year_same_q == 0:
        return np.nan, list(s.values)

    yoy = (latest / prev_year_same_q - 1.0) * 100
    return float(yoy), list(s.values)

def get_balance_value(balance_sheet: pd.DataFrame, candidate_rows):
    if balance_sheet is None or balance_sheet.empty:
        return np.nan
    for row in candidate_rows:
        if row in balance_sheet.index:
            s = balance_sheet.loc[row].dropna()
            if len(s) > 0:
                return safe_float(s.iloc[0])
    return np.nan

def get_income_value(income_stmt: pd.DataFrame, candidate_rows):
    if income_stmt is None or income_stmt.empty:
        return np.nan
    for row in candidate_rows:
        if row in income_stmt.index:
            s = income_stmt.loc[row].dropna()
            if len(s) > 0:
                return safe_float(s.iloc[0])
    return np.nan

def estimate_roic(financials: dict):
    """
    Approximate ROIC:
      ROIC = NOPAT / Invested Capital

    NOPAT ≈ EBIT * (1 - tax_rate)
    Invested Capital ≈ Total Equity + Total Debt - Cash

    This is an approximation because Yahoo data is not perfectly standardized.
    """
    income_stmt = financials.get("income_stmt")
    balance_sheet = financials.get("balance_sheet")

    ebit = get_income_value(income_stmt, ["EBIT", "Operating Income"])
    pretax = get_income_value(income_stmt, ["Pretax Income"])
    tax_expense = get_income_value(income_stmt, ["Tax Provision", "Tax Expense"])
    total_equity = get_balance_value(balance_sheet, [
        "Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"
    ])
    total_debt = get_balance_value(balance_sheet, [
        "Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"
    ])
    cash = get_balance_value(balance_sheet, [
        "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"
    ])

    if np.isnan(ebit):
        return np.nan

    if not np.isnan(pretax) and pretax != 0 and not np.isnan(tax_expense):
        tax_rate = tax_expense / pretax
        tax_rate = min(max(tax_rate, 0), 0.35)
    else:
        tax_rate = 0.21

    nopat = ebit * (1 - tax_rate)

    if np.isnan(total_equity):
        return np.nan
    if np.isnan(total_debt):
        total_debt = 0.0
    if np.isnan(cash):
        cash = 0.0

    invested_capital = total_equity + total_debt - cash
    if invested_capital <= 0:
        return np.nan

    roic = nopat / invested_capital * 100
    return float(roic)

def estimate_roic_trend(quarterly_income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame):
    """
    Simpler trend proxy:
    Compare latest approximate ROIC vs older one if possible.
    Because yfinance time-series balance sheet consistency is limited,
    we use a weak but practical approximation from available columns.
    """
    if quarterly_income_stmt is None or quarterly_income_stmt.empty:
        return np.nan, np.nan, False

    # Need enough columns
    qcols = list(quarterly_income_stmt.columns)
    bcols = list(balance_sheet.columns) if balance_sheet is not None and not balance_sheet.empty else []

    if len(qcols) < 2 or len(bcols) < 2:
        return np.nan, np.nan, False

    # Build two snapshots: latest and previous
    try:
        latest_income = quarterly_income_stmt[[qcols[0]]]
        prev_income = quarterly_income_stmt[[qcols[1]]]
        latest_bs = balance_sheet[[bcols[0]]] if len(bcols) > 0 else pd.DataFrame()
        prev_bs = balance_sheet[[bcols[1]]] if len(bcols) > 1 else pd.DataFrame()
    except Exception:
        return np.nan, np.nan, False

    latest_roic = estimate_roic({
        "income_stmt": latest_income,
        "balance_sheet": latest_bs
    })
    prev_roic = estimate_roic({
        "income_stmt": prev_income,
        "balance_sheet": prev_bs
    })

    rising = False
    if not np.isnan(latest_roic) and not np.isnan(prev_roic):
        rising = latest_roic > prev_roic

    return latest_roic, prev_roic, rising

def interpret_stock(row):
    notes = []

    if row["Revenue Growth YoY %"] >= 40:
        notes.append("very strong revenue momentum")
    elif row["Revenue Growth YoY %"] >= 25:
        notes.append("good revenue growth")

    if row["ROIC Rising"]:
        notes.append("capital efficiency improving")

    if row["MDD %"] <= -50:
        notes.append("deep correction from previous peak")
    elif row["MDD %"] <= -30:
        notes.append("meaningful pullback")

    if row["TAM Sector"] in ["AI / Semiconductors", "Cybersecurity", "Cloud / Data / Software"]:
        notes.append("large structural TAM")

    if len(notes) == 0:
        return "mixed signals"
    return ", ".join(notes)

# =========================================================
# Data loading
# =========================================================
@st.cache_data(ttl=TTL_SEC, show_spinner=False)
def download_price_data(ticker, period_years=3):
    end = datetime.today()
    start = end - timedelta(days=365 * period_years + 10)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=TTL_SEC, show_spinner=False)
def fetch_ticker_data(ticker):
    tk = yf.Ticker(ticker)

    info = {}
    try:
        info = tk.info
    except Exception:
        info = {}

    try:
        quarterly_income_stmt = tk.quarterly_income_stmt
    except Exception:
        quarterly_income_stmt = pd.DataFrame()

    try:
        quarterly_balance_sheet = tk.quarterly_balance_sheet
    except Exception:
        quarterly_balance_sheet = pd.DataFrame()

    try:
        annual_income_stmt = tk.income_stmt
    except Exception:
        annual_income_stmt = pd.DataFrame()

    try:
        annual_balance_sheet = tk.balance_sheet
    except Exception:
        annual_balance_sheet = pd.DataFrame()

    # fallback if quarterly unavailable
    if quarterly_income_stmt is None or quarterly_income_stmt.empty:
        quarterly_income_stmt = annual_income_stmt.copy()

    if quarterly_balance_sheet is None or quarterly_balance_sheet.empty:
        quarterly_balance_sheet = annual_balance_sheet.copy()

    return {
        "info": info,
        "quarterly_income_stmt": quarterly_income_stmt,
        "quarterly_balance_sheet": quarterly_balance_sheet,
        "income_stmt": annual_income_stmt,
        "balance_sheet": annual_balance_sheet
    }

def screen_one_ticker(ticker, years):
    try:
        raw = fetch_ticker_data(ticker)
        info = raw["info"]
        qis = raw["quarterly_income_stmt"]
        qbs = raw["quarterly_balance_sheet"]
        ais = raw["income_stmt"]
        abs_ = raw["balance_sheet"]

        price = download_price_data(ticker, years)
        close = price["Close"].dropna() if "Close" in price.columns else pd.Series(dtype=float)

        revenue_growth, _ = get_revenue_growth_yoy(qis)
        latest_roic, prev_roic, roic_rising = estimate_roic_trend(qis, qbs)

        # Annual fallback if quarterly trend unavailable
        if np.isnan(latest_roic):
            latest_roic = estimate_roic({
                "income_stmt": ais,
                "balance_sheet": abs_
            })

        mdd, current_dd, peak_price = calc_mdd_from_close(close)

        market_cap = safe_float(info.get("marketCap"))
        market_cap_b = market_cap / 1e9 if not np.isnan(market_cap) else np.nan

        company_name = info.get("shortName", ticker)
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        tam_sector = TICKER_TO_TAM.get(ticker, "Other")

        score = 0
        if not np.isnan(revenue_growth):
            score += min(max(revenue_growth, 0), 100) * 0.35
        if not np.isnan(latest_roic):
            score += min(max(latest_roic, 0), 50) * 0.25
        if roic_rising:
            score += 10
        if not np.isnan(mdd):
            score += min(abs(mdd), 80) * 0.20
        if tam_sector in ["AI / Semiconductors", "Cybersecurity", "Cloud / Data / Software"]:
            score += 10

        return {
            "Ticker": ticker,
            "Company": company_name,
            "Sector": sector,
            "Industry": industry,
            "TAM Sector": tam_sector,
            "Market Cap (B$)": market_cap_b,
            "Revenue Growth YoY %": revenue_growth,
            "ROIC Latest %": latest_roic,
            "ROIC Previous %": prev_roic,
            "ROIC Rising": roic_rising,
            "MDD %": mdd,
            "Current DD %": current_dd,
            "Peak Price": peak_price,
            "Score": score
        }

    except Exception as e:
        return {
            "Ticker": ticker,
            "Company": ticker,
            "Sector": "Error",
            "Industry": str(e),
            "TAM Sector": "Other",
            "Market Cap (B$)": np.nan,
            "Revenue Growth YoY %": np.nan,
            "ROIC Latest %": np.nan,
            "ROIC Previous %": np.nan,
            "ROIC Rising": False,
            "MDD %": np.nan,
            "Current DD %": np.nan,
            "Peak Price": np.nan,
            "Score": np.nan
        }

# =========================================================
# Main screening
# =========================================================
run_button = st.button("Run Screener", type="primary")

if run_button:
    tickers = [
        t for t in selected_universe
        if TICKER_TO_TAM.get(t, "Other") in tam_filter
    ]

    if len(tickers) == 0:
        st.warning("No tickers selected after TAM filter.")
        st.stop()

    progress = st.progress(0)
    results = []

    start_time = time.time()
    for i, ticker in enumerate(tickers):
        results.append(screen_one_ticker(ticker, lookback_years))
        progress.progress((i + 1) / len(tickers))

    df = pd.DataFrame(results)

    numeric_cols = [
        "Market Cap (B$)", "Revenue Growth YoY %", "ROIC Latest %",
        "ROIC Previous %", "MDD %", "Current DD %", "Peak Price", "Score"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter conditions
    screened = df.copy()
    screened = screened[screened["TAM Sector"].isin(tam_filter)]
    screened = screened[screened["Market Cap (B$)"] >= min_market_cap_b]
    screened = screened[screened["Revenue Growth YoY %"] >= rev_growth_threshold]
    screened = screened[screened["ROIC Rising"] == True]
    screened = screened[screened["MDD %"] <= -mdd_threshold]

    screened = screened.sort_values(
        by=["Score", "Revenue Growth YoY %", "ROIC Latest %"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    elapsed = time.time() - start_time
    st.success(f"Done. {len(screened)} stocks matched in {elapsed:.1f} sec.")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe Size", len(tickers))
    col2.metric("Matched Stocks", len(screened))
    col3.metric("Revenue Filter", f">{rev_growth_threshold}%")
    col4.metric("MDD Filter", f">{mdd_threshold}%")

    # Full screened table
    st.subheader("📋 Screened Results")
    if len(screened) == 0:
        st.info("No stocks matched all conditions. Try lowering the thresholds.")
    else:
        screened["Interpretation"] = screened.apply(interpret_stock, axis=1)

        display_cols = [
            "Ticker", "Company", "TAM Sector", "Sector", "Industry",
            "Market Cap (B$)", "Revenue Growth YoY %", "ROIC Latest %",
            "ROIC Previous %", "ROIC Rising", "MDD %", "Current DD %",
            "Score", "Interpretation"
        ]

        if show_all:
            st.dataframe(
                screened[display_cols].style.format({
                    "Market Cap (B$)": "{:.1f}",
                    "Revenue Growth YoY %": "{:.1f}",
                    "ROIC Latest %": "{:.1f}",
                    "ROIC Previous %": "{:.1f}",
                    "MDD %": "{:.1f}",
                    "Current DD %": "{:.1f}",
                    "Score": "{:.1f}"
                }),
                use_container_width=True,
                height=500
            )
        else:
            st.dataframe(
                screened.head(top_n)[display_cols].style.format({
                    "Market Cap (B$)": "{:.1f}",
                    "Revenue Growth YoY %": "{:.1f}",
                    "ROIC Latest %": "{:.1f}",
                    "ROIC Previous %": "{:.1f}",
                    "MDD %": "{:.1f}",
                    "Current DD %": "{:.1f}",
                    "Score": "{:.1f}"
                }),
                use_container_width=True,
                height=420
            )

        # =====================================================
        # Candidate charts
        # =====================================================
        st.subheader("📊 Top Candidate Analysis")

        top_df = screened.head(top_n).copy()

        for _, row in top_df.iterrows():
            ticker = row["Ticker"]
            name = row["Company"]

            st.markdown(f"### {ticker} — {name}")
            st.write(
                f"**TAM:** {row['TAM Sector']}  |  "
                f"**Revenue Growth:** {row['Revenue Growth YoY %']:.1f}%  |  "
                f"**ROIC:** {row['ROIC Latest %']:.1f}%  |  "
                f"**MDD:** {row['MDD %']:.1f}%"
            )
            st.write(f"**Interpretation:** {interpret_stock(row)}")

            price_df = download_price_data(ticker, lookback_years)
            if price_df.empty or "Close" not in price_df.columns:
                st.warning(f"No price data for {ticker}")
                continue

            close = price_df["Close"].dropna()
            run_max = close.cummax()
            dd = (close / run_max - 1.0) * 100

            c1, c2 = st.columns(2)

            with c1:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=close.index, y=close.values,
                    mode="lines", name="Close"
                ))
                fig_price.add_trace(go.Scatter(
                    x=run_max.index, y=run_max.values,
                    mode="lines", name="Rolling Peak"
                ))
                fig_price.update_layout(
                    title=f"{ticker} Price vs Rolling Peak",
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig_price, use_container_width=True)

            with c2:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd.index, y=dd.values,
                    mode="lines", name="Drawdown %"
                ))
                fig_dd.update_layout(
                    title=f"{ticker} Drawdown from Previous Peak",
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)"
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            st.divider()

    # =========================================================
    # Full universe snapshot
    # =========================================================
    st.subheader("🌐 Full Universe Snapshot")
    full_display = df.copy()
    full_display["Interpretation"] = full_display.apply(interpret_stock, axis=1)
    st.dataframe(
        full_display.sort_values("Score", ascending=False).style.format({
            "Market Cap (B$)": "{:.1f}",
            "Revenue Growth YoY %": "{:.1f}",
            "ROIC Latest %": "{:.1f}",
            "ROIC Previous %": "{:.1f}",
            "MDD %": "{:.1f}",
            "Current DD %": "{:.1f}",
            "Score": "{:.1f}"
        }),
        use_container_width=True,
        height=500
    )

else:
    st.info("Set your filters on the left and click 'Run Screener'.")

    st.markdown("""
    ### Strategy logic
    This screener looks for:
    - high revenue growth
    - improving ROIC
    - meaningful drawdown from prior peak
    - large TAM sectors

    ### Practical use
    This is useful for finding:
    - growth stocks under correction
    - recovering capital efficiency
    - structurally strong industries
    """)

    st.markdown("""
    ### Important note
    ROIC is estimated from Yahoo Finance data and should be treated as an approximation.
    For final investment decisions, verify with company filings or a more reliable financial database.
    """)
