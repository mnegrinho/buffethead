# app.py (Código Alternativo Corrigido)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# Helpers
# ------------------------------

def _first_col(df: pd.DataFrame):
    """Returns the first column of a DataFrame."""
    if df is None or df.empty:
        return None
    return df.iloc[:, 0]


def _get_any(series_or_df, keys):
    """Try multiple label variants (Yahoo keys vary)."""
    if series_or_df is None:
        return None
    for k in keys:
        try:
            if isinstance(series_or_df, pd.Series):
                if k in series_or_df.index:
                    return series_or_df[k]
            else:
                if k in series_or_df:
                    return series_or_df[k]
        except Exception:
            pass
    return None


def safe_div(n, d):
    """Safely performs division, handling division by zero and NaNs."""
    try:
        if d is None or d == 0 or np.isnan(d):
            return np.nan
        return n / d
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_metrics(ticker: str):
    """Fetches key financial metrics for a given ticker."""
    t = yf.Ticker(ticker)

    try:
        fi = t.fast_info
        price = fi.get("last_price", np.nan)
        market_cap = fi.get("market_cap", np.nan)
    except Exception as e:
        price, market_cap = np.nan, np.nan
        st.error(f"Error fetching fast info for {ticker}: {e}")

    try:
        info = t.get_info()
    except Exception as e:
        info = {}
        st.error(f"Error fetching info for {ticker}: {e}")

    long_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector")

    try:
        inc = t.financials
        bal = t.balance_sheet
        cfs = t.cashflow
    except Exception as e:
        inc, bal, cfs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        st.error(f"Error fetching financial statements for {ticker}: {e}")

    inc_latest = _first_col(inc)
    bal_latest = _first_col(bal)
    cfs_latest = _first_col(cfs)

    # Revenue, gross profit, EBIT / operating income, net income
    revenue = _get_any(inc_latest, ["Total Revenue", "TotalRevenue", "Revenue"])
    gross_profit = _get_any(inc_latest, ["Gross Profit", "GrossProfit"])
    ebit = _get_any(inc_latest, ["Ebit", "EBIT", "Operating Income", "OperatingIncome"])
    net_income = _get_any(inc_latest, ["Net Income", "NetIncome"])

    # Debt & cash
    total_debt = _get_any(bal_latest, ["Total Debt", "TotalDebt"]) or ((_get_any(bal_latest, ["Long Term Debt", "LongTermDebt"]) or 0) + (_get_any(bal_latest, ["Short Long Term Debt", "ShortLongTermDebt", "Current Debt"]) or 0))
    cash = (_get_any(bal_latest, ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"]) or 0) + (_get_any(bal_latest, ["Short Term Investments", "ShortTermInvestments"]) or 0)

    # OCF & CapEx -> FCF
    ocf = _get_any(cfs_latest, ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"])
    capex = _get_any(cfs_latest, ["Capital Expenditures", "CapitalExpenditures"]) or 0
    fcf = ocf + (capex if not np.isnan(capex) else 0) if not pd.isna(ocf) else np.nan

    # EV, margins, yields
    ev = market_cap + (total_debt or 0) - (cash or 0)
    fcf_margin = safe_div(fcf, revenue)
    gross_margin = safe_div(gross_profit, revenue)
    net_margin = safe_div(net_income, revenue)
    ev_ebit = safe_div(ev, ebit) if ebit and ebit > 0 else np.nan
    fcf_yield_ev = safe_div(fcf, ev)

    # Revenue growth (CAGR)
    revenue_growth = np.nan
    try:
        annual_inc = t.financials.loc["Total Revenue"].dropna()
        if len(annual_inc) >= 5:
            # 4-year CAGR
            growth = (annual_inc.iloc[0] / annual_inc.iloc[4])**(1/4) - 1
            if not np.isinf(growth):
                revenue_growth = growth
    except Exception:
        pass

    if pd.isna(revenue_growth) or np.isinf(revenue_growth):
        try:
            revenue_growth = info.get("revenueGrowth")
        except Exception:
            pass

    # Shares outstanding trend (buybacks)
    buyback_1y = np.nan
    try:
        start = (datetime.utcnow() - timedelta(days=550)).date().isoformat()
        sh = t.get_shares_full(start=start)
        if sh is not None and not sh.empty:
            sh = sh.dropna()
            if len(sh) >= 2:
                current = sh.iloc[-1]
                year_ago = sh.iloc[0]
                buyback_1y = safe_div(year_ago - current, year_ago)
    except Exception:
        pass

    net_debt = (total_debt or 0) - (cash or 0)
    net_debt_fcf = safe_div(net_debt, fcf) if fcf > 0 else np.nan

    return {
        "ticker": ticker,
        "name": long_name,
        "sector": sector,
        "price": price,
        "market_cap": market_cap,
        "enterprise_value": ev,
        "revenue": revenue,
        "gross_profit": gross_profit,
        "ebit": ebit,
        "net_income": net_income,
        "ocf": ocf,
        "capex": capex,
        "fcf": fcf,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "fcf_margin": fcf_margin,
        "ev_ebit": ev_ebit,
        "fcf_yield_ev": fcf_yield_ev,
        "revenue_growth_yoy": revenue_growth,
        "buyback_1y": buyback_1y,
        "net_debt": net_debt,
        "net_debt_to_fcf": net_debt_fcf,
    }


def score_rows(df: pd.DataFrame):
    """Scores companies based on a weighted percentile ranking."""
    def pct_rank(col, invert=False):
        series = df[col].astype(float)
        series_clean = series.dropna()
        ranked = series_clean.rank(pct=True, method="average")
        if invert:
            ranked = 1 - ranked
        series.loc[series_clean.index] = ranked
        return series

    s = pd.DataFrame(index=df.index)
    s["score_fcf_yield"] = pct_rank("fcf_yield_ev", invert=False)
    s["score_ev_ebit"] = pct_rank("ev_ebit", invert=True)
    s["score_fcf_margin"] = pct_rank("fcf_margin", invert=False)
    s["score_growth"] = pct_rank("revenue_growth_yoy", invert=False)
    s["score_netdebt"] = pct_rank("net_debt_to_fcf", invert=True)
    s["score_buyback"] = pct_rank("buyback_1y", invert=False)

    weights = {
        "score_fcf_yield": 0.35,
        "score_ev_ebit": 0.20,
        "score_fcf_margin": 0.20,
        "score_growth": 0.10,
        "score_netdebt": 0.10,
        "score_buyback": 0.05,
    }

    df["quality_value_score"] = sum(s[c] * w for c, w in weights.items())
    return df


# ------------------------------
# App UI
# ------------------------------

st.set_page_config(page_title="Money‑Printer Screener", layout="wide")
st.title("💸 Money‑Printer Screener")
st.caption("Find cash‑generating leaders that might be undervalued — quick, transparent, and hackable.")

DEFAULT_TICKERS = [
    "MSFT","NVDA","AAPL","GOOGL","META","AMZN","AVGO","ORCL","ADBE","CRM",
    "AMD","INTC","TXN","ASML","SAP","NOW","NFLX","COST","V","MA",
    "LIN","NVO","UNH","KO","PEP","MCD","HD","DE","CAT","LMT",
    "XOM","CVX","SHEL","ABBV","PFE","JPM","BAC","AXP"
]

with st.sidebar:
    st.header("Universe")
    mode = st.radio("Choose tickers from:", ["Default list", "Paste tickers"], index=0)
    if mode == "Paste tickers":
        user_input = st.text_area(
            "Paste comma‑separated tickers (Yahoo symbols)",
            value=",".join(DEFAULT_TICKERS), height=120
        )
        tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TICKERS

    st.header("Filters")
    min_fcf_margin = st.slider("Min FCF margin", 0.0, 0.6, 0.10, 0.01)
    min_gross_margin = st.slider("Min gross margin", 0.0, 0.9, 0.40, 0.01)
    min_fcf_yield = st.slider("Min FCF yield (EV)", 0.00, 0.15, 0.03, 0.005)
    max_ev_ebit = st.slider("Max EV/EBIT", 2.0, 60.0, 25.0, 0.5)
    max_net_debt_fcf = st.slider("Max Net Debt / FCF", -5.0, 10.0, 3.0, 0.1)
    min_rev_growth = st.slider("Min revenue growth", -0.5, 0.5, 0.00, 0.01)

    st.markdown("—")
    run = st.button("Run Screener", use_container_width=True)

if run:
    rows = []
    progress_bar = st.progress(0, text="Downloading data...")
    num_workers = 8

    def worker(tk):
        try:
            return fetch_metrics(tk)
        except Exception as e:
            return {"ticker": tk, "error": str(e)}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_ticker = {executor.submit(worker, tk): tk for tk in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            rows.append(future.result())
            progress_bar.progress((i + 1) / len(tickers), text=f"Processed {i + 1}/{len(tickers)}")

    df = pd.DataFrame(rows)
    progress_bar.empty()

    # Basic cleaning
    numeric_cols = [
        "price","market_cap","enterprise_value","revenue","gross_profit","ebit","net_income",
        "ocf","capex","fcf","gross_margin","net_margin","fcf_margin","ev_ebit","fcf_yield_ev",
        "revenue_growth_yoy","buyback_1y","net_debt","net_debt_to_fcf",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply filters: "money‑printer" + undervaluation proxies
    mask = (
        (df["fcf_margin"] >= min_fcf_margin) &
        (df["gross_margin"] >= min_gross_margin) &
        (df["fcf_yield_ev"] >= min_fcf_yield) &
        (df["ev_ebit"] <= max_ev_ebit) &
        (df["net_debt_to_fcf"] <= max_net_debt_fcf) &
        (df["revenue_growth_yoy"] >= min_rev_growth)
    )

    filtered = df[mask].copy()
    if not filtered.empty:
        filtered = score_rows(filtered)
        filtered = filtered.sort_values("quality_value_score", ascending=False)

    st.subheader("Results")

    if filtered.empty:
        st.info("No companies matched your thresholds. Try loosening the filters.")
    else:
        show_cols = [
            "ticker","name","sector","price",
            "quality_value_score",
            "fcf_yield_ev","ev_ebit","fcf_margin","gross_margin","net_margin",
            "revenue_growth_yoy","net_debt_to_fcf","buyback_1y",
            "market_cap","enterprise_value","fcf","revenue","ebit",
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(
            filtered[show_cols].style.format({
                "price": "${:,.2f}",
                "quality_value_score": "{:.3f}",
                "fcf_yield_ev": "{:.2%}",
                "fcf_margin": "{:.2%}",
                "gross_margin": "{:.2%}",
                "net_margin": "{:.2%}",
                "revenue_growth_yoy": "{:.2%}",
                "net_debt_to_fcf": "{:.2f}",
                "buyback_1y": "{:.2%}",
                "market_cap": "${:,.0f}",
                "enterprise_value": "${:,.0f}",
                "fcf": "${:,.0f}",
                "revenue": "${:,.0f}",
                "ebit": "${:,.0f}",
            }),
            use_container_width=True,
        )
        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="money_printer_screener.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("How scoring works"):
        st.markdown(
            """
            **Composite score** = weighted percentiles across:

            - **FCF yield (EV)** — higher better *(35% weight)*
            - **EV/EBIT** — lower better *(20%)*
            - **FCF margin** — higher better *(20%)*
            - **Revenue growth YoY/CAGR** — higher better *(10%)*
            - **Net Debt / FCF** — lower better *(10%)*
            - **Buybacks (shares ↓ YoY)** — higher better *(5%)*

            Data source: Yahoo Finance via `yfinance`. Keys can be missing/quirky; this app best‑efforts around gaps.
            """
        )
else:
    st.info("Choose your universe and thresholds on the left, then click **Run Screener**.")
