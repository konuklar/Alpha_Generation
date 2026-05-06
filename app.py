# ============================================================
# QFA PRIME FINANCE PLATFORM — STREAMLIT v4 INTERACTIVE
# Commodity Instrument Class + Advanced KPI Layout
#
# Features preserved and expanded:
# - Yahoo Finance only, no synthetic fallback
# - Crude Oil, Gold, Silver, Platinum, Natural Gas
# - Strict equal-length daily data preparation
# - Benchmark alignment with ^GSPC
# - Portfolio strategies:
#   Equal Weight, User Custom Weights, Inverse Volatility,
#   Max Sharpe, Min Volatility
# - PyPortfolioOpt optimization
# - Advanced KPI layout
# - Performance metrics
# - Rolling Beta (Enhanced with Kalman Filter alternative)
# - Tracking Error
# - VaR / CVaR (Enhanced with Cornish-Fisher)
# - Drawdown
# - Correlation
# - Portfolio weights
# - QuantStats export
# - GARCH Volatility Lab
# - Info Hub + Data Quality
# - NEW: Advanced Risk & Performance Analytics Tab
# - NEW: Gain-Loss Ratio, Martin Ratio, Pain Index, Kappa 3, Stutzer Index
# - NEW: Cornish-Fisher VaR/CVaR, Modified Sharpe
# - NEW: Up/Down Capture Ratios, Appraisal Ratio
# - NEW: Kalman Filter Beta
# - NEW: Regime-Conditional Metrics
# - NEW: IC Decay Analysis
# - ENHANCED: Tearsheet includes all QuantStats ratios + advanced metrics
#
# Run:
# streamlit run app.py
# ============================================================

from __future__ import annotations

import os
import io
import sys
import html
import warnings
import subprocess
import importlib.util
import datetime as dt
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")


# ============================================================
# PACKAGE PREFLIGHT
# ============================================================

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "streamlit": "streamlit",
    "yfinance": "yfinance",
    "plotly": "plotly",
    "arch": "arch",
    "pypfopt": "PyPortfolioOpt",
    "quantstats": "quantstats",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}


def _pkg_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def ensure_required_packages(auto_install: bool = False) -> list[str]:
    missing = [pip for imp, pip in REQUIRED_PACKAGES.items() if not _pkg_available(imp)]
    if missing and auto_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
        except Exception as exc:
            print(f"[PACKAGE WARNING] Auto install failed: {exc}")
    return [pip for imp, pip in REQUIRED_PACKAGES.items() if not _pkg_available(imp)]


MISSING_PACKAGES = ensure_required_packages(auto_install=False)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy import stats as st_stats

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from arch import arch_model

try:
    import quantstats as qs
    QS_AVAILABLE = True
except Exception:
    QS_AVAILABLE = False

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="QFA Prime Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "Streamlit Interactive v4.13 QuantStats-Format Tearsheet + Advanced Risk Analytics"
TRADING_DAYS = 252
DEFAULT_RF = 0.045
MIN_START_DATE = dt.date(2018, 1, 1)
DEFAULT_START = MIN_START_DATE
DEFAULT_END = dt.date.today()
MIN_OBSERVATIONS = 100
DEFAULT_MIN_VALID_RATIO = 0.70
DEFAULT_FFILL_LIMIT = 3

BENCHMARK_TICKER = "^GSPC"
BENCHMARK_NAME = "S&P 500 Daily Index"

COMMODITY_UNIVERSE: Dict[str, Dict[str, str]] = {
    "CL=F": {"name": "Crude Oil WTI Futures", "class": "Energy", "display": "Crude Oil"},
    "GC=F": {"name": "Gold Futures", "class": "Precious Metals", "display": "Gold"},
    "SI=F": {"name": "Silver Futures", "class": "Precious Metals", "display": "Silver"},
    "PL=F": {"name": "Platinum Futures", "class": "Precious Metals", "display": "Platinum"},
    "NG=F": {"name": "Natural Gas Futures", "class": "Energy", "display": "Natural Gas"},
}

COMMODITY_UNIVERSE_FUTURES = COMMODITY_UNIVERSE.copy()

COMMODITY_UNIVERSE_ETF = {
    "USO": {"name": "United States Oil Fund LP", "class": "Energy ETF Proxy", "display": "Crude Oil ETF Proxy"},
    "GLD": {"name": "SPDR Gold Shares", "class": "Precious Metals ETF Proxy", "display": "Gold ETF Proxy"},
    "SLV": {"name": "iShares Silver Trust", "class": "Precious Metals ETF Proxy", "display": "Silver ETF Proxy"},
    "PPLT": {"name": "abrdn Physical Platinum Shares ETF", "class": "Precious Metals ETF Proxy", "display": "Platinum ETF Proxy"},
    "UNG": {"name": "United States Natural Gas Fund LP", "class": "Energy ETF Proxy", "display": "Natural Gas ETF Proxy"},
}

UNIVERSE_MODES = {
    "Yahoo ETF Proxies — more stable for Streamlit Cloud": COMMODITY_UNIVERSE_ETF,
    "Yahoo Futures — CL=F / GC=F / SI=F / PL=F / NG=F": COMMODITY_UNIVERSE_FUTURES,
}


FUTURES_TO_PROXY_MAP = {
    "CL=F": "USO",
    "GC=F": "GLD",
    "SI=F": "SLV",
    "PL=F": "PPLT",
    "NG=F": "UNG",
}

PROXY_TRANSPARENCY_TABLE = pd.DataFrame([
    {"Exposure": "Crude Oil", "Futures Ticker": "CL=F", "ETF Proxy": "USO", "Proxy Name": "United States Oil Fund LP"},
    {"Exposure": "Gold", "Futures Ticker": "GC=F", "ETF Proxy": "GLD", "Proxy Name": "SPDR Gold Shares"},
    {"Exposure": "Silver", "Futures Ticker": "SI=F", "ETF Proxy": "SLV", "Proxy Name": "iShares Silver Trust"},
    {"Exposure": "Platinum", "Futures Ticker": "PL=F", "ETF Proxy": "PPLT", "Proxy Name": "abrdn Physical Platinum Shares ETF"},
    {"Exposure": "Natural Gas", "Futures Ticker": "NG=F", "ETF Proxy": "UNG", "Proxy Name": "United States Natural Gas Fund LP"},
])


GARCH_MODEL_OPTIONS = {
    "Fast Institutional": ["garch_t", "gjr_t", "tarch_t"],
    "Full Institutional": ["garch_normal", "garch_t", "gjr_normal", "gjr_t", "tarch_t", "egarch_t"],
    "TARCH/ZARCH Only": ["tarch_t"],
    "GARCH Student's T Only": ["garch_t"],
}


# ============================================================
# INSTITUTIONAL COLOR SYSTEM
# ============================================================
# Muted, non-rainbow palette. Used consistently across all Plotly charts.

QFA_COLORS = {
    "navy": "#0f172a",
    "slate": "#334155",
    "steel": "#475569",
    "muted_blue": "#1e3a8a",
    "muted_gold": "#b08900",
    "charcoal": "#111827",
    "gray": "#6b7280",
    "light_gray": "#e5e7eb",
    "risk_red": "#991b1b",
    "amber": "#92400e",
    "green": "#166534",
}

QFA_SEQUENCE = [
    QFA_COLORS["navy"],
    QFA_COLORS["muted_gold"],
    QFA_COLORS["steel"],
    QFA_COLORS["gray"],
    QFA_COLORS["charcoal"],
    QFA_COLORS["muted_blue"],
]

STRATEGY_COLORS = {
    "Equal Weight": QFA_COLORS["navy"],
    "Inverse Volatility": QFA_COLORS["steel"],
    "User Custom Weights": QFA_COLORS["muted_gold"],
    "Max Sharpe": QFA_COLORS["charcoal"],
    "Min Volatility": QFA_COLORS["gray"],
    BENCHMARK_NAME: "#64748b",
}

def color_for_name(name: str, idx: int = 0) -> str:
    return STRATEGY_COLORS.get(name, QFA_SEQUENCE[idx % len(QFA_SEQUENCE)])



# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: "DejaVu Sans", "Inter", "Segoe UI", sans-serif !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 100% !important;
        }
        .qfa-hero {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
            padding: 28px 34px;
            border-radius: 24px;
            border-bottom: 5px solid #c9a227;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.22);
            margin-bottom: 18px;
        }
        .qfa-hero h1 {
            margin: 0;
            font-size: 36px;
            font-weight: 900;
            letter-spacing: .3px;
        }
        .qfa-hero p {
            margin: 7px 0 0;
            color: #cbd5e1;
            font-size: 15px;
        }
        .qfa-note {
            background: #fff7ed;
            border-left: 5px solid #f59e0b;
            padding: 14px 16px;
            border-radius: 10px;
            color: #78350f;
            line-height: 1.55;
            margin: 12px 0 18px;
        }
        .kpi-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 17px 18px;
            min-height: 116px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .kpi-label {
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 800;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 900;
            color: #0f172a;
            margin-top: 8px;
        }
        .kpi-sub {
            color: #64748b;
            font-size: 12px;
            margin-top: 6px;
        }
        .kpi-good { border-left: 6px solid #166534; }
        .kpi-warn { border-left: 6px solid #92400e; }
        .kpi-bad  { border-left: 6px solid #991b1b; }
        .kpi-neutral { border-left: 6px solid #334155; }
        .small-muted { color: #64748b; font-size: 12px; }
        .qfa-transparency-badge {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            color: #0f172a;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 700;
            margin: 8px 0 14px;
        }
        .qfa-kpi-band {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .qfa-kpi-band-title {
            font-size: 13px;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: .09em;
            font-weight: 900;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            padding: 10px 16px;
            font-weight: 800;
        }

        .compact-notes table {
            font-size: 12px !important;
            line-height: 1.25 !important;
        }
        .compact-notes th {
            font-size: 11px !important;
            padding: 6px !important;
        }
        .compact-notes td {
            font-size: 11px !important;
            padding: 6px !important;
            vertical-align: top !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILS
# ============================================================

def safe_chart(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displaylogo": False})


def safe_df(df: pd.DataFrame, hide_index: bool = True):
    st.dataframe(df, use_container_width=True, hide_index=hide_index)


def fmt_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}"


def kpi_card(label: str, value: str, sub: str = "", tone: str = "neutral"):
    st.markdown(
        f"""
        <div class="kpi-card kpi-{tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tone_for_return(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x > 0.10: return "good"
    if x > 0.00: return "warn"
    return "bad"


def tone_for_drawdown(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x > -0.15: return "good"
    if x > -0.35: return "warn"
    return "bad"


def tone_for_sharpe(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x > 1.0: return "good"
    if x > 0.3: return "warn"
    return "bad"


def tone_for_te(x: float) -> str:
    if pd.isna(x): return "neutral"
    if x < 0.08: return "good"
    if x < 0.18: return "warn"
    return "bad"


def clear_cache():
    try:
        st.cache_data.clear()
        st.success("Cache cleared.")
    except Exception as exc:
        st.warning(f"Cache clear failed: {exc}")


# ============================================================
# DATA ENGINE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _download_yahoo_batch(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=False,
        ignore_tz=True,
        repair=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _download_yahoo_single(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        ignore_tz=True,
        repair=True,
    )

    if raw is None or raw.empty:
        raw = yf.Ticker(ticker).history(
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            repair=True,
        )

    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: Yahoo returned no data.")

    if "Adj Close" in raw.columns:
        s = raw["Adj Close"].copy()
    elif "Close" in raw.columns:
        s = raw["Close"].copy()
    else:
        raise ValueError(f"{ticker}: Yahoo output has no Adj Close or Close.")

    s.name = ticker
    return pd.to_numeric(s, errors="coerce")


def _extract_adjclose_from_batch(raw: pd.DataFrame, tickers: Tuple[str, ...]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            px = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            px = raw["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in raw.columns:
            px = raw[["Adj Close"]].copy()
            px.columns = list(tickers)[:1]
        elif "Close" in raw.columns:
            px = raw[["Close"]].copy()
            px.columns = list(tickers)[:1]
        else:
            return pd.DataFrame()
    return px


def download_yahoo_prices(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """
    Robust Yahoo-only downloader:
    batch download -> individual ticker retry -> Ticker.history retry.
    No synthetic data is generated.
    """
    problems = []
    frames = []

    try:
        batch_raw = _download_yahoo_batch(tickers, start, end)
        batch_px = _extract_adjclose_from_batch(batch_raw, tickers)
    except Exception as exc:
        batch_px = pd.DataFrame()
        problems.append(f"Batch download failed: {exc}")

    if batch_px is not None and not batch_px.empty:
        batch_px.index = pd.to_datetime(batch_px.index)
        batch_px = batch_px.sort_index()
        for t in tickers:
            if t in batch_px.columns and batch_px[t].notna().sum() >= MIN_OBSERVATIONS:
                frames.append(batch_px[t].rename(t))

    done = {s.name for s in frames}

    for t in tickers:
        if t in done:
            continue
        try:
            time.sleep(0.35)
            frames.append(_download_yahoo_single(t, start, end))
        except Exception as exc:
            problems.append(f"{t}: {exc}")

    if not frames:
        raise ValueError(
            "Yahoo Finance returned no usable data for selected tickers. "
            "For commodity futures, this is commonly a Yahoo/yfinance timezone/throttling issue. "
            "Use sidebar Data Universe = 'Yahoo ETF Proxies — more stable for Streamlit Cloud', "
            "clear cache, or retry later. Details: " + " | ".join(problems[:8])
        )

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep="last")]
    px = px.sort_index()
    px = px.replace([np.inf, -np.inf], np.nan)

    for col in px.columns:
        px[col] = pd.to_numeric(px[col], errors="coerce")

    ordered = [t for t in tickers if t in px.columns]
    px = px[ordered] if ordered else px
    px = px.dropna(axis=1, how="all")

    if px.shape[1] == 0:
        raise ValueError("No usable Yahoo price series after robust retry engine.")

    return px


def prepare_price_matrix(
    prices: pd.DataFrame,
    min_valid_ratio: float,
    ffill_limit: int,
    allow_single: bool = False,
) -> pd.DataFrame:
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep="last")]
    px = px.sort_index()
    px = px.replace([np.inf, -np.inf], np.nan)

    valid_ratio = px.notna().mean()
    keep = valid_ratio[valid_ratio >= min_valid_ratio].index.tolist()
    px = px[keep]

    if px.shape[1] == 0:
        raise ValueError("No instruments passed minimum valid-data ratio.")

    px = px.ffill(limit=ffill_limit)
    px = px.dropna(axis=0, how="any")

    if px.shape[0] < MIN_OBSERVATIONS:
        raise ValueError(f"Aligned price matrix has fewer than {MIN_OBSERVATIONS} observations.")
    if (not allow_single) and px.shape[1] < 2:
        raise ValueError("At least two aligned instruments are required.")

    return px


def prepare_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if r.shape[0] < MIN_OBSERVATIONS:
        raise ValueError(f"Aligned return matrix has fewer than {MIN_OBSERVATIONS} observations.")
    return r


def align_two(a: pd.Series, b: pd.Series, name_a: str = "A", name_b: str = "B") -> Tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename(name_a), b.rename(name_b)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(f"{name_a} and {name_b} cannot be aligned with enough observations.")
    return df[name_a], df[name_b]


def data_quality_report(raw: pd.DataFrame, aligned: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in raw.columns:
        rows.append({
            "Ticker": col,
            "Display Name": COMMODITY_UNIVERSE.get(col, {}).get("display", col),
            "Raw Observations": int(raw[col].notna().sum()),
            "Raw Missing %": float(raw[col].isna().mean()),
            "Aligned Price Obs": int(aligned[col].notna().sum()) if col in aligned.columns else 0,
            "Aligned Return Obs": int(returns[col].notna().sum()) if col in returns.columns else 0,
            "First Date": aligned.index.min().strftime("%Y-%m-%d") if col in aligned.columns and not aligned.empty else "",
            "Last Date": aligned.index.max().strftime("%Y-%m-%d") if col in aligned.columns and not aligned.empty else "",
        })
    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def annualized_return(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1 if len(r) else np.nan


def annualized_volatility(r: pd.Series) -> float:
    return r.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(r: pd.Series, rf: float) -> float:
    vol = annualized_volatility(r)
    return np.nan if vol == 0 or pd.isna(vol) else (annualized_return(r) - rf) / vol


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.dropna()).cumprod()
    dd = eq / eq.cummax() - 1
    return dd.min()


def historical_var(r: pd.Series, level: float) -> float:
    return r.dropna().quantile(1 - level)


def historical_cvar(r: pd.Series, level: float) -> float:
    x = r.dropna()
    v = x.quantile(1 - level)
    return x[x <= v].mean()


def tracking_error(p: pd.Series, b: pd.Series) -> float:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return (p - b).std() * np.sqrt(TRADING_DAYS)


def beta_to_benchmark(p: pd.Series, b: pd.Series) -> float:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return p.cov(b) / b.var()


def information_ratio(p: pd.Series, b: pd.Series) -> float:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    te = tracking_error(p, b)
    return (annualized_return(p) - annualized_return(b)) / te if te and not pd.isna(te) else np.nan


def rolling_tracking_error(p: pd.Series, b: pd.Series, window: int) -> pd.Series:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return (p - b).rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_beta(p: pd.Series, b: pd.Series, window: int) -> pd.Series:
    p, b = align_two(p, b, "Portfolio", "Benchmark")
    return p.rolling(window).cov(b) / b.rolling(window).var()


def rolling_var_cvar(r: pd.Series, window: int) -> pd.DataFrame:
    x = r.dropna()
    out = pd.DataFrame(index=x.index)
    for level in [0.95, 0.99]:
        q = 1 - level
        out[f"VaR {int(level*100)}%"] = x.rolling(window).quantile(q)
        out[f"CVaR {int(level*100)}%"] = x.rolling(window).apply(
            lambda z: z[z <= np.quantile(z, q)].mean() if len(pd.Series(z).dropna()) >= 20 else np.nan,
            raw=True,
        )
    return out


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return lr.dropna(axis=0, how="any")


def return_difference_series(returns: pd.DataFrame, base_ticker: str) -> pd.DataFrame:
    if base_ticker not in returns.columns:
        base_ticker = returns.columns[0]
    diff = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col != base_ticker:
            diff[f"{col} minus {base_ticker}"] = returns[col] - returns[base_ticker]
    return diff.dropna(how="all")


def bollinger_frame(series: pd.Series, window: int = 63, n_std: float = 2.0) -> pd.DataFrame:
    s = series.dropna()
    out = pd.DataFrame(index=s.index)
    out["Value"] = s
    out["Rolling Mean"] = s.rolling(window).mean()
    out["Upper Band"] = out["Rolling Mean"] + n_std * s.rolling(window).std()
    out["Lower Band"] = out["Rolling Mean"] - n_std * s.rolling(window).std()
    out["Z-Score"] = (s - out["Rolling Mean"]) / s.rolling(window).std()
    return out


def rolling_sharpe(r: pd.Series, rf: float, window: int) -> pd.Series:
    daily_rf = rf / TRADING_DAYS
    excess = r - daily_rf
    return excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ============================================================
# ADVANCED PERFORMANCE RATIOS (QFA INSTITUTIONAL)
# ============================================================

def _clean_return_series(r: pd.Series) -> pd.Series:
    x = r.copy()
    x.index = pd.to_datetime(x.index)
    try:
        x.index = x.index.tz_localize(None)
    except Exception:
        pass
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    x = x[~x.index.duplicated(keep="last")]
    x = x.sort_index()
    return x


def gain_loss_ratio(r: pd.Series) -> float:
    """Bernardo-Ledoit Gain-Loss Ratio"""
    x = _clean_return_series(r)
    pos_mean = x[x > 0].mean()
    neg_mean = abs(x[x < 0].mean())
    return np.nan if pd.isna(neg_mean) or neg_mean == 0 else pos_mean / neg_mean


def martin_ratio(r: pd.Series, rf: float = 0.0) -> float:
    """Martin Ratio = CAGR / Ulcer Index"""
    x = _clean_return_series(r)
    eq = (1 + x).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    ui = np.sqrt(np.mean(dd ** 2)) if len(dd) else np.nan
    cagr_val = cagr(r)
    return np.nan if pd.isna(ui) or ui == 0 else (cagr_val - rf) / (ui / 100)


def pain_index(r: pd.Series) -> float:
    """Pain Index = sqrt(mean(drawdown²)) - alternatif Ulcer"""
    x = _clean_return_series(r)
    eq = (1 + x).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    return np.sqrt(np.mean(dd ** 2)) if len(dd) else np.nan


def pain_ratio(r: pd.Series, rf: float = 0.0) -> float:
    """Pain Ratio = (CAGR - rf) / Pain Index"""
    pi = pain_index(r)
    cagr_val = cagr(r)
    return np.nan if pd.isna(pi) or pi == 0 else (cagr_val - rf) / (pi / 100)


def kappa_3_ratio(r: pd.Series, threshold: float = 0.0) -> float:
    """Kappa 3 ratio - third order lower partial moment"""
    x = _clean_return_series(r)
    excess = (threshold - x).clip(lower=0)
    lpm3 = np.mean(excess ** 3) ** (1/3)
    return np.nan if lpm3 == 0 else (np.mean(x) - threshold) / lpm3


def stutzer_index(r: pd.Series, max_theta: float = 10.0, steps: int = 100) -> float:
    """
    Stutzer Performance Index (approximate via theta optimization)
    Higher values indicate better risk-adjusted performance with penalty for large losses.
    """
    x = _clean_return_series(r)
    if len(x) < 10:
        return np.nan
    
    def stutzer_objective(theta):
        val = np.mean(np.exp(-theta * x))
        if val <= 0:
            return -np.inf
        return np.log(val) / theta if theta != 0 else -np.mean(x)
    
    best_theta = 0.0
    best_value = -np.inf
    for theta in np.linspace(0.01, max_theta, steps):
        v = stutzer_objective(theta)
        if v > best_value:
            best_value = v
            best_theta = theta
    
    return -best_value if best_value != -np.inf else np.nan


def cornish_fisher_var(r: pd.Series, level: float = 0.95) -> float:
    """
    Cornish-Fisher VaR with skewness and kurtosis adjustment.
    More accurate for non-normal returns.
    """
    x = _clean_return_series(r)
    if len(x) < 10:
        return historical_var(r, level)
    
    z = abs(st_stats.norm.ppf(1 - level))
    s = x.skew()
    k = x.kurtosis()
    
    # Cornish-Fisher expansion
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36
    
    mu = x.mean()
    sigma = x.std()
    # Return as loss (positive number for VaR magnitude)
    return abs(mu + sigma * z_cf)


def cornish_fisher_cvar(r: pd.Series, level: float = 0.95) -> float:
    """
    Cornish-Fisher CVaR approximation.
    """
    x = _clean_return_series(r)
    if len(x) < 10:
        return abs(historical_cvar(r, level))
    
    var_cf = cornish_fisher_var(x, level)
    # Simple approximation: CVaR ≈ VaR * (1 + skew adjustment)
    s = x.skew()
    adjustment = 1 + (s * 0.2)  # heuristic
    return var_cf * adjustment if adjustment > 0 else var_cf


def modified_sharpe_cf(r: pd.Series, rf: float = 0.0, level: float = 0.95) -> float:
    """Modified Sharpe using Cornish-Fisher VaR as risk measure"""
    cvar_cf = cornish_fisher_cvar(r, level)
    if pd.isna(cvar_cf) or cvar_cf == 0:
        return np.nan
    return (annualized_return(r) - rf) / (cvar_cf * np.sqrt(TRADING_DAYS))


def garch_conditional_var(
    r: pd.Series, 
    garch_store: Dict[str, Dict[str, Any]], 
    ticker: str, 
    model_key: str = "garch_t",
    level: float = 0.95
) -> float:
    """
    GARCH-based conditional VaR using the fitted volatility forecast.
    """
    key = f"{ticker}__{model_key}"
    if key not in garch_store:
        return np.nan
    
    g = garch_store[key]
    cond_vol = g["Conditional Vol"].iloc[-1] / 100  # convert from % to decimal
    
    # Use student-t quantile if available, else normal
    if "StudentsT" in g.get("Model", ""):
        from scipy.stats import t as t_dist
        # Approximate degrees of freedom from GARCH (10 is typical)
        df = 10
        z = t_dist.ppf(1 - level, df)
    else:
        z = st_stats.norm.ppf(1 - level)
    
    # Expected return (small, can be zero)
    mu = r.mean()
    return abs(mu + cond_vol * z)


def capture_ratios(portfolio: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    """
    Up/Down Capture Ratios relative to benchmark.
    """
    p, b = align_two(portfolio, benchmark, "Portfolio", "Benchmark")
    
    up_mask = b > 0
    down_mask = b < 0
    
    up_capture = p[up_mask].sum() / b[up_mask].sum() if b[up_mask].sum() != 0 else np.nan
    down_capture = p[down_mask].sum() / b[down_mask].sum() if b[down_mask].sum() != 0 else np.nan
    capture_ratio = up_capture / down_capture if down_capture != 0 else np.nan
    
    return {
        "Up Capture Ratio": up_capture,
        "Down Capture Ratio": down_capture,
        "Capture Ratio (Up/Down)": capture_ratio,
    }


def appraisal_ratio(portfolio: pd.Series, benchmark: pd.Series, rf: float = 0.0) -> float:
    """
    Appraisal Ratio (Treynor-Black style): Alpha / Residual Volatility.
    Measures active return per unit of active risk.
    """
    p, b = align_two(portfolio, benchmark, "Portfolio", "Benchmark")
    
    # Alpha from regression (daily)
    beta = p.cov(b) / b.var()
    alpha_daily = p.mean() - b.mean() * beta - rf / TRADING_DAYS
    
    # Residual volatility (daily)
    residual = p - beta * b - alpha_daily
    residual_vol_daily = residual.std()
    residual_vol_annual = residual_vol_daily * np.sqrt(TRADING_DAYS)
    
    alpha_annual = alpha_daily * TRADING_DAYS
    return alpha_annual / residual_vol_annual if residual_vol_annual != 0 else np.nan


def information_coefficient_decay(
    signals: Dict[str, pd.DataFrame], 
    returns: pd.DataFrame, 
    max_lag: int = 10
) -> pd.DataFrame:
    """
    IC Decay analysis: how IC changes with lag (1 to max_lag days).
    """
    results = []
    for name, sig in signals.items():
        for lag in range(1, max_lag + 1):
            shifted_returns = returns.shift(-lag)
            common_idx = sig.index.intersection(shifted_returns.index)
            if len(common_idx) > 20:
                # Manual IC calculation for this lag
                ic_values = []
                for dt in common_idx:
                    s = sig.loc[dt]
                    f = shifted_returns.loc[dt]
                    df_ic = pd.concat([s.rename("signal"), f.rename("future")], axis=1).dropna()
                    if len(df_ic) >= 3:
                        ic_values.append(df_ic["signal"].corr(df_ic["future"], method="spearman"))
                if ic_values:
                    results.append({
                        "Signal": name,
                        "Lag (Days)": lag,
                        "Mean IC": np.mean(ic_values),
                    })
    return pd.DataFrame(results)


def regime_conditional_metrics(
    portfolio: pd.Series, 
    benchmark: pd.Series, 
    volatility_thresholds: Tuple[float, float] = (0.10, 0.20)
) -> pd.DataFrame:
    """
    Compute Sharpe and other metrics conditional on volatility regime.
    """
    p, b = align_two(portfolio, benchmark, "Portfolio", "Benchmark")
    
    # Rolling volatility (21-day)
    roll_vol = p.rolling(21).std() * np.sqrt(TRADING_DAYS)
    
    low_vol_mask = roll_vol < volatility_thresholds[0]
    mid_vol_mask = (roll_vol >= volatility_thresholds[0]) & (roll_vol < volatility_thresholds[1])
    high_vol_mask = roll_vol >= volatility_thresholds[1]
    
    regimes = {
        "Low Volatility": low_vol_mask,
        "Moderate Volatility": mid_vol_mask,
        "High Volatility": high_vol_mask,
    }
    
    rows = []
    for regime_name, mask in regimes.items():
        p_regime = p[mask]
        if len(p_regime) < 20:
            rows.append({"Regime": regime_name, "Sharpe": np.nan, "Volatility": np.nan, "Return": np.nan, "Observations": 0})
        else:
            rows.append({
                "Regime": regime_name,
                "Sharpe": sharpe_ratio(p_regime, DEFAULT_RF),
                "Volatility": annualized_volatility(p_regime),
                "Return": annualized_return(p_regime),
                "Observations": len(p_regime),
            })
    
    return pd.DataFrame(rows)


# ============================================================
# KALMAN FILTER BETA (ALTERNATIVE TO ROLLING BETA)
# ============================================================

def kalman_beta(portfolio: pd.Series, benchmark: pd.Series, delta: float = 1e-5) -> pd.Series:
    """
    Time-varying beta using Kalman filter (state-space model).
    Returns smoother and more adaptive beta than rolling window.
    """
    p, b = align_two(portfolio, benchmark, "Portfolio", "Benchmark")
    
    n = len(p)
    beta_hat = np.zeros(n)
    P = np.ones(n)
    
    # Initial state
    beta_hat[0] = p.iloc[0] / b.iloc[0] if b.iloc[0] != 0 else 1.0
    P[0] = 1.0
    
    for t in range(1, n):
        # Prediction
        beta_pred = beta_hat[t-1]
        P_pred = P[t-1] + delta
        
        # Update
        y = p.iloc[t]
        x = b.iloc[t]
        
        if x != 0:
            K = P_pred * x / (x * P_pred * x + 1e-6)
            beta_hat[t] = beta_pred + K * (y - x * beta_pred)
            P[t] = (1 - K * x) * P_pred
        else:
            beta_hat[t] = beta_pred
            P[t] = P_pred
    
    return pd.Series(beta_hat, index=p.index, name="Kalman Beta")


def rolling_beta_enhanced(portfolio: pd.Series, benchmark: pd.Series, window: int = 63) -> Dict[str, pd.Series]:
    """
    Return both traditional rolling beta and Kalman filter beta for comparison.
    """
    rolling = rolling_beta(portfolio, benchmark, window)
    kalman = kalman_beta(portfolio, benchmark)
    return {
        "Rolling Beta": rolling,
        "Kalman Beta": kalman,
    }


def metrics_for_returns(return_map: Dict[str, pd.Series], benchmark: pd.Series, rf: float) -> pd.DataFrame:
    rows = []
    for name, r in return_map.items():
        p, b = align_two(r, benchmark, name, "Benchmark")
        rows.append({
            "Strategy / Instrument": name,
            "Annual Return": annualized_return(p),
            "Annual Volatility": annualized_volatility(p),
            "Sharpe": sharpe_ratio(p, rf),
            "Max Drawdown": max_drawdown(p),
            "VaR 95% Daily": historical_var(p, 0.95),
            "CVaR 95% Daily": historical_cvar(p, 0.95),
            "VaR 99% Daily": historical_var(p, 0.99),
            "CVaR 99% Daily": historical_cvar(p, 0.99),
            "Tracking Error vs Benchmark": tracking_error(p, b),
            "Beta vs Benchmark": beta_to_benchmark(p, b),
            "Information Ratio": information_ratio(p, b),
            "Skew": p.skew(),
            "Kurtosis": p.kurtosis(),
            # NEW ADVANCED RATIOS
            "Gain-Loss Ratio": gain_loss_ratio(p),
            "Martin Ratio": martin_ratio(p, rf),
            "Pain Index": pain_index(p),
            "Pain Ratio": pain_ratio(p, rf),
            "Kappa 3": kappa_3_ratio(p, 0),
            "Stutzer Index": stutzer_index(p),
            "Cornish-Fisher VaR 95%": cornish_fisher_var(p, 0.95),
            "Cornish-Fisher CVaR 95%": cornish_fisher_cvar(p, 0.95),
            "Modified Sharpe (CF-VaR)": modified_sharpe_cf(p, rf, 0.95),
            "Up Capture Ratio": capture_ratios(p, b).get("Up Capture Ratio", np.nan),
            "Down Capture Ratio": capture_ratios(p, b).get("Down Capture Ratio", np.nan),
            "Appraisal Ratio": appraisal_ratio(p, b, rf),
        })
    return pd.DataFrame(rows)


def interpret_strategy_table(metrics: pd.DataFrame) -> pd.DataFrame:
    def beta_text(x):
        if pd.isna(x): return "N/A"
        if x > 1.2: return "High equity sensitivity"
        if x > 0.8: return "Market-like beta"
        if x > 0.3: return "Moderate equity sensitivity"
        if x > -0.3: return "Low equity sensitivity"
        return "Negative beta"

    def te_text(x):
        if pd.isna(x): return "N/A"
        if x < 0.05: return "Low active risk"
        if x < 0.10: return "Moderate active risk"
        if x < 0.20: return "High active risk"
        return "Very high active risk"

    def dd_text(x):
        if pd.isna(x): return "N/A"
        if x > -0.10: return "Contained drawdown"
        if x > -0.25: return "Moderate drawdown"
        if x > -0.40: return "High drawdown"
        return "Severe drawdown"

    rows = []
    for _, r in metrics.iterrows():
        rows.append({
            "Strategy": r["Strategy / Instrument"],
            "Plain-English Read": (
                f"{dd_text(r['Max Drawdown'])}; {te_text(r['Tracking Error vs Benchmark'])}; "
                f"{beta_text(r['Beta vs Benchmark'])}; Sharpe {r['Sharpe']:.2f}."
            ),
            "Risk Snapshot": (
                f"VaR95 {r['VaR 95% Daily']:.2%}, CVaR95 {r['CVaR 95% Daily']:.2%}, "
                f"Vol {r['Annual Volatility']:.2%}."
            ),
        })
    return pd.DataFrame(rows)


def metric_dictionary() -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Annual Return", "Meaning": "Compounded yearly return.", "How to read": "Higher is better only if risk is acceptable."},
        {"Metric": "Annual Volatility", "Meaning": "Annualized fluctuation.", "How to read": "Higher means rougher return path."},
        {"Metric": "Sharpe", "Meaning": "Return per unit of risk after RF.", "How to read": "Above 1 strong; below 0 weak."},
        {"Metric": "Max Drawdown", "Meaning": "Worst peak-to-trough loss.", "How to read": "Capital pain measure."},
        {"Metric": "VaR", "Meaning": "Loss threshold at a confidence level.", "How to read": "VaR 95% is exceeded in worst 5% of days."},
        {"Metric": "CVaR", "Meaning": "Average loss beyond VaR.", "How to read": "More conservative tail-loss measure."},
        {"Metric": "Tracking Error", "Meaning": "Annualized active risk vs benchmark.", "How to read": "Low means benchmark-like; high means active/different."},
        {"Metric": "Rolling Beta", "Meaning": "Time-varying benchmark sensitivity.", "How to read": "1 is equity-like; 0 is low market sensitivity."},
        {"Metric": "Information Ratio", "Meaning": "Active return per unit TE.", "How to read": "Higher is better versus benchmark."},
        {"Metric": "Gain-Loss Ratio", "Meaning": "Bernardo-Ledoit ratio of avg gain to avg loss.", "How to read": ">1.5 indicates favorable asymmetry."},
        {"Metric": "Martin Ratio", "Meaning": "CAGR / Ulcer Index (drawdown penalty).", "How to read": "Higher is better; penalizes drawdown severity."},
        {"Metric": "Pain Index", "Meaning": "sqrt(mean(drawdown²)) - drawdown severity.", "How to read": "Lower is better for capital preservation."},
        {"Metric": "Kappa 3", "Meaning": "Third-order lower partial moment ratio.", "How to read": "Higher is better; focuses on large losses."},
        {"Metric": "Stutzer Index", "Meaning": "Performance with penalty for catastrophic losses.", "How to read": "Higher = better large-loss avoidance."},
        {"Metric": "Cornish-Fisher VaR", "Meaning": "VaR with skewness/kurtosis adjustment.", "How to read": "Compare with historical VaR; difference indicates non-normality."},
        {"Metric": "Capture Ratios", "Meaning": "Up/Down market participation relative to benchmark.", "How to read": "Up > 100% and Down < 100% is ideal."},
        {"Metric": "Appraisal Ratio", "Meaning": "Alpha per unit of residual risk.", "How to read": "Higher indicates better active management efficiency."},
    ])


def portfolio_strategy_notes_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Strategy": "Equal Weight",
            "Core Idea": "Neutral reference portfolio. Every selected instrument receives identical capital weight.",
            "Weight Logic": "1/N allocation. No return forecast, no volatility forecast, no optimization.",
            "Why It Matters": "This is the governance benchmark. If a complex model cannot beat it after risk and drawdown, the model is not justified.",
            "Strength": "Transparent, robust, low model risk.",
            "Weakness": "Ignores volatility, correlation and regime change.",
            "User Question Answered": "What happens if we avoid all model assumptions?",
            "Governance Note": "Use as baseline for alpha, risk and drawdown comparison.",
        },
        {
            "Strategy": "User Custom Weights",
            "Core Idea": "Investment committee or portfolio manager view translated into allocation.",
            "Weight Logic": "Sidebar inputs are normalized to 100%.",
            "Why It Matters": "Allows policy-driven or discretionary exposure to be tested against systematic alternatives.",
            "Strength": "Reflects real decision maker views and constraints.",
            "Weakness": "Can hide concentration risk if not governed by caps and risk budgets.",
            "User Question Answered": "What is the risk/return profile of our preferred allocation?",
            "Governance Note": "Always compare against Equal Weight and Min Volatility.",
        },
        {
            "Strategy": "Inverse Volatility",
            "Core Idea": "Defensive risk-aware portfolio; gives more weight to lower-volatility instruments.",
            "Weight Logic": "Weight is proportional to 1 / historical volatility.",
            "Why It Matters": "A practical risk-budget proxy when expected returns are uncertain.",
            "Strength": "Reduces dominance of high-volatility instruments.",
            "Weakness": "May overweight low-volatility assets with poor expected return.",
            "User Question Answered": "Can we reduce volatility without complex optimization?",
            "Governance Note": "Useful during unstable regimes and as a conservative benchmark.",
        },
        {
            "Strategy": "Max Sharpe",
            "Core Idea": "Optimization candidate targeting best expected excess return per unit of volatility.",
            "Weight Logic": "PyPortfolioOpt uses expected returns and Ledoit-Wolf covariance under max-weight cap.",
            "Why It Matters": "Converts return and risk forecasts into a disciplined portfolio candidate.",
            "Strength": "Explicitly targets risk-adjusted performance.",
            "Weakness": "Sensitive to expected return estimation error.",
            "User Question Answered": "Which allocation is most efficient under our assumptions?",
            "Governance Note": "Never approve blindly; validate with TE, drawdown, VaR/CVaR and robustness checks.",
        },
        {
            "Strategy": "Min Volatility",
            "Core Idea": "Defensive optimized portfolio designed to minimize expected volatility.",
            "Weight Logic": "PyPortfolioOpt minimizes covariance-based portfolio variance under constraints.",
            "Why It Matters": "Provides a capital-preservation candidate.",
            "Strength": "Usually has lower volatility and drawdown.",
            "Weakness": "May sacrifice upside and concentrate in low-vol assets.",
            "User Question Answered": "What is the lowest-risk allocation in this universe?",
            "Governance Note": "Compare against Inverse Volatility to detect optimizer instability.",
        },
    ])


def alpha_generation_methods_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Alpha Pillar": "Momentum / Trend",
            "Quant Technique": "Time-series momentum, cross-sectional momentum, moving-average breakouts, 3/6/12-month lookback signals.",
            "Signal Definition": "Rank instruments by past return or trend strength; favor persistent winners and penalize losers.",
            "Validation Metric": "Information Coefficient, hit ratio, turnover, post-cost alpha.",
            "Risk Control": "Volatility scaling, drawdown stop, sector/asset cap, TE budget.",
            "Board-Level Explanation": "We test whether price persistence exists and whether it survives transaction costs and risk controls.",
        },
        {
            "Alpha Pillar": "Mean Reversion / Relative Value",
            "Quant Technique": "Z-score spreads, Bollinger bands on log-return differences, cointegration-style spread monitoring.",
            "Signal Definition": "When relative return spread is unusually stretched, expect partial normalization.",
            "Validation Metric": "Spread half-life, z-score reversal rate, profit factor, drawdown during trend regimes.",
            "Risk Control": "Stop-loss by z-score expansion, regime filter, volatility cap.",
            "Board-Level Explanation": "We exploit temporary dislocations, but only when the spread historically reverts.",
        },
        {
            "Alpha Pillar": "Carry / Roll Yield Proxy",
            "Quant Technique": "Futures curve proxy, ETF roll-cost diagnostics, commodity carry indicators where data is available.",
            "Signal Definition": "Prefer assets with favorable carry/roll profile; avoid persistent negative roll drag.",
            "Validation Metric": "Carry contribution, return decomposition, rolling alpha after volatility adjustment.",
            "Risk Control": "Commodity-specific caps and liquidity screen.",
            "Board-Level Explanation": "Commodity returns are not only spot moves; roll/carry can dominate long-run ETF outcomes.",
        },
        {
            "Alpha Pillar": "Volatility Risk Premia",
            "Quant Technique": "GARCH volatility forecasts, realized-volatility regimes, volatility breakout filters.",
            "Signal Definition": "Reduce exposure when forecast volatility rises; increase risk only when compensated.",
            "Validation Metric": "Vol forecast error, drawdown reduction, Sharpe improvement, VaR breach frequency.",
            "Risk Control": "GARCH forecast volatility cap, CVaR budget, dynamic de-risking.",
            "Board-Level Explanation": "Alpha is protected by avoiding uncompensated volatility spikes.",
        },
        {
            "Alpha Pillar": "Factor / Macro Exposure",
            "Quant Technique": "Regression beta to USD, rates, equities, inflation proxies, PCA factor extraction.",
            "Signal Definition": "Identify whether returns are driven by macro regimes rather than asset-specific alpha.",
            "Validation Metric": "Alpha after factor adjustment, residual IC, factor attribution.",
            "Risk Control": "Factor exposure limits and residual-risk monitoring.",
            "Board-Level Explanation": "We separate true manager skill from hidden macro bets.",
        },
        {
            "Alpha Pillar": "Black-Litterman Views",
            "Quant Technique": "Bayesian blending of equilibrium returns with explicit active views and confidence levels.",
            "Signal Definition": "Convert investment views into posterior expected returns and optimized weights.",
            "Validation Metric": "View hit rate, posterior vs prior attribution, active risk efficiency.",
            "Risk Control": "Confidence slider, max weight, TE budget, stress testing.",
            "Board-Level Explanation": "Views are not hard-coded; confidence controls how strongly they affect the portfolio.",
        },
        {
            "Alpha Pillar": "Machine Learning Ranking",
            "Quant Technique": "Regularized regression, tree models, walk-forward validation, feature importance and SHAP-style diagnostics.",
            "Signal Definition": "Predict next-period relative return or risk-adjusted score from lagged features.",
            "Validation Metric": "Out-of-sample IC, rank IC, turnover, degradation analysis.",
            "Risk Control": "Purged walk-forward testing, feature stability, model governance.",
            "Board-Level Explanation": "ML is only accepted if it improves out-of-sample ranking after costs and risk limits.",
        },
    ])


def alpha_research_pipeline_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Step": "1. Hypothesis", "Question": "Why should this signal work?", "Output": "Documented economic rationale."},
        {"Step": "2. Signal Construction", "Question": "How exactly is alpha measured?", "Output": "Formula, lookback, rebalance rule."},
        {"Step": "3. Validation", "Question": "Does it predict future returns?", "Output": "IC, rank IC, hit ratio, t-stat, decay curve."},
        {"Step": "4. Cost & Turnover", "Question": "Does alpha survive implementation?", "Output": "Net alpha after transaction and slippage assumptions."},
        {"Step": "5. Risk Model", "Question": "Is it alpha or hidden beta?", "Output": "Factor-adjusted alpha and residual risk."},
        {"Step": "6. Portfolio Construction", "Question": "How does signal become weights?", "Output": "Constraints, TE budget, max weight, CVaR budget."},
        {"Step": "7. Monitoring", "Question": "When do we stop trusting it?", "Output": "Live IC monitoring, drawdown triggers, regime diagnostics."},
    ])


# ============================================================
# STRATEGIES
# ============================================================

def equal_weight(returns: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    w = pd.Series(1 / returns.shape[1], index=returns.columns)
    return returns.dot(w).rename("Equal Weight"), w.to_dict()


def inverse_vol(returns: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    vol = returns.std().replace(0, np.nan)
    inv = 1 / vol
    w = (inv / inv.sum()).fillna(0)
    return returns.dot(w).rename("Inverse Volatility"), w.to_dict()


def custom_weight_strategy(returns: pd.DataFrame, weights: Dict[str, float]) -> Tuple[pd.Series, Dict[str, float]]:
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    if w.sum() <= 0:
        w = pd.Series(1 / returns.shape[1], index=returns.columns)
    else:
        w = w / w.sum()
    return returns.dot(w).rename("User Custom Weights"), w.to_dict()


def optimize_strategies(prices: pd.DataFrame, returns: pd.DataFrame, rf: float, max_weight: float) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, float]], pd.DataFrame]:
    strategy_returns = {}
    weights = {}
    perf_rows = []

    if not PYPFOPT_AVAILABLE:
        return strategy_returns, weights, pd.DataFrame([{"Strategy": "PyPortfolioOpt", "Error": "PyPortfolioOpt package unavailable"}])

    if not _pkg_available("sklearn"):
        return strategy_returns, weights, pd.DataFrame([{
            "Strategy": "PyPortfolioOpt",
            "Error": "scikit-learn is missing. Add scikit-learn to requirements.txt for Ledoit-Wolf covariance."
        }])

    try:
        mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
        S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    except Exception as exc:
        return strategy_returns, weights, pd.DataFrame([{
            "Strategy": "PyPortfolioOpt",
            "Error": f"Covariance/expected-return preparation failed: {exc}"
        }])

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.max_sharpe(risk_free_rate=rf)
        clean = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=rf, verbose=False)
        w = pd.Series(clean).reindex(returns.columns).fillna(0)
        strategy_returns["Max Sharpe"] = returns.dot(w).rename("Max Sharpe")
        weights["Max Sharpe"] = w.to_dict()
        perf_rows.append({"Strategy": "Max Sharpe", "Expected Return": perf[0], "Expected Volatility": perf[1], "Expected Sharpe": perf[2]})
    except Exception as exc:
        perf_rows.append({"Strategy": "Max Sharpe", "Error": str(exc)})

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.min_volatility()
        clean = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=rf, verbose=False)
        w = pd.Series(clean).reindex(returns.columns).fillna(0)
        strategy_returns["Min Volatility"] = returns.dot(w).rename("Min Volatility")
        weights["Min Volatility"] = w.to_dict()
        perf_rows.append({"Strategy": "Min Volatility", "Expected Return": perf[0], "Expected Volatility": perf[1], "Expected Sharpe": perf[2]})
    except Exception as exc:
        perf_rows.append({"Strategy": "Min Volatility", "Error": str(exc)})

    return strategy_returns, weights, pd.DataFrame(perf_rows)


def build_strategy_set(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf: float,
    max_weight: float,
    custom_weights: Dict[str, float],
) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, float]], pd.DataFrame, pd.DataFrame]:
    strategy_returns = {}
    strategy_weights = {}

    r, w = equal_weight(returns)
    strategy_returns["Equal Weight"] = r
    strategy_weights["Equal Weight"] = w

    r, w = inverse_vol(returns)
    strategy_returns["Inverse Volatility"] = r
    strategy_weights["Inverse Volatility"] = w

    r, w = custom_weight_strategy(returns, custom_weights)
    strategy_returns["User Custom Weights"] = r
    strategy_weights["User Custom Weights"] = w

    opt_r, opt_w, opt_perf = optimize_strategies(prices, returns, rf, max_weight)
    strategy_returns.update(opt_r)
    strategy_weights.update(opt_w)

    wdf = pd.DataFrame({"Ticker": returns.columns})
    wdf["Display Name"] = [COMMODITY_UNIVERSE.get(t, {}).get("display", t) for t in returns.columns]
    for name, w in strategy_weights.items():
        wdf[name] = pd.Series(w).reindex(returns.columns).fillna(0).values

    return strategy_returns, strategy_weights, wdf, opt_perf



# ============================================================
# ALPHA ENGINE — LIVE SIGNALS, IC, ALPHA WEIGHTS
# ============================================================

def zscore_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu = df.rolling(window).mean()
    sig = df.rolling(window).std().replace(0, np.nan)
    return (df - mu) / sig


def cross_sectional_rank_score(df: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """
    Converts each date's cross-section into [-1, +1] rank score.
    """
    ranks = df.rank(axis=1, pct=True, ascending=not higher_is_better)
    return (ranks - 0.5) * 2.0


def build_alpha_signals(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    momentum_short: int = 63,
    momentum_long: int = 126,
    meanrev_window: int = 63,
    vol_window: int = 63,
) -> Dict[str, pd.DataFrame]:
    """
    Produces transparent alpha signal families:
    - Momentum: cross-sectional rank of trailing returns
    - Mean Reversion: negative z-score of price deviation
    - Volatility Quality: lower realized vol receives higher score
    - Composite: average of available normalized scores
    """
    px = prices.reindex(returns.index).dropna(how="any")
    r = returns.reindex(px.index).dropna(how="any")

    mom_short_raw = px.pct_change(momentum_short)
    mom_long_raw = px.pct_change(momentum_long)
    momentum_score = 0.5 * cross_sectional_rank_score(mom_short_raw, True) + 0.5 * cross_sectional_rank_score(mom_long_raw, True)

    price_z = zscore_frame(np.log(px), meanrev_window)
    meanrev_score = cross_sectional_rank_score(-price_z, True)

    realized_vol = r.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    vol_quality_score = cross_sectional_rank_score(-realized_vol, True)

    composite_score = (
        momentum_score.reindex(r.index).fillna(0.0)
        + meanrev_score.reindex(r.index).fillna(0.0)
        + vol_quality_score.reindex(r.index).fillna(0.0)
    ) / 3.0

    return {
        "Momentum": momentum_score.dropna(how="all"),
        "Mean Reversion": meanrev_score.dropna(how="all"),
        "Volatility Quality": vol_quality_score.dropna(how="all"),
        "Composite Alpha": composite_score.dropna(how="all"),
    }


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Approximate forward compounded returns from daily returns.
    """
    return (1 + returns).rolling(horizon).apply(lambda x: np.prod(x) - 1, raw=False).shift(-horizon)


def information_coefficient(signal: pd.DataFrame, returns: pd.DataFrame, horizon: int = 21, method: str = "spearman") -> pd.Series:
    """
    Cross-sectional IC: date-by-date correlation between today's signal and future returns.
    """
    fwd = forward_returns(returns.reindex(signal.index), horizon)
    common_idx = signal.index.intersection(fwd.index)
    out = {}
    for dt_idx in common_idx:
        s = signal.loc[dt_idx]
        f = fwd.loc[dt_idx]
        df = pd.concat([s.rename("signal"), f.rename("future")], axis=1).dropna()
        if len(df) >= 3:
            out[dt_idx] = df["signal"].corr(df["future"], method=method)
    return pd.Series(out, name=f"{horizon}D {method.title()} IC").dropna()


def ic_summary_table(signals: Dict[str, pd.DataFrame], returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for name, sig in signals.items():
        ic = information_coefficient(sig, returns, horizon=horizon, method="spearman")
        if len(ic) == 0:
            rows.append({"Signal": name, "Mean IC": np.nan, "IC Vol": np.nan, "ICIR": np.nan, "Hit Ratio": np.nan, "Observations": 0})
        else:
            rows.append({
                "Signal": name,
                "Mean IC": ic.mean(),
                "IC Vol": ic.std(),
                "ICIR": ic.mean() / ic.std() * np.sqrt(12) if ic.std() and not pd.isna(ic.std()) else np.nan,
                "Hit Ratio": (ic > 0).mean(),
                "Observations": len(ic),
            })
    return pd.DataFrame(rows)


def alpha_weights_from_signal(
    signal: pd.DataFrame,
    max_weight: float = 0.45,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Converts alpha scores into portfolio weights.
    Long-only mode clips negative scores to zero.
    """
    sig = signal.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if long_only:
        raw = sig.clip(lower=0.0)
        weights = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
        eq = pd.DataFrame(1.0 / sig.shape[1], index=sig.index, columns=sig.columns)
        weights = weights.fillna(eq)
    else:
        demeaned = sig.sub(sig.mean(axis=1), axis=0)
        denom = demeaned.abs().sum(axis=1).replace(0, np.nan)
        weights = demeaned.div(denom, axis=0).fillna(0.0)

    if long_only:
        weights = weights.clip(upper=max_weight)
        weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / sig.shape[1])
    return weights


def portfolio_returns_from_dynamic_weights(returns: pd.DataFrame, weights: pd.DataFrame, rebalance_lag: int = 1) -> pd.Series:
    """
    Uses signal weights with a lag to avoid look-ahead bias.
    """
    w = weights.shift(rebalance_lag).reindex(returns.index).ffill().dropna(how="all")
    r = returns.reindex(w.index).dropna(how="any")
    w = w.reindex(r.index).fillna(0.0)
    out = (w * r).sum(axis=1)
    out.name = "Alpha Composite Portfolio"
    return out


def alpha_turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def alpha_signal_snapshot(signal: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if signal.empty:
        return pd.DataFrame()
    last_date = signal.dropna(how="all").index.max()
    s = signal.loc[last_date].sort_values(ascending=False)
    w = weights.reindex(signal.index).ffill().loc[last_date]
    rows = []
    for ticker, score in s.items():
        rows.append({
            "Ticker": ticker,
            "Display Name": COMMODITY_UNIVERSE.get(ticker, {}).get("display", ticker),
            "Latest Alpha Score": score,
            "Latest Alpha Weight": w.get(ticker, np.nan),
            "Signal Rank": int(s.rank(ascending=False).loc[ticker]),
            "As Of": last_date.strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def alpha_governance_table(alpha_returns: pd.Series, benchmark: pd.Series, weights: pd.DataFrame, ic_table: pd.DataFrame, rf: float) -> pd.DataFrame:
    p, b = align_two(alpha_returns, benchmark, "Alpha", "Benchmark")
    avg_turnover = alpha_turnover(weights).reindex(p.index).mean()
    mean_ic = ic_table.loc[ic_table["Signal"] == "Composite Alpha", "Mean IC"].iloc[0] if "Composite Alpha" in list(ic_table["Signal"]) else np.nan
    icir = ic_table.loc[ic_table["Signal"] == "Composite Alpha", "ICIR"].iloc[0] if "Composite Alpha" in list(ic_table["Signal"]) else np.nan

    rows = [
        {"Check": "Positive Mean IC", "Value": mean_ic, "Status": "PASS" if pd.notna(mean_ic) and mean_ic > 0 else "WATCH"},
        {"Check": "ICIR", "Value": icir, "Status": "PASS" if pd.notna(icir) and icir > 0.25 else "WATCH"},
        {"Check": "Sharpe", "Value": sharpe_ratio(p, rf), "Status": "PASS" if sharpe_ratio(p, rf) > 0.5 else "WATCH"},
        {"Check": "Max Drawdown", "Value": max_drawdown(p), "Status": "PASS" if max_drawdown(p) > -0.35 else "WATCH"},
        {"Check": "Tracking Error", "Value": tracking_error(p, b), "Status": "PASS" if tracking_error(p, b) < 0.25 else "WATCH"},
        {"Check": "Average Daily Turnover", "Value": avg_turnover, "Status": "PASS" if avg_turnover < 0.50 else "WATCH"},
    ]
    return pd.DataFrame(rows)


def build_alpha_engine(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf: float,
    max_weight: float,
    momentum_short: int,
    momentum_long: int,
    meanrev_window: int,
    vol_window: int,
    ic_horizon: int,
) -> Dict[str, Any]:
    signals = build_alpha_signals(
        returns=returns,
        prices=prices,
        momentum_short=momentum_short,
        momentum_long=momentum_long,
        meanrev_window=meanrev_window,
        vol_window=vol_window,
    )
    composite = signals["Composite Alpha"].reindex(returns.index).dropna(how="all")
    weights = alpha_weights_from_signal(composite, max_weight=max_weight, long_only=True)
    alpha_ret = portfolio_returns_from_dynamic_weights(returns, weights, rebalance_lag=1)
    ic_table = ic_summary_table(signals, returns, horizon=ic_horizon)
    snapshot = alpha_signal_snapshot(composite, weights)

    return {
        "signals": signals,
        "composite": composite,
        "weights": weights,
        "returns": alpha_ret,
        "ic_table": ic_table,
        "snapshot": snapshot,
        "turnover": alpha_turnover(weights),
    }


# ============================================================
# GARCH
# ============================================================

@dataclass
class GarchSpec:
    key: str
    display: str
    vol: str = "GARCH"
    p: int = 1
    o: int = 0
    q: int = 1
    power: float = 2.0
    dist: str = "normal"


GARCH_SPECS = {
    "garch_normal": GarchSpec("garch_normal", "GARCH(1,1) Normal", "GARCH", 1, 0, 1, 2.0, "normal"),
    "garch_t": GarchSpec("garch_t", "GARCH(1,1) Student's T", "GARCH", 1, 0, 1, 2.0, "StudentsT"),
    "gjr_normal": GarchSpec("gjr_normal", "GJR-GARCH Normal", "GARCH", 1, 1, 1, 2.0, "normal"),
    "gjr_t": GarchSpec("gjr_t", "GJR-GARCH Student's T", "GARCH", 1, 1, 1, 2.0, "StudentsT"),
    "tarch_t": GarchSpec("tarch_t", "TARCH/ZARCH Student's T", "GARCH", 1, 1, 1, 1.0, "StudentsT"),
    "egarch_t": GarchSpec("egarch_t", "EGARCH Student's T", "EGARCH", 1, 1, 1, 2.0, "StudentsT"),
}


def _param(params: pd.Series, names: List[str]) -> float:
    for n in names:
        if n in params.index:
            return float(params[n])
    return 0.0


def fit_garch(r: pd.Series, ticker: str, spec: GarchSpec) -> Dict[str, Any]:
    x = 100 * r.dropna()
    if len(x) < 500:
        raise ValueError("Not enough observations for robust GARCH.")

    if spec.vol == "EGARCH":
        am = arch_model(x, mean="Constant", vol="EGARCH", p=spec.p, o=spec.o, q=spec.q, dist=spec.dist, rescale=False)
    else:
        am = arch_model(x, mean="Constant", vol="GARCH", p=spec.p, o=spec.o, q=spec.q, power=spec.power, dist=spec.dist, rescale=False)

    res = am.fit(update_freq=0, disp="off", show_warning=False)
    params = res.params.copy()
    alpha = _param(params, ["alpha[1]", "alpha"])
    beta = _param(params, ["beta[1]", "beta"])
    gamma = _param(params, ["gamma[1]", "gamma"])
    omega = _param(params, ["omega"])

    if spec.vol == "EGARCH":
        persistence = beta
    else:
        persistence = alpha + beta + 0.5 * gamma if spec.o > 0 else alpha + beta

    half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan

    try:
        f = res.forecast(horizon=21, reindex=False)
        var_path = f.variance.iloc[-1]
        vol1 = float(np.sqrt(var_path.iloc[0]))
        vol5 = float(np.sqrt(var_path.iloc[:5].mean()))
        vol21 = float(np.sqrt(var_path.iloc[:21].mean()))
    except Exception:
        vol1 = vol5 = vol21 = np.nan

    return {
        "Ticker": ticker,
        "Instrument": COMMODITY_UNIVERSE.get(ticker, {}).get("display", ticker),
        "Model Key": spec.key,
        "Model": spec.display,
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "Log Likelihood": float(res.loglikelihood),
        "Omega": omega,
        "Alpha": alpha,
        "Gamma": gamma,
        "Beta": beta,
        "Persistence": persistence,
        "Half-Life Days": half_life,
        "1D Forecast Vol %": vol1,
        "5D Forecast Vol %": vol5,
        "21D Forecast Vol %": vol21,
        "Conditional Vol": res.conditional_volatility.copy(),
        "Std Resid": res.std_resid.copy(),
        "Returns %": x,
        "Summary": str(res.summary()),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def run_garch_suite(returns: pd.DataFrame, model_keys: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    best_rows = []
    store = {}

    for ticker in returns.columns:
        packs = []
        for key in model_keys:
            try:
                p = fit_garch(returns[ticker], ticker, GARCH_SPECS[key])
                packs.append(p)
                store[f"{ticker}__{key}"] = p
            except Exception as exc:
                rows.append({
                    "Ticker": ticker,
                    "Instrument": COMMODITY_UNIVERSE.get(ticker, {}).get("display", ticker),
                    "Model": GARCH_SPECS[key].display,
                    "Status": f"Failed: {exc}",
                })

        packs = sorted(packs, key=lambda z: (z["BIC"], z["AIC"]))
        for rank, p in enumerate(packs, 1):
            row = {k: v for k, v in p.items() if k not in ["Conditional Vol", "Std Resid", "Returns %", "Summary"]}
            row["Rank by BIC"] = rank
            row["Status"] = "OK"
            rows.append(row)

        if packs:
            b = packs[0]
            best_rows.append({
                "Ticker": ticker,
                "Instrument": b["Instrument"],
                "Best Model": b["Model"],
                "Best Model Key": b["Model Key"],
                "BIC": b["BIC"],
                "Persistence": b["Persistence"],
                "Half-Life Days": b["Half-Life Days"],
                "1D Forecast Vol %": b["1D Forecast Vol %"],
                "5D Forecast Vol %": b["5D Forecast Vol %"],
                "21D Forecast Vol %": b["21D Forecast Vol %"],
            })

    return pd.DataFrame(rows), pd.DataFrame(best_rows), store


# ============================================================
# CHARTS
# ============================================================

def layout(fig: go.Figure, title: str, height: int = 650) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=QFA_COLORS["charcoal"], size=20)),
        template="plotly_white",
        height=height,
        autosize=True,
        margin=dict(l=35, r=35, t=75, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        font=dict(family="DejaVu Sans, Arial, sans-serif", size=13, color=QFA_COLORS["charcoal"]),
        colorway=QFA_SEQUENCE,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        gridcolor="#e5e7eb",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color=QFA_COLORS["slate"]),
        titlefont=dict(color=QFA_COLORS["slate"]),
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
        tickfont=dict(color=QFA_COLORS["slate"]),
        titlefont=dict(color=QFA_COLORS["slate"]),
    )
    return fig


def fig_prices(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=COMMODITY_UNIVERSE.get(col, {}).get("display", col), line=dict(color=QFA_SEQUENCE[list(norm.columns).index(col) % len(QFA_SEQUENCE)], width=2.4)))
    fig.update_yaxes(title="Base 100")
    fig.update_xaxes(title="Date")
    return layout(fig, "Normalized Commodity Prices — Base 100")


def fig_strategy_cumulative(strategy_returns: Dict[str, pd.Series], benchmark: pd.Series, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for name in selected:
        if name in strategy_returns:
            r = strategy_returns[name]
            eq = (1 + r.dropna()).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name, line=dict(width=2.8, color=color_for_name(name, selected.index(name) if name in selected else 0))))
    b = (1 + benchmark.dropna()).cumprod()
    fig.add_trace(go.Scatter(x=b.index, y=b.values, mode="lines", name=BENCHMARK_NAME, line=dict(width=2.4, dash="dash", color=STRATEGY_COLORS[BENCHMARK_NAME])))
    fig.update_yaxes(title="Growth of $1")
    return layout(fig, "Strategy Cumulative Return vs Benchmark", 700)


def fig_drawdowns(strategy_returns: Dict[str, pd.Series], benchmark: pd.Series, selected: List[str]) -> go.Figure:
    def dd(r):
        eq = (1 + r.dropna()).cumprod()
        return eq / eq.cummax() - 1
    fig = go.Figure()
    for name in selected:
        if name in strategy_returns:
            d = dd(strategy_returns[name])
            fig.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", name=name, line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)))
    bd = dd(benchmark)
    fig.add_trace(go.Scatter(x=bd.index, y=bd.values, mode="lines", name=BENCHMARK_NAME, line=dict(dash="dash", color=STRATEGY_COLORS[BENCHMARK_NAME], width=2.2)))
    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return layout(fig, "Strategy Drawdown Comparison", 700)


def fig_weights(weights_df: pd.DataFrame, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for col in selected:
        if col in weights_df.columns:
            fig.add_trace(go.Bar(x=weights_df["Display Name"], y=weights_df[col], name=col, marker_color=color_for_name(col, selected.index(col) if col in selected else 0)))
    fig.update_yaxes(title="Weight", tickformat=".0%")
    return layout(fig, "Portfolio Weights by Strategy", 680)


def fig_risk_return(metrics: pd.DataFrame) -> go.Figure:
    df = metrics.copy()
    fig = px.scatter(
        df,
        x="Annual Volatility",
        y="Annual Return",
        color="Strategy / Instrument",
        size=(df["Sharpe"].clip(lower=0).fillna(0.01) + 0.05),
        hover_data=["Sharpe", "Max Drawdown", "Tracking Error vs Benchmark", "Beta vs Benchmark"],
        template="plotly_white",
        color_discrete_sequence=QFA_SEQUENCE,
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Risk / Return Map", 680)


def fig_performance_bars(metrics: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Annual Return", "Annual Volatility", "Sharpe", "Max Drawdown"])
    df = metrics.copy()
    x = df["Strategy / Instrument"]
    fig.add_trace(go.Bar(x=x, y=df["Annual Return"], name="Annual Return", marker_color=QFA_COLORS["navy"]), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=df["Annual Volatility"], name="Annual Volatility", marker_color=QFA_COLORS["steel"]), row=1, col=2)
    fig.add_trace(go.Bar(x=x, y=df["Sharpe"], name="Sharpe", marker_color=QFA_COLORS["muted_gold"]), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=df["Max Drawdown"], name="Max Drawdown", marker_color=QFA_COLORS["risk_red"]), row=2, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=2, col=2)
    return layout(fig, "Performance Metrics Dashboard", 820)


def fig_te_beta(strategy_returns: Dict[str, pd.Series], benchmark: pd.Series, selected: List[str], window: int, te_target: float, te_band: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling Tracking Error", "Rolling Beta"])
    for name in selected:
        if name in strategy_returns:
            te = rolling_tracking_error(strategy_returns[name], benchmark, window)
            beta = rolling_beta(strategy_returns[name], benchmark, window)
            fig.add_trace(go.Scatter(x=te.index, y=te.values, mode="lines", name=f"{name} TE", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=beta.index, y=beta.values, mode="lines", name=f"{name} Beta", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=2, col=1)
    fig.add_hline(y=te_target, line_dash="dash", annotation_text="TE Target", row=1, col=1)
    fig.add_hline(y=te_target + te_band, line_dash="dot", annotation_text="Upper Band", row=1, col=1)
    fig.add_hline(y=max(te_target - te_band, 0), line_dash="dot", annotation_text="Lower Band", row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Beta = 1", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return layout(fig, "Rolling Tracking Error and Rolling Beta", 820)


def fig_te_beta_map(metrics: pd.DataFrame) -> go.Figure:
    df = metrics.copy()
    fig = px.scatter(
        df,
        x="Tracking Error vs Benchmark",
        y="Beta vs Benchmark",
        color="Strategy / Instrument",
        size=(df["Annual Volatility"].abs().fillna(0.01) + 0.01),
        hover_data=["Annual Return", "Sharpe", "Max Drawdown", "Information Ratio"],
        template="plotly_white",
        color_discrete_sequence=QFA_SEQUENCE,
    )
    fig.update_xaxes(tickformat=".0%")
    return layout(fig, "Benchmark-Relative Map — Tracking Error vs Beta", 680)


def fig_var_cvar(strategy_returns: Dict[str, pd.Series], selected: List[str], window: int) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling VaR 95%", "Rolling CVaR 95%"])
    for name in selected:
        if name in strategy_returns:
            risk = rolling_var_cvar(strategy_returns[name], window)
            fig.add_trace(go.Scatter(x=risk.index, y=risk["VaR 95%"], mode="lines", name=f"{name} VaR", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=risk.index, y=risk["CVaR 95%"], mode="lines", name=f"{name} CVaR", line=dict(color=color_for_name(name, selected.index(name) if name in selected else 0), width=2.2)), row=2, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    return layout(fig, "Rolling VaR / CVaR by Strategy", 800)


def fig_tail_bars(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ["VaR 95% Daily", "CVaR 95% Daily", "VaR 99% Daily", "CVaR 99% Daily"]:
        fig.add_trace(go.Bar(x=metrics["Strategy / Instrument"], y=metrics[col], name=col, marker_color=QFA_SEQUENCE[["VaR 95% Daily", "CVaR 95% Daily", "VaR 99% Daily", "CVaR 99% Daily"].index(col) % len(QFA_SEQUENCE)]))
    fig.update_yaxes(tickformat=".1%")
    return layout(fig, "Tail Risk Comparison — VaR / CVaR", 680)


def fig_corr(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    labels = [COMMODITY_UNIVERSE.get(c, {}).get("display", c) for c in corr.columns]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale=[[0, "#0f172a"], [0.5, "#f8fafc"], [1, "#b08900"]], zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="Correlation")
    ))
    return layout(fig, "Commodity Return Correlation Matrix", 650)



def fig_log_return_difference(log_ret: pd.DataFrame, base_ticker: str) -> go.Figure:
    diff = return_difference_series(log_ret, base_ticker)
    fig = go.Figure()
    for i, col in enumerate(diff.columns):
        fig.add_trace(go.Scatter(
            x=diff.index,
            y=diff[col],
            mode="lines",
            name=col,
            line=dict(color=QFA_SEQUENCE[i % len(QFA_SEQUENCE)], width=1.8),
        ))
    fig.update_yaxes(title="Daily Log Return Difference")
    return layout(fig, f"ETF / Instrument Log Return Difference vs {base_ticker}", 720)


def fig_log_return_difference_bollinger(log_ret: pd.DataFrame, spread_name: str, window: int, n_std: float) -> go.Figure:
    if spread_name not in log_ret.columns:
        base = log_ret.columns[0]
        diff = return_difference_series(log_ret, base)
        if diff.empty:
            return layout(go.Figure(), "Log Return Difference Bollinger Bands")
        s = diff.iloc[:, 0]
    else:
        s = log_ret[spread_name]

    bb = bollinger_frame(s, window=window, n_std=n_std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Value"], mode="lines", name="Difference", line=dict(color=QFA_COLORS["navy"], width=1.8)))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Rolling Mean"], mode="lines", name="Rolling Mean", line=dict(color=QFA_COLORS["muted_gold"], width=2.2)))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Upper Band"], mode="lines", name="Upper Band", line=dict(color=QFA_COLORS["slate"], width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["Lower Band"], mode="lines", name="Lower Band", line=dict(color=QFA_COLORS["slate"], width=1.2, dash="dash"), fill="tonexty", fillcolor="rgba(15, 23, 42, 0.06)"))
    fig.update_yaxes(title="Log Return Difference")
    return layout(fig, f"Bollinger Bands — {spread_name}", 760)


def fig_rolling_sharpe(strategy_returns: Dict[str, pd.Series], selected: List[str], rf: float, window: int) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate(selected):
        if name in strategy_returns:
            rs = rolling_sharpe(strategy_returns[name], rf, window)
            fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name=name, line=dict(color=color_for_name(name, i), width=2.0)))
    fig.add_hline(y=0, line_dash="dash", annotation_text="Sharpe = 0")
    fig.update_yaxes(title="Rolling Sharpe")
    return layout(fig, f"{window}-Day Rolling Sharpe by Strategy", 720)


def fig_underwater_recovery(strategy_returns: Dict[str, pd.Series], selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate(selected):
        if name not in strategy_returns:
            continue
        r = strategy_returns[name].dropna()
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name, line=dict(color=color_for_name(name, i), width=2.0)))
    fig.update_yaxes(title="Underwater Drawdown", tickformat=".0%")
    return layout(fig, "Underwater Chart — Strategy Recovery Profile", 720)



def fig_alpha_scores(signal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(signal.columns):
        fig.add_trace(go.Scatter(
            x=signal.index,
            y=signal[col],
            mode="lines",
            name=COMMODITY_UNIVERSE.get(col, {}).get("display", col),
            line=dict(color=QFA_SEQUENCE[i % len(QFA_SEQUENCE)], width=1.9),
        ))
    fig.update_yaxes(title="Alpha Score")
    return layout(fig, "Composite Alpha Scores by Instrument", 720)


def fig_alpha_weights(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Scatter(
            x=weights.index,
            y=weights[col],
            mode="lines",
            stackgroup="one",
            name=COMMODITY_UNIVERSE.get(col, {}).get("display", col),
            line=dict(color=QFA_SEQUENCE[i % len(QFA_SEQUENCE)], width=1.0),
        ))
    fig.update_yaxes(title="Dynamic Weight", tickformat=".0%")
    return layout(fig, "Alpha Engine Dynamic Weights", 720)


def fig_ic_series(signals: Dict[str, pd.DataFrame], returns: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    for i, (name, sig) in enumerate(signals.items()):
        ic = information_coefficient(sig, returns, horizon=horizon, method="spearman")
        fig.add_trace(go.Scatter(x=ic.index, y=ic.values, mode="lines", name=name, line=dict(color=QFA_SEQUENCE[i % len(QFA_SEQUENCE)], width=1.9)))
    fig.add_hline(y=0, line_dash="dash", annotation_text="IC = 0")
    fig.update_yaxes(title="Spearman Rank IC")
    return layout(fig, f"{horizon}-Day Forward Information Coefficient", 720)


def fig_alpha_cumulative(alpha_ret: pd.Series, benchmark: pd.Series, equal_weight_ret: pd.Series) -> go.Figure:
    fig = go.Figure()
    a = (1 + alpha_ret.dropna()).cumprod()
    e = (1 + equal_weight_ret.reindex(a.index).dropna()).cumprod()
    b = (1 + benchmark.reindex(a.index).dropna()).cumprod()

    fig.add_trace(go.Scatter(x=a.index, y=a.values, mode="lines", name="Alpha Composite", line=dict(color=QFA_COLORS["navy"], width=3.0)))
    fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines", name="Equal Weight", line=dict(color=QFA_COLORS["muted_gold"], width=2.4)))
    fig.add_trace(go.Scatter(x=b.index, y=b.values, mode="lines", name=BENCHMARK_NAME, line=dict(color=QFA_COLORS["gray"], width=2.2, dash="dash")))
    fig.update_yaxes(title="Growth of $1")
    return layout(fig, "Alpha Portfolio vs Equal Weight vs Benchmark", 740)


def fig_alpha_drawdown(alpha_ret: pd.Series, equal_weight_ret: pd.Series) -> go.Figure:
    def dd(r):
        eq = (1 + r.dropna()).cumprod()
        return eq / eq.cummax() - 1
    a = dd(alpha_ret)
    e = dd(equal_weight_ret.reindex(a.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.index, y=a.values, mode="lines", fill="tozeroy", name="Alpha Composite", line=dict(color=QFA_COLORS["risk_red"], width=2.5)))
    fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines", name="Equal Weight", line=dict(color=QFA_COLORS["gray"], width=2.0)))
    fig.update_yaxes(title="Drawdown", tickformat=".0%")
    return layout(fig, "Alpha Portfolio Drawdown", 700)


def fig_alpha_turnover(turnover: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=turnover.index, y=turnover.values, mode="lines", name="Daily Turnover", line=dict(color=QFA_COLORS["slate"], width=1.8)))
    fig.add_trace(go.Scatter(x=turnover.index, y=turnover.rolling(21).mean(), mode="lines", name="21D Avg Turnover", line=dict(color=QFA_COLORS["muted_gold"], width=2.4)))
    fig.update_yaxes(title="Turnover")
    return layout(fig, "Alpha Portfolio Turnover", 650)


def fig_garch_best(best: pd.DataFrame, store: Dict[str, Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for _, row in best.iterrows():
        key = f"{row['Ticker']}__{row['Best Model Key']}"
        if key in store:
            v = store[key]["Conditional Vol"]
            fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name=f"{row['Instrument']} — {row['Best Model']}"))
    fig.update_yaxes(title="Daily Conditional Volatility (%)")
    return layout(fig, "Best-BIC GARCH Conditional Volatility", 720)


def fig_garch_rank(garch: pd.DataFrame) -> go.Figure:
    df = garch[garch["Status"] == "OK"].copy()
    if df.empty:
        return layout(go.Figure(), "GARCH Ranking Unavailable")
    fig = px.bar(df, x="Instrument", y="BIC", color="Model", barmode="group", template="plotly_white")
    return layout(fig, "GARCH Model Ranking by BIC — Lower is Better", 680)


def fig_garch_resid(ticker: str, model_key: str, store: Dict[str, Dict[str, Any]]) -> go.Figure:
    key = f"{ticker}__{model_key}"
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Returns %", "Conditional Vol", "Std Residuals", "Squared Std Residuals"])
    if key not in store:
        return layout(fig, "GARCH Diagnostics Unavailable")
    p = store[key]
    r = p["Returns %"]
    v = p["Conditional Vol"]
    resid = p["Std Resid"].dropna()
    fig.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines", name="Returns %"), row=1, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name="Cond Vol"), row=1, col=2)
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines", name="Std Resid"), row=2, col=1)
    fig.add_trace(go.Scatter(x=resid.index, y=(resid**2).values, mode="lines", name="Sq Resid"), row=2, col=2)
    return layout(fig, f"{COMMODITY_UNIVERSE.get(ticker, {}).get('display', ticker)} — GARCH Diagnostics", 850)


# ============================================================
# QFA QUANTSTATS-STYLE METRICS ENGINE
# ============================================================

def cumulative_return(r: pd.Series) -> float:
    x = _clean_return_series(r)
    return (1 + x).prod() - 1 if len(x) else np.nan


def cagr(r: pd.Series) -> float:
    x = _clean_return_series(r)
    if len(x) < 2:
        return np.nan
    years = max((x.index[-1] - x.index[0]).days / 365.25, len(x) / TRADING_DAYS)
    return (1 + x).prod() ** (1 / years) - 1 if years > 0 else np.nan


def downside_deviation(r: pd.Series, rf: float = 0.0) -> float:
    x = _clean_return_series(r)
    daily_rf = rf / TRADING_DAYS
    downside = np.minimum(x - daily_rf, 0.0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS) if len(downside) else np.nan


def sortino_ratio_qfa(r: pd.Series, rf: float = 0.0) -> float:
    dd = downside_deviation(r, rf)
    return np.nan if pd.isna(dd) or dd == 0 else (cagr(r) - rf) / dd


def calmar_ratio_qfa(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    return np.nan if pd.isna(mdd) or mdd == 0 else cagr(r) / mdd


def omega_ratio_qfa(r: pd.Series, threshold: float = 0.0) -> float:
    x = _clean_return_series(r)
    gains = (x - threshold).clip(lower=0).sum()
    losses = (threshold - x).clip(lower=0).sum()
    return np.nan if losses == 0 else gains / losses


def ulcer_index_qfa(r: pd.Series) -> float:
    x = _clean_return_series(r)
    eq = (1 + x).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    return np.sqrt(np.mean(dd ** 2)) if len(dd) else np.nan


def ulcer_performance_index_qfa(r: pd.Series, rf: float = 0.0) -> float:
    ui = ulcer_index_qfa(r)
    return np.nan if pd.isna(ui) or ui == 0 else (cagr(r) - rf) / (ui / 100)


def expected_return_by_period(r: pd.Series, periods: int) -> float:
    x = _clean_return_series(r)
    return x.mean() * periods if len(x) else np.nan


def win_rate_qfa(r: pd.Series) -> float:
    x = _clean_return_series(r)
    wins = (x > 0).sum()
    total = (x != 0).sum()
    return np.nan if total == 0 else wins / total


def payoff_ratio_qfa(r: pd.Series) -> float:
    x = _clean_return_series(r)
    avg_win = x[x > 0].mean()
    avg_loss = abs(x[x < 0].mean())
    return np.nan if pd.isna(avg_win) or pd.isna(avg_loss) or avg_loss == 0 else avg_win / avg_loss


def profit_factor_qfa(r: pd.Series) -> float:
    x = _clean_return_series(r)
    gross_profit = x[x > 0].sum()
    gross_loss = abs(x[x < 0].sum())
    return np.nan if gross_loss == 0 else gross_profit / gross_loss


def gain_to_pain_qfa(r: pd.Series) -> float:
    x = _clean_return_series(r)
    total_gain = x.sum()
    total_pain = abs(x[x < 0].sum())
    return np.nan if total_pain == 0 else total_gain / total_pain


def tail_ratio_qfa(r: pd.Series) -> float:
    x = _clean_return_series(r)
    left = abs(x.quantile(0.05))
    right = x.quantile(0.95)
    return np.nan if left == 0 else right / left


def common_sense_ratio_qfa(r: pd.Series) -> float:
    pf = profit_factor_qfa(r)
    tr = tail_ratio_qfa(r)
    return pf * tr if pd.notna(pf) and pd.notna(tr) else np.nan


def kelly_criterion_qfa(r: pd.Series) -> float:
    wr = win_rate_qfa(r)
    pr = payoff_ratio_qfa(r)
    return np.nan if pd.isna(wr) or pd.isna(pr) or pr == 0 else wr - (1 - wr) / pr


def max_consecutive_count(r: pd.Series, positive: bool = True) -> int:
    x = _clean_return_series(r)
    cond = x > 0 if positive else x < 0
    max_run = run = 0
    for flag in cond:
        if flag:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def longest_drawdown_days_qfa(r: pd.Series) -> int:
    x = _clean_return_series(r)
    eq = (1 + x).cumprod()
    underwater = eq < eq.cummax()
    max_days = days = 0
    for flag in underwater:
        if flag:
            days += 1
            max_days = max(max_days, days)
        else:
            days = 0
    return int(max_days)


def active_return_series(p: pd.Series, b: pd.Series) -> pd.Series:
    pp, bb = align_two(_clean_return_series(p), _clean_return_series(b), "Portfolio", "Benchmark")
    return pp - bb


def r_squared_qfa(p: pd.Series, b: pd.Series) -> float:
    pp, bb = align_two(_clean_return_series(p), _clean_return_series(b), "Portfolio", "Benchmark")
    corr = pp.corr(bb)
    return corr ** 2 if pd.notna(corr) else np.nan


def treynor_ratio_qfa(p: pd.Series, b: pd.Series, rf: float) -> float:
    beta = beta_to_benchmark(p, b)
    return np.nan if pd.isna(beta) or beta == 0 else (cagr(p) - rf) / beta


def qfa_quantstats_style_metrics(strategy_name: str, r: pd.Series, benchmark: pd.Series, rf: float) -> pd.DataFrame:
    p, b = align_two(_clean_return_series(r), _clean_return_series(benchmark), "Strategy", "Benchmark")

    rows = [
        ("Risk-Free Rate", rf, rf, "pct"),
        ("Time in Market", (p != 0).mean(), (b != 0).mean(), "pct"),
        ("Cumulative Return", cumulative_return(p), cumulative_return(b), "pct"),
        ("CAGR", cagr(p), cagr(b), "pct"),
        ("Sharpe", sharpe_ratio(p, rf), sharpe_ratio(b, rf), "num"),
        ("Sortino", sortino_ratio_qfa(p, rf), sortino_ratio_qfa(b, rf), "num"),
        ("Calmar", calmar_ratio_qfa(p), calmar_ratio_qfa(b), "num"),
        ("Omega", omega_ratio_qfa(p), omega_ratio_qfa(b), "num"),
        ("Max Drawdown", max_drawdown(p), max_drawdown(b), "pct"),
        ("Longest DD Days", longest_drawdown_days_qfa(p), longest_drawdown_days_qfa(b), "int"),
        ("Volatility (ann.)", annualized_volatility(p), annualized_volatility(b), "pct"),
        ("Downside Deviation", downside_deviation(p, rf), downside_deviation(b, rf), "pct"),
        ("R-Squared vs Benchmark", r_squared_qfa(p, b), 1.0, "num"),
        ("Beta", beta_to_benchmark(p, b), 1.0, "num"),
        ("Treynor Ratio", treynor_ratio_qfa(p, b, rf), treynor_ratio_qfa(b, b, rf), "num"),
        ("Information Ratio", information_ratio(p, b), 0.0, "num"),
        ("Skew", p.skew(), b.skew(), "num"),
        ("Kurtosis", p.kurtosis(), b.kurtosis(), "num"),
        ("Expected Daily", expected_return_by_period(p, 1), expected_return_by_period(b, 1), "pct"),
        ("Expected Monthly", expected_return_by_period(p, 21), expected_return_by_period(b, 21), "pct"),
        ("Expected Yearly", expected_return_by_period(p, TRADING_DAYS), expected_return_by_period(b, TRADING_DAYS), "pct"),
        ("Kelly Criterion", kelly_criterion_qfa(p), kelly_criterion_qfa(b), "pct"),
        ("Daily Value-at-Risk 95%", historical_var(p, 0.95), historical_var(b, 0.95), "pct"),
        ("Expected Shortfall / CVaR 95%", historical_cvar(p, 0.95), historical_cvar(b, 0.95), "pct"),
        ("Daily Value-at-Risk 99%", historical_var(p, 0.99), historical_var(b, 0.99), "pct"),
        ("Expected Shortfall / CVaR 99%", historical_cvar(p, 0.99), historical_cvar(b, 0.99), "pct"),
        ("Win Rate", win_rate_qfa(p), win_rate_qfa(b), "pct"),
        ("Max Consecutive Wins", max_consecutive_count(p, True), max_consecutive_count(b, True), "int"),
        ("Max Consecutive Losses", max_consecutive_count(p, False), max_consecutive_count(b, False), "int"),
        ("Gain/Pain Ratio", gain_to_pain_qfa(p), gain_to_pain_qfa(b), "num"),
        ("Payoff Ratio", payoff_ratio_qfa(p), payoff_ratio_qfa(b), "num"),
        ("Profit Factor", profit_factor_qfa(p), profit_factor_qfa(b), "num"),
        ("Tail Ratio", tail_ratio_qfa(p), tail_ratio_qfa(b), "num"),
        ("Common Sense Ratio", common_sense_ratio_qfa(p), common_sense_ratio_qfa(b), "num"),
        ("Ulcer Index", ulcer_index_qfa(p), ulcer_index_qfa(b), "num"),
        ("Ulcer Performance Index", ulcer_performance_index_qfa(p, rf), ulcer_performance_index_qfa(b, rf), "num"),
    ]

    return pd.DataFrame(rows, columns=["Metric", strategy_name, BENCHMARK_NAME, "Format"])


def _format_metric_value(v, fmt: str) -> str:
    if pd.isna(v):
        return ""
    if fmt == "pct":
        return f"{v:.2%}"
    if fmt == "int":
        return f"{int(v):,}"
    return f"{v:.3f}"


def qfa_metrics_display_table(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    strategy_col = [c for c in out.columns if c not in ["Metric", BENCHMARK_NAME, "Format"]][0]
    out[strategy_col] = [_format_metric_value(v, f) for v, f in zip(out[strategy_col], out["Format"])]
    out[BENCHMARK_NAME] = [_format_metric_value(v, f) for v, f in zip(out[BENCHMARK_NAME], out["Format"])]
    return out.drop(columns=["Format"])


# ============================================================
# EXPORTS
# ============================================================

def df_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def returns_csv(series_map: Dict[str, pd.Series]) -> bytes:
    return pd.concat(series_map, axis=1).to_csv().encode("utf-8")


def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="qfa-table", escape=False)


def _fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True, "displaylogo": False})


def monthly_returns_table_qfa(r: pd.Series) -> pd.DataFrame:
    x = _clean_return_series(r)
    if x.empty:
        return pd.DataFrame()
    monthly = (1 + x).resample("M").prod() - 1
    tbl = monthly.to_frame("Return")
    tbl["Year"] = tbl.index.year
    tbl["Month"] = tbl.index.strftime("%b")
    pivot = tbl.pivot(index="Year", columns="Month", values="Return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    pivot["Year Total"] = (1 + monthly).groupby(monthly.index.year).prod() - 1
    return pivot


def annual_returns_table_qfa(r: pd.Series) -> pd.DataFrame:
    x = _clean_return_series(r)
    if x.empty:
        return pd.DataFrame()
    annual = (1 + x).resample("Y").prod() - 1
    return pd.DataFrame({
        "Year": annual.index.year,
        "Annual Return": annual.values,
    })


def drawdown_details_qfa(r: pd.Series, top_n: int = 10) -> pd.DataFrame:
    x = _clean_return_series(r)
    if x.empty:
        return pd.DataFrame()
    eq = (1 + x).cumprod()
    dd = eq / eq.cummax() - 1
    underwater = dd < 0

    episodes = []
    start = None
    valley = None
    valley_dd = 0.0

    for date, flag in underwater.items():
        if flag and start is None:
            start = date
            valley = date
            valley_dd = dd.loc[date]
        elif flag and start is not None:
            if dd.loc[date] < valley_dd:
                valley = date
                valley_dd = dd.loc[date]
        elif (not flag) and start is not None:
            end = date
            episodes.append({
                "Start": start.strftime("%Y-%m-%d"),
                "Valley": valley.strftime("%Y-%m-%d"),
                "Recovered": end.strftime("%Y-%m-%d"),
                "Max Drawdown": valley_dd,
                "Days": int((end - start).days),
            })
            start = valley = None
            valley_dd = 0.0

    if start is not None:
        end = dd.index[-1]
        episodes.append({
            "Start": start.strftime("%Y-%m-%d"),
            "Valley": valley.strftime("%Y-%m-%d"),
            "Recovered": "Not Recovered",
            "Max Drawdown": valley_dd,
            "Days": int((end - start).days),
        })

    out = pd.DataFrame(episodes)
    if out.empty:
        return out
    return out.sort_values("Max Drawdown").head(top_n)


def html_table_formatted(df: pd.DataFrame, pct_cols: list[str] | None = None) -> str:
    pct_cols = pct_cols or []
    x = df.copy()
    for col in x.columns:
        if col in pct_cols:
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        elif pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:,.3f}")
    return x.to_html(index=False, border=0, classes="qfa-table", escape=False)


def monthly_heatmap_figure_qfa(r: pd.Series) -> go.Figure:
    monthly = monthly_returns_table_qfa(r)
    if monthly.empty:
        return layout(go.Figure(), "Monthly Returns Heatmap")
    z = monthly.drop(columns=["Year Total"], errors="ignore").values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(monthly.drop(columns=["Year Total"], errors="ignore").columns),
        y=[str(y) for y in monthly.index],
        colorscale=[[0, "#991b1b"], [0.5, "#f8fafc"], [1, "#166534"]],
        zmid=0,
        text=[[("" if pd.isna(v) else f"{v:.2%}") for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="Return"),
    ))
    return layout(fig, "Monthly Returns Heatmap", 620)


def annual_returns_figure_qfa(r: pd.Series) -> go.Figure:
    annual = annual_returns_table_qfa(r)
    fig = go.Figure()
    if not annual.empty:
        fig.add_trace(go.Bar(
            x=annual["Year"].astype(str),
            y=annual["Annual Return"],
            name="Annual Return",
            marker_color=QFA_COLORS["navy"],
        ))
        fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Annual Returns", 560)


def return_distribution_figure_qfa(r: pd.Series) -> go.Figure:
    x = _clean_return_series(r)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x.values,
        nbinsx=80,
        name="Daily Returns",
        marker_color=QFA_COLORS["slate"],
        opacity=0.85,
    ))
    fig.add_vline(x=x.mean(), line_dash="dash", annotation_text="Mean")
    fig.add_vline(x=historical_var(x, 0.95), line_dash="dot", annotation_text="VaR 95%")
    fig.update_xaxes(tickformat=".1%", title="Daily Return")
    fig.update_yaxes(title="Frequency")
    return layout(fig, "Daily Return Distribution", 560)


def rolling_metrics_figure_qfa(p: pd.Series, b: pd.Series, rf: float) -> go.Figure:
    pp, bb = align_two(p, b, "Portfolio", "Benchmark")
    roll_sharpe = rolling_sharpe(pp, rf, 63)
    roll_vol = pp.rolling(63).std() * np.sqrt(TRADING_DAYS)
    roll_beta = rolling_beta(pp, bb, 63)
    roll_te = rolling_tracking_error(pp, bb, 63)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Rolling Sharpe", "Rolling Volatility", "Rolling Beta vs ^GSPC", "Rolling Tracking Error vs ^GSPC"],
    )
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, mode="lines", name="Rolling Sharpe", line=dict(color=QFA_COLORS["navy"], width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, mode="lines", name="Rolling Volatility", line=dict(color=QFA_COLORS["steel"], width=2.0)), row=2, col=1)
    fig.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta.values, mode="lines", name="Rolling Beta", line=dict(color=QFA_COLORS["muted_gold"], width=2.0)), row=3, col=1)
    fig.add_trace(go.Scatter(x=roll_te.index, y=roll_te.values, mode="lines", name="Rolling TE", line=dict(color=QFA_COLORS["gray"], width=2.0)), row=4, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    return layout(fig, "Rolling Risk / Return Metrics", 980)


def report_content_audit_table_qfa() -> pd.DataFrame:
    return pd.DataFrame([
        {"Section": "QuantStats-style ratios", "Included": "YES", "Details": "Sharpe, Sortino, Calmar, Omega, Treynor, Information Ratio, Kelly, Gain/Pain, Profit Factor, Tail Ratio, Ulcer."},
        {"Section": "Return metrics", "Included": "YES", "Details": "CAGR, cumulative return, annual volatility, expected daily/monthly/yearly returns."},
        {"Section": "Benchmark-relative metrics", "Included": "YES", "Details": "Beta, R², Tracking Error, Treynor, Information Ratio vs S&P 500 (^GSPC)."},
        {"Section": "Tail risk", "Included": "YES", "Details": "VaR 95/99 and CVaR 95/99."},
        {"Section": "Drawdown analytics", "Included": "YES", "Details": "Max drawdown, longest drawdown days, drawdown episodes, underwater chart."},
        {"Section": "Charts", "Included": "YES", "Details": "Cumulative return, drawdown, rolling metrics, VaR/CVaR, monthly heatmap, annual return, return distribution."},
    ])


# ============================================================
# QFA INSTITUTIONAL TEARSHEET HTML (ENHANCED)
# ============================================================

def qfa_institutional_tearsheet_html(
    strategy_name: str,
    r: pd.Series,
    benchmark: pd.Series,
    rf: float,
    metrics_row: pd.Series,
) -> bytes:
    """
    QFA Institutional Tearsheet — QuantStats-format style report.
    ENHANCED: Includes all QuantStats ratios + advanced risk metrics + additional figures.
    """
    p, b = align_two(r, benchmark, "Portfolio", "Benchmark")

    # ============================================================
    # 1. QuantStats-Style Metrics (existing)
    # ============================================================
    qfa_qs_metrics = qfa_quantstats_style_metrics(strategy_name, p, b, rf)
    qfa_qs_display = qfa_metrics_display_table(qfa_qs_metrics)

    # ============================================================
    # 2. New Advanced Performance Ratios
    # ============================================================
    advanced_ratios = pd.DataFrame([
        {"Metric": "Gain-Loss Ratio (Bernardo-Ledoit)", "Value": f"{gain_loss_ratio(p):.4f}", "Benchmark": f"{gain_loss_ratio(b):.4f}"},
        {"Metric": "Martin Ratio (CAGR / Ulcer)", "Value": f"{martin_ratio(p, rf):.4f}", "Benchmark": f"{martin_ratio(b, rf):.4f}"},
        {"Metric": "Pain Index", "Value": f"{pain_index(p):.4f}", "Benchmark": f"{pain_index(b):.4f}"},
        {"Metric": "Pain Ratio", "Value": f"{pain_ratio(p, rf):.4f}", "Benchmark": f"{pain_ratio(b, rf):.4f}"},
        {"Metric": "Kappa 3 (Lower Partial Moment)", "Value": f"{kappa_3_ratio(p):.4f}", "Benchmark": f"{kappa_3_ratio(b):.4f}"},
        {"Metric": "Stutzer Performance Index", "Value": f"{stutzer_index(p):.4f}", "Benchmark": f"{stutzer_index(b):.4f}"},
    ])

    # ============================================================
    # 3. Cornish-Fisher Risk Metrics
    # ============================================================
    cf_metrics = pd.DataFrame([
        {"Metric": "Historical VaR 95%", "Value": f"{historical_var(p, 0.95):.2%}", "Benchmark": f"{historical_var(b, 0.95):.2%}"},
        {"Metric": "Cornish-Fisher VaR 95%", "Value": f"{cornish_fisher_var(p, 0.95):.2%}", "Benchmark": f"{cornish_fisher_var(b, 0.95):.2%}"},
        {"Metric": "Historical CVaR 95%", "Value": f"{historical_cvar(p, 0.95):.2%}", "Benchmark": f"{historical_cvar(b, 0.95):.2%}"},
        {"Metric": "Cornish-Fisher CVaR 95%", "Value": f"{cornish_fisher_cvar(p, 0.95):.2%}", "Benchmark": f"{cornish_fisher_cvar(b, 0.95):.2%}"},
        {"Metric": "Modified Sharpe (CF-VaR)", "Value": f"{modified_sharpe_cf(p, rf, 0.95):.4f}", "Benchmark": f"{modified_sharpe_cf(b, rf, 0.95):.4f}"},
        {"Metric": "Historical VaR 99%", "Value": f"{historical_var(p, 0.99):.2%}", "Benchmark": f"{historical_var(b, 0.99):.2%}"},
        {"Metric": "Cornish-Fisher VaR 99%", "Value": f"{cornish_fisher_var(p, 0.99):.2%}", "Benchmark": f"{cornish_fisher_var(b, 0.99):.2%}"},
    ])

    # ============================================================
    # 4. Benchmark-Relative Advanced Metrics
    # ============================================================
    capture = capture_ratios(p, b)
    benchmark_relative = pd.DataFrame([
        {"Metric": "Up Capture Ratio", "Value": f"{capture.get('Up Capture Ratio', np.nan):.4f}"},
        {"Metric": "Down Capture Ratio", "Value": f"{capture.get('Down Capture Ratio', np.nan):.4f}"},
        {"Metric": "Capture Ratio (Up/Down)", "Value": f"{capture.get('Capture Ratio (Up/Down)', np.nan):.4f}"},
        {"Metric": "Appraisal Ratio (Alpha / Residual Vol)", "Value": f"{appraisal_ratio(p, b, rf):.4f}"},
        {"Metric": "Information Ratio", "Value": f"{information_ratio(p, b):.4f}"},
        {"Metric": "Treynor Ratio", "Value": f"{treynor_ratio_qfa(p, b, rf):.4f}"},
        {"Metric": "R-Squared", "Value": f"{r_squared_qfa(p, b):.4f}"},
    ])

    # ============================================================
    # 5. Additional Figures (Charts)
    # ============================================================
    
    # Cumulative return
    eq = (1 + p).cumprod()
    beq = (1 + b).cumprod()
    dd = eq / eq.cummax() - 1
    risk = rolling_var_cvar(p, 63)
    te = rolling_tracking_error(p, b, 63)
    beta = rolling_beta(p, b, 63)
    
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=strategy_name, line=dict(color=QFA_COLORS["navy"], width=2.8)))
    fig_eq.add_trace(go.Scatter(x=beq.index, y=beq.values, mode="lines", name=BENCHMARK_NAME, line=dict(color=QFA_COLORS["gray"], width=2.2, dash="dash")))
    fig_eq.update_yaxes(title="Growth of $1")
    fig_eq = layout(fig_eq, f"{strategy_name} — Cumulative Return vs {BENCHMARK_NAME}", 560)
    
    # Drawdown
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown", line=dict(color=QFA_COLORS["risk_red"], width=2.2)))
    fig_dd.update_yaxes(tickformat=".0%")
    fig_dd = layout(fig_dd, f"{strategy_name} — Underwater Drawdown", 560)
    
    # Rolling risk
    fig_risk = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling Tracking Error", "Rolling Beta"])
    fig_risk.add_trace(go.Scatter(x=te.index, y=te.values, mode="lines", name="Tracking Error", line=dict(color=QFA_COLORS["muted_gold"], width=2.2)), row=1, col=1)
    fig_risk.add_trace(go.Scatter(x=beta.index, y=beta.values, mode="lines", name="Beta", line=dict(color=QFA_COLORS["slate"], width=2.2)), row=2, col=1)
    fig_risk.update_yaxes(tickformat=".0%", row=1, col=1)
    fig_risk = layout(fig_risk, f"{strategy_name} — Benchmark Relative Risk vs {BENCHMARK_NAME}", 680)
    
    # Tail risk
    fig_tail = go.Figure()
    for i, col in enumerate(risk.columns):
        fig_tail.add_trace(go.Scatter(x=risk.index, y=risk[col], mode="lines", name=col, line=dict(color=QFA_SEQUENCE[i % len(QFA_SEQUENCE)], width=2.0)))
    fig_tail.update_yaxes(tickformat=".1%")
    fig_tail = layout(fig_tail, f"{strategy_name} — Rolling VaR / CVaR", 560)
    
    # Rolling metrics
    fig_roll = rolling_metrics_figure_qfa(p, b, rf)
    
    # Monthly and annual figures
    fig_month = monthly_heatmap_figure_qfa(p)
    fig_annual = annual_returns_figure_qfa(p)
    fig_dist = return_distribution_figure_qfa(p)
    
    # NEW: Advanced ratios bar chart
    adv_plot_data = pd.DataFrame({
        "Metric": ["Gain-Loss", "Martin", "Pain Ratio", "Kappa 3", "Stutzer"],
        "Strategy": [gain_loss_ratio(p), martin_ratio(p, rf), pain_ratio(p, rf), kappa_3_ratio(p), stutzer_index(p)],
        "Benchmark": [gain_loss_ratio(b), martin_ratio(b, rf), pain_ratio(b, rf), kappa_3_ratio(b), stutzer_index(b)]
    })
    fig_adv = go.Figure()
    fig_adv.add_trace(go.Bar(x=adv_plot_data["Metric"], y=adv_plot_data["Strategy"], name=strategy_name, marker_color=QFA_COLORS["navy"]))
    fig_adv.add_trace(go.Bar(x=adv_plot_data["Metric"], y=adv_plot_data["Benchmark"], name=BENCHMARK_NAME, marker_color=QFA_COLORS["gray"]))
    fig_adv.update_yaxes(title="Ratio Value")
    fig_adv = layout(fig_adv, "Advanced Performance Ratios Comparison", 500)
    
    # NEW: Cornish-Fisher vs Historical VaR comparison
    cf_plot_data = pd.DataFrame({
        "VaR Type": ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"],
        "Historical": [historical_var(p, 0.95), historical_cvar(p, 0.95), historical_var(p, 0.99), historical_cvar(p, 0.99)],
        "Cornish-Fisher": [cornish_fisher_var(p, 0.95), cornish_fisher_cvar(p, 0.95), cornish_fisher_var(p, 0.99), cornish_fisher_cvar(p, 0.99)]
    })
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(x=cf_plot_data["VaR Type"], y=cf_plot_data["Historical"], name="Historical", marker_color=QFA_COLORS["steel"]))
    fig_cf.add_trace(go.Bar(x=cf_plot_data["VaR Type"], y=cf_plot_data["Cornish-Fisher"], name="Cornish-Fisher", marker_color=QFA_COLORS["muted_gold"]))
    fig_cf.update_yaxes(title="Risk Level", tickformat=".1%")
    fig_cf = layout(fig_cf, "Historical vs Cornish-Fisher Risk Metrics", 500)
    
    # Key metrics snapshot table
    metrics_df = pd.DataFrame([{
        "Strategy": strategy_name,
        "Annual Return": f"{metrics_row['Annual Return']:.2%}",
        "Annual Volatility": f"{metrics_row['Annual Volatility']:.2%}",
        "Sharpe": f"{metrics_row['Sharpe']:.2f}",
        "Max Drawdown": f"{metrics_row['Max Drawdown']:.2%}",
        "VaR 95% Daily": f"{metrics_row['VaR 95% Daily']:.2%}",
        "CVaR 95% Daily": f"{metrics_row['CVaR 95% Daily']:.2%}",
        "Tracking Error": f"{metrics_row['Tracking Error vs Benchmark']:.2%}",
        "Beta": f"{metrics_row['Beta vs Benchmark']:.2f}",
        "Information Ratio": f"{metrics_row['Information Ratio']:.2f}",
        "Gain-Loss Ratio": f"{gain_loss_ratio(p):.2f}",
        "Martin Ratio": f"{martin_ratio(p, rf):.2f}",
        "Kappa 3": f"{kappa_3_ratio(p):.2f}",
        "Modified Sharpe (CF)": f"{modified_sharpe_cf(p, rf, 0.95):.2f}",
    }])
    
    # Monthly and annual tables
    monthly_tbl = monthly_returns_table_qfa(p).copy()
    monthly_html = monthly_tbl.applymap(lambda v: "" if pd.isna(v) else f"{v:.2%}").to_html(
        border=0, classes="qfa-table", escape=False
    ) if not monthly_tbl.empty else "<p>No monthly return table available.</p>"
    
    annual_tbl = annual_returns_table_qfa(p)
    annual_html = html_table_formatted(annual_tbl, pct_cols=["Annual Return"]) if not annual_tbl.empty else "<p>No annual return table available.</p>"
    
    dd_tbl = drawdown_details_qfa(p)
    dd_html = html_table_formatted(dd_tbl, pct_cols=["Max Drawdown"]) if not dd_tbl.empty else "<p>No drawdown episode table available.</p>"
    
    audit_html = html_table_formatted(report_content_audit_table_qfa())
    
    # Capture ratios pie chart for interpretation
    capture_pie_data = pd.DataFrame({
        "Category": ["Up Capture", "Down Capture"],
        "Ratio": [capture.get("Up Capture Ratio", 0), capture.get("Down Capture Ratio", 0)]
    })
    fig_capture = go.Figure(data=[go.Pie(
        labels=capture_pie_data["Category"], 
        values=capture_pie_data["Ratio"],
        marker_colors=[QFA_COLORS["green"], QFA_COLORS["risk_red"]],
        hole=0.4,
        textinfo="label+percent"
    )])
    fig_capture = layout(fig_capture, "Capture Ratios Distribution", 450)

    # ============================================================
    # 6. HTML Document Assembly
    # ============================================================
    
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>QFA Institutional Tearsheet — {html.escape(strategy_name)}</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            body {{
                font-family: DejaVu Sans, Segoe UI, Arial, sans-serif;
                background:#f5f7fb;
                color:#111827;
                margin:0;
            }}
            header {{
                background:#0f172a;
                color:white;
                padding:28px 36px;
                border-bottom:5px solid #b08900;
            }}
            main {{ padding:28px 36px; }}
            h1 {{ margin:0; font-size:30px; }}
            h2 {{
                margin-top:30px;
                color:#0f172a;
                border-bottom:2px solid #e5e7eb;
                padding-bottom:8px;
            }}
            h3 {{
                margin-top:20px;
                color:#334155;
                font-size:18px;
            }}
            .note {{
                background:#fff7ed;
                border-left:5px solid #92400e;
                padding:14px 16px;
                border-radius:8px;
                color:#78350f;
                margin:16px 0;
                line-height:1.45;
            }}
            .verified {{
                background:#ecfdf5;
                border-left:5px solid #166534;
                padding:12px 16px;
                border-radius:8px;
                color:#064e3b;
                margin:16px 0;
                font-weight:700;
            }}
            .qfa-table {{
                width:100%;
                border-collapse:collapse;
                background:white;
                margin:18px 0;
                border-radius:12px;
                overflow:hidden;
                table-layout:fixed;
                box-shadow:0 8px 18px rgba(15,23,42,.06);
            }}
            .qfa-table th {{
                background:#111827;
                color:white;
                text-align:left;
                padding:9px;
                font-size:12px;
                white-space:normal;
                word-break:break-word;
            }}
            .qfa-table td {{
                border-bottom:1px solid #e5e7eb;
                padding:8px;
                font-size:12px;
                white-space:normal;
                word-break:break-word;
                vertical-align:top;
            }}
            .two-col {{
                display:grid;
                grid-template-columns: 1fr 1fr;
                gap:18px;
            }}
            .three-col {{
                display:grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap:18px;
            }}
            @media (max-width: 1000px) {{
                .two-col {{ grid-template-columns: 1fr; }}
                .three-col {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>QFA Institutional Tearsheet — {html.escape(strategy_name)}</h1>
            <p>Benchmark: S&amp;P 500 (^GSPC) | Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        <main>
            <div class="verified">
                v4.13 ENHANCED — This report includes QuantStats-style ratios, Advanced Risk Metrics (Gain-Loss, Martin, Kappa 3, Stutzer), 
                Cornish-Fisher VaR/CVaR, Capture Ratios, Appraisal Ratio, and comprehensive chart suite.
            </div>
            <div class="note">
                This report is generated from the selected strategy's validated daily return stream (Yahoo Finance only, no synthetic data).
                Benchmark-relative metrics are calculated against S&amp;P 500 (^GSPC).
            </div>

            <!-- Report Content Audit -->
            <h2>Report Content Audit</h2>
            {audit_html}

            <!-- QuantStats-Style Summary Metrics -->
            <h2>QuantStats-Style Summary Metrics</h2>
            {_html_table(qfa_qs_display)}

            <!-- Key Metrics Snapshot -->
            <h2>Key Metrics Snapshot</h2>
            {_html_table(metrics_df)}

            <!-- Advanced Performance Ratios -->
            <h2>Advanced Performance Ratios</h2>
            <div class="two-col">
                <div>
                    <h3>Advanced Ratios</h3>
                    {_html_table(advanced_ratios)}
                </div>
                <div>
                    <h3>Cornish-Fisher Risk Metrics</h3>
                    {_html_table(cf_metrics)}
                </div>
            </div>

            <!-- Benchmark-Relative Metrics -->
            <h2>Benchmark-Relative Metrics</h2>
            <div class="two-col">
                <div>
                    {_html_table(benchmark_relative)}
                </div>
                <div>
                    {_fig_to_html(fig_capture)}
                </div>
            </div>

            <!-- Performance Charts -->
            <h2>Performance Overview</h2>
            {_fig_to_html(fig_eq)}

            <h2>Drawdown Analytics</h2>
            <div class="two-col">
                <div>{_fig_to_html(fig_dd)}</div>
                <div>
                    <h3>Top Drawdown Episodes</h3>
                    {dd_html}
                </div>
            </div>

            <!-- Rolling Risk Metrics -->
            <h2>Rolling Risk / Return Metrics</h2>
            {_fig_to_html(fig_roll)}

            <!-- Benchmark Relative Risk -->
            <h2>Benchmark Relative Risk</h2>
            {_fig_to_html(fig_risk)}

            <!-- Tail Risk -->
            <h2>Tail Risk — VaR / CVaR</h2>
            <div class="two-col">
                <div>{_fig_to_html(fig_tail)}</div>
                <div>{_fig_to_html(fig_cf)}</div>
            </div>

            <!-- Return Distribution -->
            <h2>Return Distribution</h2>
            <div class="two-col">
                <div>{_fig_to_html(fig_dist)}</div>
                <div>{_fig_to_html(fig_adv)}</div>
            </div>

            <!-- Monthly and Annual Returns -->
            <h2>Monthly and Annual Returns</h2>
            <div class="two-col">
                <div>{_fig_to_html(fig_month)}</div>
                <div>{_fig_to_html(fig_annual)}</div>
            </div>

            <h3>Monthly Returns Table</h3>
            {monthly_html}

            <h3>Annual Returns Table</h3>
            {annual_html}

            <!-- Footer with interpretation guide -->
            <div class="note" style="margin-top:30px;">
                <strong>Interpretation Guide:</strong><br>
                • <strong>Gain-Loss Ratio &gt; 1.5</strong> indicates favorable return asymmetry.<br>
                • <strong>Martin Ratio / Pain Ratio</strong> penalize drawdown severity (higher is better).<br>
                • <strong>Kappa 3</strong> focuses on large loss avoidance (higher is better).<br>
                • <strong>Stutzer Index</strong> penalizes catastrophic losses.<br>
                • <strong>Cornish-Fisher VaR</strong> adjusts for skewness/kurtosis - compare with historical VaR.<br>
                • <strong>Capture Ratio &gt; 1</strong> means portfolio outperforms benchmark in both up/down markets.<br>
                • <strong>Appraisal Ratio</strong> measures active return per unit of residual risk (higher = better active management).<br>
                • <strong>Modified Sharpe (CF)</strong> uses Cornish-Fisher VaR as risk measure.
            </div>
        </main>
    </body>
    </html>
    """
    return html_doc.encode("utf-8")


def quantstats_html(strategy_name: str, r: pd.Series, rf: float) -> Tuple[Optional[bytes], str]:
    """
    Robust QuantStats report generator.

    QuantStats is preserved. The strategy return stream is cleaned before report
    generation. Any technical status is handled internally so board-facing UI stays clean.
    """
    if not QS_AVAILABLE:
        return None, "QuantStats package not available."

    safe_name = strategy_name.lower().replace(" ", "_").replace("/", "_")
    path = f"qfa_quantstats_{safe_name}.html"

    try:
        clean = r.copy()
        clean.index = pd.to_datetime(clean.index)
        clean = pd.to_numeric(clean, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean.sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]
        clean.name = strategy_name
        try:
            clean.index = clean.index.tz_localize(None)
        except Exception:
            pass

        if len(clean) < MIN_OBSERVATIONS:
            return None, "Insufficient observations."

        try:
            qs.reports.html(
                clean,
                benchmark=None,
                rf=rf,
                output=path,
                title=f"QFA Prime QuantStats — {strategy_name}",
                download_filename=path,
            )
        except TypeError:
            qs.reports.html(
                clean,
                benchmark=None,
                rf=rf,
                output=path,
                title=f"QFA Prime QuantStats — {strategy_name}",
            )

        with open(path, "rb") as f:
            data = f.read()

        try:
            os.remove(path)
        except Exception:
            pass

        if not data or len(data) < 1000:
            return None, "Empty QuantStats report."

        return data, "OK"

    except Exception as exc:
        return None, str(exc)


# ============================================================
# UI
# ============================================================

def main():
    global COMMODITY_UNIVERSE
    st.markdown(
        f"""
        <div class="qfa-hero">
            <h1>QFA Prime Finance Platform</h1>
            <p>Commodity Instrument Class — Institutional Interactive Strategy Lab</p>
            <p>Alpha Engine + Advanced Risk Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MISSING_PACKAGES:
        st.error("Missing packages: " + ", ".join(MISSING_PACKAGES))
        st.stop()

    with st.sidebar:
        st.header("Portfolio Gate")

        if st.button("Clear cache"):
            clear_cache()

        universe_mode = st.selectbox(
            "Data Universe",
            options=list(UNIVERSE_MODES.keys()),
            index=0,
        )
        active_universe = UNIVERSE_MODES[universe_mode]
        COMMODITY_UNIVERSE = active_universe

        if "ETF Proxies" in universe_mode:
            st.markdown(
                "<div class='qfa-transparency-badge'>ETF PROXY MODE ACTIVE</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Show proxy mapping", expanded=True):
                st.dataframe(PROXY_TRANSPARENCY_TABLE, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                "<div class='qfa-transparency-badge'>FUTURES MODE ACTIVE</div>",
                unsafe_allow_html=True,
            )

        selected_display = st.multiselect(
            "Commodity instruments",
            options=[v["display"] for v in active_universe.values()],
            default=[v["display"] for v in active_universe.values()],
        )
        display_to_ticker = {v["display"]: k for k, v in active_universe.items()}
        selected_tickers = [display_to_ticker[d] for d in selected_display]

        start_date = st.date_input("Start date", DEFAULT_START, min_value=MIN_START_DATE, max_value=DEFAULT_END)
        end_date = st.date_input("End date", DEFAULT_END, min_value=MIN_START_DATE, max_value=DEFAULT_END)

        rf = st.number_input("Risk-free rate", 0.0, 0.30, DEFAULT_RF, 0.005, format="%.3f")
        max_weight = st.slider("Optimization max single weight", 0.20, 1.00, 0.45, 0.05)

        st.divider()
        st.subheader("Data Discipline")
        min_valid_ratio = st.slider("Minimum valid data ratio", 0.50, 0.95, DEFAULT_MIN_VALID_RATIO, 0.05)
        ffill_limit = st.slider("Forward-fill limit", 0, 7, DEFAULT_FFILL_LIMIT, 1)

        st.divider()
        st.subheader("Custom Portfolio Weights")
        custom_raw = {}
        for ticker in selected_tickers:
            label = COMMODITY_UNIVERSE[ticker]["display"]
            custom_raw[ticker] = st.number_input(f"{label} weight", 0.0, 100.0, 100.0 / max(len(selected_tickers), 1), 1.0)
        raw_sum = sum(custom_raw.values())
        custom_weights = {k: (v / raw_sum if raw_sum > 0 else 0) for k, v in custom_raw.items()}

        st.divider()
        st.subheader("Risk Controls")
        rolling_window = st.slider("Rolling window", 21, 252, 63, 21)
        te_target = st.number_input("Tracking Error target", 0.0, 0.50, 0.06, 0.01)
        te_band = st.number_input("Tracking Error band", 0.0, 0.20, 0.02, 0.01)

        st.divider()
        st.subheader("Advanced Spread / Bollinger")
        boll_window = st.slider("Bollinger window", 21, 252, 63, 21)
        boll_std = st.slider("Bollinger standard deviations", 1.0, 3.0, 2.0, 0.25)

        st.divider()
        st.subheader("Alpha Engine Controls")
        alpha_mom_short = st.slider("Momentum short lookback", 21, 126, 63, 21)
        alpha_mom_long = st.slider("Momentum long lookback", 63, 252, 126, 21)
        alpha_meanrev_window = st.slider("Mean-reversion z-score window", 21, 252, 63, 21)
        alpha_vol_window = st.slider("Volatility quality window", 21, 252, 63, 21)
        alpha_ic_horizon = st.slider("IC forward horizon", 5, 63, 21, 7)

        st.divider()
        st.subheader("GARCH Controls")
        run_garch = st.checkbox("Run GARCH Lab", value=False)
        garch_mode = st.selectbox("GARCH model set", list(GARCH_MODEL_OPTIONS.keys()), index=0)

    if len(selected_tickers) < 2:
        st.error("Select at least two commodity instruments.")
        return
    if start_date < MIN_START_DATE:
        st.error("Start date cannot be earlier than 2018-01-01.")
        return
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    try:
        with st.spinner("Downloading Yahoo Finance data..."):
            raw_prices = download_yahoo_prices(tuple(selected_tickers), str(start_date), str(end_date))
            prices = prepare_price_matrix(raw_prices, min_valid_ratio, ffill_limit, allow_single=False)
            returns = prepare_returns(prices)

            raw_benchmark = download_yahoo_prices((BENCHMARK_TICKER,), str(start_date), str(end_date))
            benchmark_prices = prepare_price_matrix(raw_benchmark, min_valid_ratio, ffill_limit, allow_single=True)
            benchmark_returns = prepare_returns(benchmark_prices).iloc[:, 0].rename(BENCHMARK_TICKER)

            common = returns.index.intersection(benchmark_returns.index)
            returns = returns.loc[common]
            prices = prices.loc[common]
            benchmark_returns = benchmark_returns.loc[common]

    except Exception as exc:
        st.error(f"Data error: {exc}")
        st.info("Try ETF Proxy mode, fewer instruments, longer date range, or clear cache.")
        return

    strategy_returns, strategy_weights, weights_df, opt_perf = build_strategy_set(prices, returns, rf, max_weight, custom_weights)

    alpha_engine = build_alpha_engine(
        prices=prices,
        returns=returns,
        rf=rf,
        max_weight=max_weight,
        momentum_short=alpha_mom_short,
        momentum_long=alpha_mom_long,
        meanrev_window=alpha_meanrev_window,
        vol_window=alpha_vol_window,
        ic_horizon=alpha_ic_horizon,
    )
    strategy_returns["Alpha Composite"] = alpha_engine["returns"]
    strategy_weights["Alpha Composite"] = alpha_engine["weights"].iloc[-1].to_dict() if not alpha_engine["weights"].empty else {}

    if "Alpha Composite" not in weights_df.columns:
        weights_df["Alpha Composite"] = pd.Series(strategy_weights["Alpha Composite"]).reindex(weights_df["Ticker"]).fillna(0.0).values

    # Align strategies to benchmark
    for name in list(strategy_returns.keys()):
        p, b = align_two(strategy_returns[name], benchmark_returns, name, "Benchmark")
        strategy_returns[name] = p
    benchmark_returns = benchmark_returns.reindex(next(iter(strategy_returns.values())).index).dropna()

    all_strategy_names = list(strategy_returns.keys())
    selected_strategies = st.sidebar.multiselect(
        "Strategies shown in charts",
        options=all_strategy_names,
        default=all_strategy_names,
    )
    if not selected_strategies:
        selected_strategies = all_strategy_names

    strategy_metrics = metrics_for_returns({k: strategy_returns[k] for k in selected_strategies}, benchmark_returns, rf)
    all_strategy_metrics = metrics_for_returns(strategy_returns, benchmark_returns, rf)
    interpretation = interpret_strategy_table(all_strategy_metrics)

    instrument_map = {f"{COMMODITY_UNIVERSE[t]['display']} ({t})": returns[t] for t in returns.columns}
    instrument_metrics = metrics_for_returns(instrument_map, benchmark_returns, rf)
    quality = data_quality_report(raw_prices, prices, returns)

    primary = selected_strategies[0]
    primary_row = all_strategy_metrics[all_strategy_metrics["Strategy / Instrument"] == primary].iloc[0]

    tabs = st.tabs([
        "Executive Dashboard",
        "Portfolio Weights",
        "Optimization",
        "Performance Metrics",
        "Risk Metrics",
        "Rolling Beta",
        "Tracking Error",
        "VaR / CVaR",
        "Drawdown",
        "Correlation",
        "Log Return Differences",
        "Rolling Sharpe",
        "Advanced Risk & Performance",
        "GARCH Volatility Lab",
        "QFA Tearsheet Reports | Select Your Strategy First!",
        "Portfolio Strategy Notes",
        "Alpha Generation",
        "Alpha Engine LIVE",
        "Alpha IC Validation",
        "Alpha Portfolio",
        "Info Hub",
        "Data Quality",
        "Export Center",
    ])

    with tabs[0]:
        st.subheader("Advanced Institutional KPI Layout")
        mode_label = "ETF Proxy" if "ETF Proxies" in universe_mode else "Futures"
        st.markdown(
            f"""
            <div class="qfa-note">
            Primary strategy: <b>{primary}</b>. Active data mode: <b>{mode_label}</b>.
            Sidebar parameters remain active: custom weights, RF rate, max weight constraint, rolling window, TE bands, GARCH model set and Bollinger settings.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='qfa-kpi-band'><div class='qfa-kpi-band-title'>Portfolio Risk / Return Command Center</div>", unsafe_allow_html=True)
        c = st.columns(8)
        with c[0]: kpi_card("Annual Return", fmt_pct(primary_row["Annual Return"]), primary, tone_for_return(primary_row["Annual Return"]))
        with c[1]: kpi_card("Annual Volatility", fmt_pct(primary_row["Annual Volatility"]), "risk level", "neutral")
        with c[2]: kpi_card("Sharpe", fmt_num(primary_row["Sharpe"]), f"RF {rf:.2%}", tone_for_sharpe(primary_row["Sharpe"]))
        with c[3]: kpi_card("Max Drawdown", fmt_pct(primary_row["Max Drawdown"]), "peak-to-trough", tone_for_drawdown(primary_row["Max Drawdown"]))
        with c[4]: kpi_card("VaR 95%", fmt_pct(primary_row["VaR 95% Daily"]), "daily", "warn")
        with c[5]: kpi_card("CVaR 95%", fmt_pct(primary_row["CVaR 95% Daily"]), "tail loss", "bad")
        with c[6]: kpi_card("Tracking Error", fmt_pct(primary_row["Tracking Error vs Benchmark"]), "vs ^GSPC", tone_for_te(primary_row["Tracking Error vs Benchmark"]))
        with c[7]: kpi_card("Beta", fmt_num(primary_row["Beta vs Benchmark"]), "vs ^GSPC", "neutral")

        st.markdown("</div>", unsafe_allow_html=True)

        c2 = st.columns(5)
        with c2[0]: kpi_card("Data Mode", mode_label, "transparent source", "neutral")
        with c2[1]: kpi_card("Instruments", str(len(selected_tickers)), "selected", "neutral")
        with c2[2]: kpi_card("Obs.", str(len(returns)), "daily aligned", "neutral")
        with c2[3]: kpi_card("Max Weight", fmt_pct(max_weight), "optimization cap", "neutral")
        with c2[4]: kpi_card("Rolling Window", str(rolling_window), "trading days", "neutral")

        safe_chart(fig_strategy_cumulative(strategy_returns, benchmark_returns, selected_strategies))
        safe_chart(fig_risk_return(strategy_metrics))
        safe_df(interpretation)

    with tabs[1]:
        st.subheader("Portfolio Weights")
        st.markdown("<div class='qfa-note'>Weights are recalculated from sidebar inputs and optimization settings.</div>", unsafe_allow_html=True)
        safe_chart(fig_weights(weights_df, selected_strategies))
        safe_df(weights_df)

    with tabs[2]:
        st.subheader("Optimization")
        st.markdown("<div class='qfa-note'>PyPortfolioOpt computes Max Sharpe and Min Volatility with the selected max-weight constraint.</div>", unsafe_allow_html=True)
        safe_chart(fig_weights(weights_df, ["Max Sharpe", "Min Volatility"] if "Max Sharpe" in weights_df.columns else selected_strategies))
        safe_chart(fig_risk_return(all_strategy_metrics))
        safe_df(opt_perf)

    with tabs[3]:
        st.subheader("Performance Metrics")
        safe_chart(fig_performance_bars(strategy_metrics))
        safe_chart(fig_strategy_cumulative(strategy_returns, benchmark_returns, selected_strategies))
        safe_df(strategy_metrics)

    with tabs[4]:
        st.subheader("Risk Metrics")
        safe_chart(fig_te_beta_map(strategy_metrics))
        safe_chart(fig_tail_bars(strategy_metrics))
        safe_df(strategy_metrics)
        st.subheader("User-Friendly Metric Dictionary")
        safe_df(metric_dictionary())

    with tabs[5]:
        st.subheader("Rolling Beta")
        beta_fig = make_subplots(rows=1, cols=1)
        for name in selected_strategies:
            beta = rolling_beta(strategy_returns[name], benchmark_returns, rolling_window)
            beta_fig.add_trace(go.Scatter(x=beta.index, y=beta.values, mode="lines", name=name))
        beta_fig.add_hline(y=1.0, line_dash="dash", annotation_text="Beta = 1")
        safe_chart(layout(beta_fig, f"{rolling_window}-Day Rolling Beta vs {BENCHMARK_NAME}", 720))

    with tabs[6]:
        st.subheader("Tracking Error")
        te_fig = make_subplots(rows=1, cols=1)
        for name in selected_strategies:
            te = rolling_tracking_error(strategy_returns[name], benchmark_returns, rolling_window)
            te_fig.add_trace(go.Scatter(x=te.index, y=te.values, mode="lines", name=name))
        te_fig.add_hline(y=te_target, line_dash="dash", annotation_text="TE Target")
        te_fig.add_hline(y=te_target + te_band, line_dash="dot", annotation_text="Upper Band")
        te_fig.add_hline(y=max(te_target - te_band, 0), line_dash="dot", annotation_text="Lower Band")
        te_fig.update_yaxes(tickformat=".0%")
        safe_chart(layout(te_fig, f"{rolling_window}-Day Rolling Tracking Error", 720))

    with tabs[7]:
        st.subheader("VaR / CVaR")
        safe_chart(fig_var_cvar(strategy_returns, selected_strategies, rolling_window))
        safe_chart(fig_tail_bars(strategy_metrics))

    with tabs[8]:
        st.subheader("Drawdown")
        safe_chart(fig_drawdowns(strategy_returns, benchmark_returns, selected_strategies))
        safe_chart(fig_underwater_recovery(strategy_returns, selected_strategies))

    with tabs[9]:
        st.subheader("Correlation")
        safe_chart(fig_corr(returns))
        st.subheader("Instrument Metrics")
        safe_df(instrument_metrics)

    with tabs[10]:
        st.subheader("ETF / Instrument Log Return Differences and Bollinger Bands")
        st.markdown(
            "<div class='qfa-note'>This tab shows log-return differences between selected instruments. In ETF Proxy mode, this makes proxy behavior transparent. Bollinger bands help identify unusual relative-return deviations.</div>",
            unsafe_allow_html=True,
        )
        lr = log_returns(prices)
        base_ticker = st.selectbox(
            "Base ticker for log-return differences",
            options=list(lr.columns),
            index=0,
            format_func=lambda t: COMMODITY_UNIVERSE.get(t, {}).get("display", t),
        )
        diff = return_difference_series(lr, base_ticker)
        safe_chart(fig_log_return_difference(lr, base_ticker))
        if not diff.empty:
            spread_choice = st.selectbox("Spread for Bollinger analysis", options=list(diff.columns), index=0)
            safe_chart(fig_log_return_difference_bollinger(diff, spread_choice, boll_window, boll_std))
            safe_df(diff.tail(250).reset_index().rename(columns={"index": "Date"}))
        else:
            st.info("No spread available. Select at least two instruments.")

    with tabs[11]:
        st.subheader("Rolling Sharpe")
        safe_chart(fig_rolling_sharpe(strategy_returns, selected_strategies, rf, rolling_window))
        st.markdown("<div class='qfa-note'>Rolling Sharpe helps users understand whether risk-adjusted performance is stable or only period-specific.</div>", unsafe_allow_html=True)

    # ============================================================
    # NEW TAB: Advanced Risk & Performance Analytics
    # ============================================================
    with tabs[12]:
        st.subheader("Advanced Risk & Performance Analytics")
        st.markdown(
            "<div class='qfa-note'>This tab provides institutional-grade advanced risk metrics: "
            "Cornish-Fisher VaR/CVaR, Gain-Loss Ratio, Martin Ratio, Pain Index, Kappa 3, Stutzer Index, "
            "Up/Down Capture Ratios, Appraisal Ratio, Kalman Filter Beta, and Regime-Conditional Metrics.</div>",
            unsafe_allow_html=True,
        )
        
        # Seçilen strateji için gelişmiş metrikler
        adv_strategy = st.selectbox(
            "Select strategy for advanced analysis",
            options=selected_strategies,
            index=0,
            key="adv_strategy_select"
        )
        
        if adv_strategy in strategy_returns:
            adv_r = strategy_returns[adv_strategy]
            adv_p, adv_b = align_two(adv_r, benchmark_returns, adv_strategy, "Benchmark")
            
            # Yeni rasyoları hesapla
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Risk-Adjusted Performance")
                adv_metrics = pd.DataFrame([
                    {"Metric": "Gain-Loss Ratio", "Value": f"{gain_loss_ratio(adv_p):.3f}"},
                    {"Metric": "Martin Ratio", "Value": f"{martin_ratio(adv_p, rf):.3f}"},
                    {"Metric": "Pain Ratio", "Value": f"{pain_ratio(adv_p, rf):.3f}"},
                    {"Metric": "Kappa 3", "Value": f"{kappa_3_ratio(adv_p):.3f}"},
                    {"Metric": "Stutzer Index", "Value": f"{stutzer_index(adv_p):.3f}"},
                ])
                safe_df(adv_metrics)
            
            with col2:
                st.markdown("### Cornish-Fisher Risk Metrics")
                cf_metrics = pd.DataFrame([
                    {"Metric": "Historical VaR 95%", "Value": f"{historical_var(adv_p, 0.95):.2%}"},
                    {"Metric": "Cornish-Fisher VaR 95%", "Value": f"{cornish_fisher_var(adv_p, 0.95):.2%}"},
                    {"Metric": "Historical CVaR 95%", "Value": f"{historical_cvar(adv_p, 0.95):.2%}"},
                    {"Metric": "Cornish-Fisher CVaR 95%", "Value": f"{cornish_fisher_cvar(adv_p, 0.95):.2%}"},
                    {"Metric": "Modified Sharpe (CF-VaR)", "Value": f"{modified_sharpe_cf(adv_p, rf, 0.95):.3f}"},
                ])
                safe_df(cf_metrics)
            
            with col3:
                st.markdown("### Benchmark-Relative")
                capture = capture_ratios(adv_p, adv_b)
                capture_df = pd.DataFrame([
                    {"Metric": "Up Capture Ratio", "Value": f"{capture.get('Up Capture Ratio', np.nan):.3f}"},
                    {"Metric": "Down Capture Ratio", "Value": f"{capture.get('Down Capture Ratio', np.nan):.3f}"},
                    {"Metric": "Capture Ratio (Up/Down)", "Value": f"{capture.get('Capture Ratio (Up/Down)', np.nan):.3f}"},
                    {"Metric": "Appraisal Ratio", "Value": f"{appraisal_ratio(adv_p, adv_b, rf):.3f}"},
                ])
                safe_df(capture_df)
            
            st.divider()
            
            # Kalman Beta vs Rolling Beta karşılaştırması
            st.subheader("Kalman Filter Beta vs Rolling Beta")
            col_k1, col_k2 = st.columns(2)
            
            with col_k1:
                beta_dict = rolling_beta_enhanced(adv_r, benchmark_returns, rolling_window)
                fig_beta_compare = go.Figure()
                fig_beta_compare.add_trace(go.Scatter(
                    x=beta_dict["Rolling Beta"].index, 
                    y=beta_dict["Rolling Beta"].values, 
                    mode="lines", 
                    name=f"Rolling Beta ({rolling_window}d)",
                    line=dict(color=QFA_COLORS["steel"], width=2)
                ))
                fig_beta_compare.add_trace(go.Scatter(
                    x=beta_dict["Kalman Beta"].index, 
                    y=beta_dict["Kalman Beta"].values, 
                    mode="lines", 
                    name="Kalman Filter Beta",
                    line=dict(color=QFA_COLORS["muted_gold"], width=2.2)
                ))
                fig_beta_compare.add_hline(y=1.0, line_dash="dash", annotation_text="Beta = 1")
                fig_beta_compare.update_yaxes(title="Beta")
                safe_chart(layout(fig_beta_compare, f"Beta Estimation Methods Comparison — {adv_strategy}", 550))
            
            with col_k2:
                # Beta farkı istatistikleri
                common_idx = beta_dict["Rolling Beta"].index.intersection(beta_dict["Kalman Beta"].index)
                if len(common_idx) > 0:
                    diff = beta_dict["Kalman Beta"].loc[common_idx] - beta_dict["Rolling Beta"].loc[common_idx]
                    diff_fig = go.Figure()
                    diff_fig.add_trace(go.Histogram(x=diff.values, nbinsx=30, marker_color=QFA_COLORS["navy"]))
                    diff_fig.update_xaxes(title="Kalman Beta - Rolling Beta")
                    diff_fig.update_yaxes(title="Frequency")
                    safe_chart(layout(diff_fig, "Beta Estimation Difference Distribution", 550))
            
            st.divider()
            
            # Regime-Conditional Metrics
            st.subheader("Volatility Regime Analysis")
            regime_df = regime_conditional_metrics(adv_r, benchmark_returns)
            safe_df(regime_df)
            
            # GARCH VaR (if available)
            if run_garch and 'garch_store' in locals() and garch_store:
                st.subheader("GARCH-Conditional VaR (from fitted models)")
                garch_var_df = []
                for ticker in returns.columns:
                    gvar = garch_conditional_var(returns[ticker], garch_store, ticker, "garch_t", 0.95)
                    garch_var_df.append({
                        "Instrument": COMMODITY_UNIVERSE.get(ticker, {}).get("display", ticker), 
                        "GARCH VaR 95%": f"{gvar:.2%}" if not pd.isna(gvar) else "N/A"
                    })
                safe_df(pd.DataFrame(garch_var_df))
            
            st.caption("""
            **Interpretation Guide:**
            - **Gain-Loss Ratio > 1.5** indicates favorable asymmetry.
            - **Martin Ratio / Pain Ratio** penalize drawdown severity (higher is better).
            - **Kappa 3** focuses on large loss avoidance (higher is better).
            - **Stutzer Index** penalizes catastrophic losses.
            - **Cornish-Fisher VaR** adjusts for skewness/kurtosis - compare with historical VaR.
            - **Capture Ratio > 1** means portfolio outperforms benchmark in both up/down markets.
            - **Kalman Beta** is smoother and more adaptive than rolling window beta.
            """)

    with tabs[13]:
        st.subheader("GARCH Volatility Lab")
        st.markdown("<div class='qfa-note'>GARCH is fitted to individual commodities, not portfolio strategies.</div>", unsafe_allow_html=True)
        if run_garch:
            with st.spinner("Running GARCH models..."):
                garch_table, best_garch, garch_store_local = run_garch_suite(returns, tuple(GARCH_MODEL_OPTIONS[garch_mode]))
                # Store for later use
                garch_store = garch_store_local
            safe_df(best_garch)
            safe_chart(fig_garch_best(best_garch, garch_store))
            safe_chart(fig_garch_rank(garch_table))

            if not best_garch.empty:
                choice = st.selectbox(
                    "Instrument diagnostics",
                    options=list(best_garch["Ticker"]),
                    format_func=lambda t: COMMODITY_UNIVERSE.get(t, {}).get("display", t),
                )
                best_key = best_garch[best_garch["Ticker"] == choice].iloc[0]["Best Model Key"]
                safe_chart(fig_garch_resid(choice, best_key, garch_store))
                with st.expander("ARCH summary"):
                    key = f"{choice}__{best_key}"
                    st.text(garch_store.get(key, {}).get("Summary", "Unavailable"))
            safe_df(garch_table)
        else:
            st.info("Enable 'Run GARCH Lab' in the sidebar.")
            garch_store = {}

    with tabs[14]:
        st.subheader("QFA Tearsheet Reports | Select Your Strategy First!")
        st.markdown(
            "<div class='qfa-note'>Select a strategy first. Reports are generated from the validated daily strategy return stream and benchmark-aligned against S&P 500 (^GSPC).</div>",
            unsafe_allow_html=True,
        )
        qs_strategy = st.selectbox("Report strategy", options=all_strategy_names, index=all_strategy_names.index(primary) if primary in all_strategy_names else 0)
        qs_bytes, qs_status = quantstats_html(qs_strategy, strategy_returns[qs_strategy], rf)

        selected_metric_row = all_strategy_metrics[all_strategy_metrics["Strategy / Instrument"] == qs_strategy].iloc[0]
        institutional_tearsheet_bytes = qfa_institutional_tearsheet_html(
            qs_strategy,
            strategy_returns[qs_strategy],
            benchmark_returns,
            rf,
            selected_metric_row,
        )

        c_report_1, c_report_2 = st.columns(2)

        safe_strategy_name = qs_strategy.lower().replace(" ", "_").replace("/", "_")

        with c_report_1:
            st.download_button(
                f"Download QFA Institutional Tearsheet — {qs_strategy}",
                data=institutional_tearsheet_bytes,
                file_name=f"qfa_institutional_tearsheet_v413_{safe_strategy_name}.html",
                mime="text/html",
            )

        with c_report_2:
            if qs_bytes:
                st.download_button(
                    f"Download QuantStats HTML — {qs_strategy}",
                    data=qs_bytes,
                    file_name=f"qfa_quantstats_v413_{safe_strategy_name}.html",
                    mime="text/html",
                )
            else:
                st.info("QFA Institutional Tearsheet is ready.")

        st.subheader("QuantStats-Style Summary Metrics — v4.13 Verified")
        qfa_qs_metrics = qfa_quantstats_style_metrics(qs_strategy, strategy_returns[qs_strategy], benchmark_returns, rf)
        safe_df(qfa_metrics_display_table(qfa_qs_metrics))

        with st.expander("Verify QFA Institutional Tearsheet content before download", expanded=False):
            st.markdown("The downloaded QFA Institutional Tearsheet contains the full table shown above under **QuantStats-Style Summary Metrics — v4.13 Verified**.")

        st.subheader("Strategy Report Metrics")
        safe_df(all_strategy_metrics[all_strategy_metrics["Strategy / Instrument"] == qs_strategy])

    with tabs[15]:
        st.subheader("Portfolio Strategy Notes")
        st.markdown(
            "<div class='qfa-note'>This table explains each portfolio strategy in a board-readable format: objective, weighting logic, strengths, weaknesses and governance interpretation.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='compact-notes'>", unsafe_allow_html=True)
        safe_df(portfolio_strategy_notes_table())
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[16]:
        st.subheader("Alpha Generation")
        st.markdown(
            "<div class='qfa-note'>Alpha generation must be governed as a research pipeline: economic rationale, signal design, validation, cost control, risk model, portfolio construction and live monitoring.</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Alpha Methods Library")
        st.markdown("<div class='compact-notes'>", unsafe_allow_html=True)
        safe_df(alpha_generation_methods_table())
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Alpha Research Pipeline")
        safe_df(alpha_research_pipeline_table())

        st.subheader("Practical Alpha Module Roadmap")
        safe_df(pd.DataFrame([
            {"Module": "Signal Lab", "Feature": "Momentum, mean reversion, volatility, carry and factor signals.", "Purpose": "Generate candidate alpha scores."},
            {"Module": "IC Monitor", "Feature": "Rolling IC / rank IC and signal decay curves.", "Purpose": "Validate predictive power."},
            {"Module": "Alpha-to-Weights Engine", "Feature": "Convert alpha scores into constrained active weights.", "Purpose": "Translate research into portfolio decisions."},
            {"Module": "Active Risk Budget", "Feature": "Tracking Error, beta and factor exposure limits.", "Purpose": "Prevent uncontrolled benchmark-relative risk."},
            {"Module": "Governance Panel", "Feature": "Signal health, drawdown, turnover and model confidence flags.", "Purpose": "Explain to management when the model should or should not be trusted."},
        ]))

    with tabs[17]:
        st.subheader("Alpha Engine LIVE")
        st.markdown(
            "<div class='qfa-note'>This tab calculates live alpha signals from the selected universe. Signals are converted into dynamic long-only weights with a one-day lag to reduce look-ahead bias.</div>",
            unsafe_allow_html=True,
        )
        safe_df(alpha_engine["snapshot"])
        safe_chart(fig_alpha_scores(alpha_engine["composite"]))
        safe_chart(fig_alpha_weights(alpha_engine["weights"]))

    with tabs[18]:
        st.subheader("Alpha IC Validation")
        st.markdown(
            "<div class='qfa-note'>Information Coefficient measures whether today’s signal ranks predict future return ranks. Positive and stable IC supports alpha credibility.</div>",
            unsafe_allow_html=True,
        )
        safe_df(alpha_engine["ic_table"])
        safe_chart(fig_ic_series(alpha_engine["signals"], returns, alpha_ic_horizon))
        
        # IC Decay Analysis
        st.subheader("IC Decay Analysis")
        st.markdown("<div class='qfa-note'>How does predictive power decay with forecast horizon?</div>", unsafe_allow_html=True)
        ic_decay_df = information_coefficient_decay(alpha_engine["signals"], returns, max_lag=10)
        if not ic_decay_df.empty:
            fig_ic_decay = go.Figure()
            for signal_name in ic_decay_df["Signal"].unique():
                subset = ic_decay_df[ic_decay_df["Signal"] == signal_name]
                fig_ic_decay.add_trace(go.Scatter(
                    x=subset["Lag (Days)"], 
                    y=subset["Mean IC"], 
                    mode="lines+markers", 
                    name=signal_name,
                    line=dict(width=2)
                ))
            fig_ic_decay.add_hline(y=0, line_dash="dash", annotation_text="IC = 0")
            fig_ic_decay.update_xaxes(title="Forecast Horizon (Days)")
            fig_ic_decay.update_yaxes(title="Mean Information Coefficient")
            safe_chart(layout(fig_ic_decay, "Information Coefficient Decay Curve", 600))
            safe_df(ic_decay_df)
        else:
            st.info("Not enough data for IC decay analysis. Try longer date range.")

    with tabs[19]:
        st.subheader("Alpha Portfolio")
        ew_ret = strategy_returns.get("Equal Weight")
        safe_chart(fig_alpha_cumulative(alpha_engine["returns"], benchmark_returns, ew_ret))
        safe_chart(fig_alpha_drawdown(alpha_engine["returns"], ew_ret))
        safe_chart(fig_alpha_turnover(alpha_engine["turnover"]))

        alpha_metrics = metrics_for_returns({"Alpha Composite": alpha_engine["returns"], "Equal Weight": ew_ret}, benchmark_returns, rf)
        safe_df(alpha_metrics)

        st.subheader("Alpha Governance Diagnostics")
        safe_df(alpha_governance_table(alpha_engine["returns"], benchmark_returns, alpha_engine["weights"], alpha_engine["ic_table"], rf))

    with tabs[20]:
        st.subheader("Info Hub")
        st.subheader("Proxy Transparency")
        safe_df(PROXY_TRANSPARENCY_TABLE)
        st.caption("ETF proxy mode is explicit and visible. It is not synthetic data and not hidden futures replacement.")

        info_df = pd.DataFrame([
            {"Ticker": k, "Display Name": v["display"], "Full Name": v["name"], "Instrument Class": v["class"], "Source": "Yahoo Finance"}
            for k, v in COMMODITY_UNIVERSE.items()
        ])
        safe_df(info_df)
        st.subheader("Strategy Definitions")
        safe_df(pd.DataFrame([
            {"Strategy": "Equal Weight", "Definition": "Same weight for each selected instrument."},
            {"Strategy": "User Custom Weights", "Definition": "Sidebar weights normalized to 100%."},
            {"Strategy": "Inverse Volatility", "Definition": "Lower-volatility instruments receive larger weights."},
            {"Strategy": "Max Sharpe", "Definition": "PyPortfolioOpt expected Sharpe-maximizing portfolio."},
            {"Strategy": "Min Volatility", "Definition": "PyPortfolioOpt minimum-variance portfolio."},
            {"Strategy": "Alpha Composite", "Definition": "Dynamic alpha portfolio generated from momentum, mean-reversion and volatility-quality signals."},
        ]))
        st.subheader("Deployment Diagnostics")
        safe_df(pd.DataFrame([
            {"Item": "Version", "Value": VERSION},
            {"Item": "Python", "Value": sys.version.split()[0]},
            {"Item": "Streamlit", "Value": getattr(st, "__version__", "unknown")},
            {"Item": "Synthetic Fallback", "Value": "Disabled"},
            {"Item": "Benchmark", "Value": f"{BENCHMARK_NAME} ({BENCHMARK_TICKER})"},
        ]))

    with tabs[21]:
        st.subheader("Data Quality")
        safe_df(quality)
        st.caption(f"Aligned prices: {prices.shape}; aligned returns: {returns.shape}; benchmark returns: {benchmark_returns.shape}")

    with tabs[22]:
        st.subheader("Export Center")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download strategy metrics CSV", df_csv(all_strategy_metrics), "qfa_strategy_metrics.csv", "text/csv")
            st.download_button("Download strategy weights CSV", df_csv(weights_df), "qfa_strategy_weights.csv", "text/csv")
        with col2:
            st.download_button("Download aligned prices CSV", prices.to_csv().encode("utf-8"), "qfa_aligned_prices.csv", "text/csv")
            st.download_button("Download aligned returns CSV", returns.to_csv().encode("utf-8"), "qfa_aligned_returns.csv", "text/csv")
        with col3:
            st.download_button("Download strategy returns CSV", returns_csv(strategy_returns), "qfa_strategy_returns.csv", "text/csv")
            st.download_button("Download data quality CSV", df_csv(quality), "qfa_data_quality.csv", "text/csv")
            st.download_button("Download alpha methods CSV", df_csv(alpha_generation_methods_table()), "qfa_alpha_methods.csv", "text/csv")
            st.download_button("Download alpha IC CSV", df_csv(alpha_engine["ic_table"]), "qfa_alpha_ic.csv", "text/csv")
            st.download_button("Download alpha snapshot CSV", df_csv(alpha_engine["snapshot"]), "qfa_alpha_snapshot.csv", "text/csv")
            st.download_button("Download QFA QuantStats-style metrics CSV", df_csv(qfa_quantstats_style_metrics(primary, strategy_returns[primary], benchmark_returns, rf)), "qfa_quantstats_style_metrics.csv", "text/csv")


if __name__ == "__main__":
    main()
