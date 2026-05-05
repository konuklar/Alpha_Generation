# QFA Prime Finance Platform — Streamlit v4.10 Tearsheet Metrics Render Fix

## Critical Fix

v4.10 guarantees that the downloaded QFA Institutional Tearsheet HTML includes the full QFA QuantStats-style metrics table.

Included in HTML:

- Sharpe, Sortino, Calmar, Omega
- CAGR, cumulative return, annual volatility
- Max drawdown, longest DD days
- Beta, R², Treynor, Information Ratio vs ^GSPC
- VaR 95/99, CVaR 95/99
- Win rate, payoff ratio, profit factor
- Gain/Pain, tail ratio, common sense ratio
- Kelly Criterion
- Ulcer Index, Ulcer Performance Index

Benchmark-relative metrics use S&P 500 (`^GSPC`).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
