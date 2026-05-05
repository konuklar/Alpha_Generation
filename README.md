# QFA Prime Finance Platform — Streamlit v4.9 QuantStats-Style QFA Metrics Engine

## New in v4.9

QuantStats is preserved, but QFA Institutional Tearsheet now includes an internal QuantStats-style metrics engine.

Added QFA-calculated metrics:

- Risk-Free Rate
- Time in Market
- Cumulative Return
- CAGR
- Sharpe
- Sortino
- Calmar
- Omega
- Max Drawdown
- Longest Drawdown Days
- Annual Volatility
- Downside Deviation
- R-Squared vs Benchmark
- Beta
- Treynor Ratio
- Information Ratio
- Skew / Kurtosis
- Expected Daily / Monthly / Yearly
- Kelly Criterion
- VaR 95 / 99
- CVaR 95 / 99
- Win Rate
- Consecutive Wins / Losses
- Gain/Pain Ratio
- Payoff Ratio
- Profit Factor
- Tail Ratio
- Common Sense Ratio
- Ulcer Index
- Ulcer Performance Index

Benchmark-relative metrics use S&P 500 (`^GSPC`).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
