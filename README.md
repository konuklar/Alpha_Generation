# QFA Prime Finance Platform — Streamlit v4.12 QuantStats-Format Tearsheet

## Critical Fix

The QFA Institutional Tearsheet HTML has been rebuilt into a QuantStats-style report format.

The downloaded HTML includes:

- Report Content Audit
- QuantStats-style summary metrics table
- Key metrics snapshot
- Cumulative return vs S&P 500 (^GSPC)
- Drawdown chart
- Top drawdown episodes
- Rolling Sharpe / volatility / beta / tracking error
- Benchmark-relative risk
- Rolling VaR / CVaR
- Return distribution
- Monthly returns heatmap
- Monthly returns table
- Annual returns chart/table

The report filename includes `v412`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
