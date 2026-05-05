# QFA Prime Finance Platform — Streamlit v4.13 Advanced Risk Analytics Merge

Used uploaded improved v4.13 code as base.

## Preserved

- Yahoo Finance only; no synthetic fallback
- ETF proxy / futures transparency
- S&P 500 benchmark: `^GSPC`
- Portfolio strategies
- Alpha Engine
- QuantStats-format QFA Institutional Tearsheet

## Merged from uploaded improved code

- Advanced Risk & Performance Analytics tab
- Gain-Loss Ratio
- Martin Ratio
- Pain Index / Pain Ratio
- Kappa 3
- Stutzer Index
- Cornish-Fisher VaR / CVaR
- Modified Sharpe using Cornish-Fisher VaR
- Up / Down Capture Ratios
- Appraisal Ratio
- Kalman Filter Beta
- Regime-Conditional Metrics
- IC Decay Analysis

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
