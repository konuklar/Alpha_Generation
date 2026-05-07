# QFA Prime Finance Platform — Streamlit v4.14 Institutional Build

This package uses the uploaded TMA MASTER DEEPSEEK v0003 code as the primary codebase, preserving the expanded institutional analytics and HTML report generation.

## Preserved / included

- Yahoo Finance only, no synthetic fallback
- ETF proxy / futures transparency
- Benchmark alignment with S&P 500 (^GSPC)
- QFA Institutional Tearsheet HTML generation
- QuantStats export
- Full QuantStats-style ratios and advanced metrics
- Advanced Risk & Performance Analytics
- Gain-Loss Ratio, Martin Ratio, Pain Index, Pain Ratio
- Kappa 3, Stutzer Index
- Cornish-Fisher VaR/CVaR and Modified Sharpe
- Up/Down Capture Ratios and Appraisal Ratio
- Kalman Filter Beta
- Regime-Conditional Metrics
- IC Decay Analysis
- Fixed dynamic max-weight cap logic

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
