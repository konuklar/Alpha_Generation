# QFA Prime Finance Platform — v4.15 Tracking Error Fix

Fixes Tracking Error reference lines.

## What changed

- Added TE reference mode in sidebar.
- Default mode: Realized TE distribution.
- TE target line = median of calculated rolling TE panel.
- Lower/upper TE bands = 25th/75th percentile of calculated rolling TE panel.
- Manual target mode still available.
- Tracking Error tab now shows TE summary table.

This keeps target and band lines on the same annualized scale as the rolling TE calculations.
