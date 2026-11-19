# What changed from the original PrecarityScan

| Problem found in the original code | Fix in this version |
|---|---|
| Score = hand-picked weights only; no ML, no statistics | Added statistics (`stats_tests.py`), ML with a real outcome variable (`models.py`), clustering/PCA, time series |
| `metrics.py`, `scorer.py`, `visualize.py` were never imported by `app.py`; two different scoring formulas | One scoring module (`scoring.py`) used by app, API and analysis; dead code removed |
| `days_per_week` of 20-26 in the sample data inflated weekly hours (up to ~260) | Range validation sets invalid values to missing and reports them; generator cannot produce them |
| `income_volatility` computed from one income value, so always 0.5 | Volatility (CV) and trend computed from 6 months of earnings |
| Silent defaults (age 30, income 15000) | Explicit median/mode imputation with an audit report |
| Fuzzy substring column matching (e.g. `id` matched `paid_leave`) | Exact normalised alias matching (unit-tested) |
| Unsalted SHA-256 passwords, demo credentials in the repo | Salted PBKDF2-HMAC-SHA256 (200k iterations), lockout after 5 failures, no committed accounts; random demo passwords at setup |
| Weights unjustified | Sensitivity analysis (random weights) + PCA-derived weights for comparison |
| No tests, no README, unpinned runtime | 34 unit tests (one needs fastapi installed), docs, CI workflow, Docker files |
| No deployment story | FastAPI service, versioned model registry, drift check (PSI), retraining script with promotion rule |
