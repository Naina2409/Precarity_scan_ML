# Mapping to Computing for Data Science Lab (PBL)

The PBL asks for a real-world data science application in Python covering data collection, preprocessing,
statistical analysis, ML modelling and interpretation, with limitations (CO4) and deployment/ethics (CO5).

| Course module | What the project does | Where | Evidence in `reports/` |
|---|---|---|---|
| 1. Probability & statistics | Descriptive stats, skew/kurtosis, normality test, bootstrap CI for mean income | `src/stats_tests.py` | results.md §2, fig 01 |
| 2. Python for DS | Reusable cleaning pipeline with audit report; feature engineering (weekly hours, income/hour, volatility CV, trend, benefits count) | `src/data_cleaner.py`, `src/features.py` | results.md §1 |
| 3. Computing techniques | sklearn `Pipeline`/`ColumnTransformer`; logistic regression, KNN, decision tree; linear/RF regression; PCA, K-Means, hierarchical clustering; time-series smoothing + back-test | `src/models.py`, `src/clustering.py`, `src/timeseries.py` | §5-7, figs 05, 09-11 |
| 4. Advanced techniques | Random forest, gradient boosting, grid/random search, 5-fold CV, imbalance handling (class weights vs SMOTE), permutation importance (SHAP if installed), threshold policy | `src/models.py` | §7.1-7.4, figs 05-07 |
| 5. Statistical modelling | Welch t-test, ANOVA, Kruskal/Mann-Whitney, chi-square, Bonferroni, effect sizes, Type I/II error and power simulation, bias-variance curve | `src/stats_tests.py`, `src/models.py` | §3, §7.5, fig 08 |
| 6. Visualization | 13 report figures + interactive Plotly dashboard | `src/visualize.py`, `app.py` | `reports/figures/` |
| 7. Production ML | FastAPI scoring service, versioned model registry, PSI drift monitoring, retraining with promotion rule, Docker, CI | `api/`, `src/predict.py`, `src/monitoring.py`, `scripts/retrain.py` | `models/registry.json` |
| CO4 limitations | Documented in `docs/ETHICS_AND_LIMITATIONS.md` | | |
| CO5 ethics/deployment | Consent, anonymisation, fairness audit, salted hashing, no committed credentials | `src/auth.py`, `fairness_audit()` | §7.7 |

## Suggested split across your three evaluations (adjust to your actual dates)
1. **Evaluation 1** - data collection plan, cleaned dataset, EDA, descriptive stats and hypothesis tests (modules 1, 2, 5).
2. **Evaluation 2** - models, tuning, interpretation, clustering/PCA, time series (modules 3, 4).
3. **PBL final** - dashboard + API + monitoring, storytelling report, ethics and limitations (modules 6, 7).

## Team roles (4-5 members)
Data collection & survey | Cleaning/EDA/statistics | ML modelling | Dashboard & API/deployment | Report, ethics, testing.

## Report outline (follows the format of your earlier report)
Ch1 Introduction (background, problem, relevance, scope) | Ch2 Methodology (data, preprocessing, index, statistics, models) |
Ch3 Implementation & Results (setup, figures, tables, fairness) | Ch4 Conclusion, limitations, future scope | References.
Copy tables and figures straight from `reports/results.md` and `reports/figures/`.
