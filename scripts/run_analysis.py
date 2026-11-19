"""End-to-end analysis: cleaning audit -> EDA -> statistics -> index -> clustering -> time series ->
ML (compare, tune, imbalance, interpret, fairness) -> figures + reports/results.json + reports/results.md.
Usage: python scripts/run_analysis.py [--data path] [--no-tune]"""
import argparse, json
import numpy as np, pandas as pd
import _common  # noqa
from src import visualize as viz
from src.clustering import segment
from src.config import CLEAN_DATA, RAW_MESSY, REPORTS_DIR, MODEL_FEATURES, TARGET
from src.data_cleaner import clean_dataframe
from src.features import engineer
from src.models import (bias_variance_curve, fairness_audit, get_xy, imbalance_comparison, importance,
                        income_regression, save_versioned, train_and_select)
from src.scoring import pca_weights, precarity_index, risk_level, sensitivity_analysis
from src.stats_tests import (anova, bootstrap_ci, chi_square, correlation, describe, error_rates_simulation,
                             normality, pairwise_welch, welch_ttest)
from src.timeseries import analyse as ts_analyse


def md(df, floatfmt=3):
    d = df.copy()
    if d.index.name or not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index().rename(columns={"index": "variable"})
    d = d.astype(object)
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not float(v).is_integer() else str(v) for v in r) + " |")
    return "\n".join(rows)


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.DataFrame,)): return o.round(4).to_dict("records")
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(CLEAN_DATA)); ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    R, M = {}, ["# Analysis results (auto-generated)\n",
                "> Data are SYNTHETIC unless you replaced `data/sample_data.csv` with real survey data.\n"]
    # 1. cleaning audit on the messy file
    if RAW_MESSY.exists():
        _, rep = clean_dataframe(pd.read_csv(RAW_MESSY)); R["cleaning_audit_on_messy_file"] = rep
        M += ["## 1. Cleaning audit (raw_survey_messy.csv)\n", f"Rows in {rep['rows_in']} -> out {rep['rows_out']}; duplicates removed {rep['duplicates_removed']}; "
              f"out-of-range set to missing {rep['out_of_range_set_to_missing']}; imputed {rep['imputed']}\n"]
    df, _ = clean_dataframe(pd.read_csv(a.data)); d = engineer(df)
    # 2. EDA + descriptive stats
    num_cols = ["age", "monthly_income", "income_per_hour", "income_cv", "weekly_hours", "dependents", "savings_months", "benefits_count"]
    desc = describe(d, num_cols); R["descriptive"] = desc.reset_index().round(3).to_dict("records")
    d["any_benefit"] = (d["benefits_count"] > 0).astype(int)
    R["bootstrap_mean_income_ci"] = bootstrap_ci(d["monthly_income"]); R["normality_income"] = normality(d["monthly_income"])
    viz.income_distribution(d); corr = correlation(d, num_cols + [TARGET]); viz.correlation_heatmap(corr)
    viz.pairplot(d, ["monthly_income", "income_cv", "savings_months", "dependents"])
    M += ["## 2. Descriptive statistics\n", md(desc), f"\nMean income {R['bootstrap_mean_income_ci']['estimate']:.0f} (95% bootstrap CI {R['bootstrap_mean_income_ci']['ci_low']:.0f} - {R['bootstrap_mean_income_ci']['ci_high']:.0f}). "
          f"Shapiro-Wilk p={R['normality_income']['p_value']:.3g} (normal at 5%: {R['normality_income']['normal_at_5pct']}).\n"]
    # 3. hypothesis tests
    tests = {"multi_platform_income": welch_ttest(d, "monthly_income", "multiple_platforms"),
             "platform_income_anova": anova(d, "monthly_income", "platform"),
             "benefits_vs_platform_chi2": chi_square(d, "any_benefit", "platform"),
             "stress_vs_platform_chi2": chi_square(d, TARGET, "platform") if d[TARGET].notna().all() else None,
             "error_rates": error_rates_simulation()}
    pw = pairwise_welch(d, "monthly_income", "platform"); tests["pairwise_significant_pairs_bonferroni"] = int(pw["significant"].sum()) if len(pw) else 0
    R["hypothesis_tests"] = tests
    M += ["## 3. Hypothesis tests\n"] + [f"- **{k}**: " + ", ".join(f"{kk}={vv:.4g}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()) for k, v in tests.items() if isinstance(v, dict)] + ["\n"]
    # 4. precarity index
    idx = precarity_index(d); lvl = risk_level(idx); viz.risk_distribution(idx, lvl)
    R["index"] = {"mean": float(idx.mean()), "bands": {str(k): int(v) for k, v in lvl.value_counts().items()}, "sensitivity": sensitivity_analysis(d), "pca_weights": pca_weights(d)}
    M += ["## 4. Precarity index\n", f"Mean {idx.mean():.1f}; bands { {str(k): int(v) for k, v in lvl.value_counts().items()} }. Sensitivity (500 random weight draws): "
          f"mean Spearman {R['index']['sensitivity']['mean_spearman']:.2f}, workers changing band {R['index']['sensitivity']['mean_share_changing_risk_band']:.0%} on average.\n"]
    # 5. segmentation + time series
    seg = segment(df); viz.clustering_figs(seg)
    R["segments"] = {"k": seg["k"], "silhouette": seg["silhouette"], "ari_kmeans_vs_hier": seg["ari_kmeans_vs_hier"], "pca_explained": seg["pca_explained"], "profile": seg["profile"].reset_index()}
    M += ["## 5. Worker segments (K-Means)\n", md(seg["profile"]), f"\nk={seg['k']}, silhouette {seg['silhouette']:.2f}; PCA first two components explain {sum(seg['pca_explained'][:2]):.0%} of variance.\n"]
    ts = ts_analyse(df); viz.timeseries_fig(ts["table"])
    R["timeseries"] = {"trend_per_month": ts["overall_trend_per_month"], "trend_p": ts["trend_p_value"], "mae_ses": ts["mae_ses"], "mae_naive": ts["mae_naive"]}
    M += ["## 6. Time series\n", f"Average earnings trend {ts['overall_trend_per_month']:.0f} INR/month (p={ts['trend_p_value']:.3g}); month-6 back-test MAE: exponential smoothing {ts['mae_ses']:.0f} vs naive {ts['mae_naive']:.0f}.\n"]
    # 7. ML
    X, y, _ = get_xy(df)
    res = train_and_select(df, do_tune=not a.no_tune)
    viz.model_comparison(res["cv_table"]); viz.roc_and_confusion(res["test"], res["best_name"])
    imp = importance(res["pipeline"], res["Xte"], res["yte"]); viz.feature_importance(imp)
    bv = bias_variance_curve(X, y); viz.bias_variance(bv)
    imb = imbalance_comparison(X, y); reg = income_regression(df)
    fair = {}
    for col in ["gender", "city"]:
        fa = fairness_audit(res["d_test"], res["test"]["proba"], res["yte"], col, res["threshold"]); fair[col] = fa; viz.fairness_fig(fa, col)
    t = res["test"]
    R["ml"] = {"n": res["n"], "prevalence": res["prevalence"], "best_model": res["best_name"], "threshold": res["threshold"],
               "cv_table": res["cv_table"], "tuned": res.get("tuned"), "holdout": {k: v for k, v in t.items() if isinstance(v, float)},
               "holdout_confusion_matrix": t["confusion_matrix"], "holdout_at_0.5": {k: v for k, v in res["test_at_0.5"].items() if isinstance(v, float)},
               "importance_top": imp.head(8), "imbalance": imb, "regression": reg, "fairness": fair,
               "best_depth_by_cv": int(bv.loc[bv["cv_auc"].idxmax(), "max_depth"])}
    M += ["## 7. Machine learning (target = financial_stress)\n", f"n={res['n']}, prevalence {res['prevalence']:.1%}.\n", "### 7.1 5-fold CV model comparison\n", md(res["cv_table"][["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_roc_auc"]]),
          f"\n### 7.2 Selected model: {res['best_name']} (threshold {res['threshold']} = highest threshold keeping CV recall >= 0.70)\n",
          "Hold-out: " + ", ".join(f"{k}={v:.3f}" for k, v in t.items() if isinstance(v, float)) + f"; confusion matrix {t['confusion_matrix']}\n",
          "### 7.3 Imbalance strategies (CV)\n", md(imb), "\n### 7.4 Permutation importance\n", md(imp.head(8)),
          f"\n### 7.5 Bias-variance: best tree depth by CV = {R['ml']['best_depth_by_cv']}; deeper trees overfit (train AUC -> 1.0, CV AUC falls)\n",
          "### 7.6 Income regression\n", md(reg), "\n### 7.7 Fairness audit (hold-out; small groups are unreliable)\n"] + [md(v) + "\n" for v in fair.values()]
    entry = save_versioned(res["pipeline"], {k: round(float(v), 4) for k, v in t.items() if isinstance(v, float)},
                           {"model": res["best_name"], "threshold": res["threshold"]}, df, d[MODEL_FEATURES], threshold=res["threshold"], note="run_analysis")
    R["registered_model"] = entry["version"]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "results.json").write_text(json.dumps(jsonable(R), indent=2, default=str))
    (REPORTS_DIR / "results.md").write_text("\n".join(M))
    print("Done. See reports/results.md, reports/results.json and reports/figures/. Registered model", entry["version"])
