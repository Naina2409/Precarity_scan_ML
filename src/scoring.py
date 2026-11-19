"""Transparent Precarity Index (0-100) plus sensitivity analysis and PCA-based weights.

The index is a *rule-based indicator* with documented assumptions (benchmark income, weights).
It is NOT a prediction: the ML model in models.py predicts a separately observed outcome.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import INDEX_WEIGHTS, LIVING_INCOME_BENCHMARK, RISK_BINS
from .features import engineer

DEFAULT_CV = 0.15   # used only when earnings history is unavailable


def components(df, cv_default=DEFAULT_CV):
    d = engineer(df) if "weekly_hours" not in df else df
    cv = d["income_cv"].fillna(cv_default)
    out = pd.DataFrame(index=d.index)
    out["income_inadequacy"] = ((LIVING_INCOME_BENCHMARK - d["monthly_income"]) / LIVING_INCOME_BENCHMARK).clip(0, 1)
    out["income_volatility"] = (cv / 0.5).clip(0, 1)
    out["benefits_gap"] = 1 - d["benefits_count"] / 3
    out["platform_dependence"] = np.where(d["multiple_platforms"] == 1, 0.5, 1.0)
    out["hours_strain"] = ((d["weekly_hours"] - 48) / 32).clip(0, 1)
    out["asset_risk"] = d["has_work_loan"].astype(float)
    return out


def precarity_index(df, weights=None, cv_default=DEFAULT_CV):
    w = pd.Series(weights or INDEX_WEIGHTS)
    comp = components(df, cv_default)[w.index]
    return (comp * w).sum(axis=1) * 100 / w.sum()


def risk_level(score):
    lo, hi = RISK_BINS
    return pd.cut(score, [-np.inf, lo, hi, np.inf], labels=["Low", "Medium", "High"], right=False)


def explain(df, weights=None):
    """Point contribution of each component to the index (sums to the index)."""
    w = pd.Series(weights or INDEX_WEIGHTS)
    return components(df)[w.index] * w * 100 / w.sum()


def sensitivity_analysis(df, n_draws=500, seed=42):
    """Re-score with random weight vectors (Dirichlet around the baseline) and measure stability."""
    rng = np.random.default_rng(seed)
    base_w = pd.Series(INDEX_WEIGHTS)
    base = precarity_index(df)
    base_lvl = risk_level(base)
    comp = components(df)[base_w.index]
    rhos, changed = [], []
    for _ in range(n_draws):
        w = rng.dirichlet(base_w.values * 30)          # concentrated around the baseline weights
        s = (comp * w).sum(axis=1) * 100
        rhos.append(spearmanr(base, s)[0])
        changed.append(float((risk_level(s) != base_lvl).mean()))
    return {"n_draws": n_draws, "mean_spearman": float(np.mean(rhos)), "min_spearman": float(np.min(rhos)),
            "mean_share_changing_risk_band": float(np.mean(changed)),
            "max_share_changing_risk_band": float(np.max(changed))}


def pca_weights(df):
    """Exploratory data-driven weights: |loadings| of the first principal component."""
    from sklearn.decomposition import PCA
    comp = components(df)
    X = (comp - comp.mean()) / comp.std(ddof=0).replace(0, 1)
    pca = PCA(n_components=1).fit(X.fillna(0))
    load = pd.Series(np.abs(pca.components_[0]), index=comp.columns)
    return {"weights": (load / load.sum()).round(3).to_dict(),
            "explained_variance_pc1": float(pca.explained_variance_ratio_[0])}
