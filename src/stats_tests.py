"""Module 1 and 5: descriptive statistics, inference, effect sizes, error rates and power."""
import numpy as np
import pandas as pd
from scipy import stats


def describe(df, cols):
    out = df[cols].describe().T
    out["skew"] = df[cols].skew()
    out["kurtosis"] = df[cols].kurt()
    return out.round(3)


def bootstrap_ci(x, stat=np.mean, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.Series(x).dropna())
    boots = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(stat(x)), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(x))}


def normality(x):
    x = np.asarray(pd.Series(x).dropna())[:5000]
    w, p = stats.shapiro(x)
    return {"shapiro_W": float(w), "p_value": float(p), "normal_at_5pct": bool(p > 0.05)}


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp else 0.0


def welch_ttest(df, value, group):
    """Welch t-test between the two levels of a binary column, with Mann-Whitney as a robustness check."""
    g = df[group].astype(int)
    a, b = df.loc[g == 1, value].dropna(), df.loc[g == 0, value].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"test": "Welch t-test", "value": value, "group": group, "mean_group1": float(a.mean()),
            "mean_group0": float(b.mean()), "t": float(t), "p_value": float(p),
            "mannwhitney_p": float(pu), "cohens_d": cohens_d(a, b), "n1": len(a), "n0": len(b)}


def anova(df, value, group):
    groups = [g[value].dropna().values for _, g in df.groupby(group) if len(g) > 1]
    f, p = stats.f_oneway(*groups)
    h, pk = stats.kruskal(*groups)
    k, n = len(groups), sum(len(g) for g in groups)
    eta2 = (f * (k - 1)) / (f * (k - 1) + (n - k))
    return {"test": "One-way ANOVA", "value": value, "group": group, "F": float(f), "p_value": float(p),
            "kruskal_p": float(pk), "eta_squared": float(eta2), "k_groups": k}


def pairwise_welch(df, value, group, alpha=0.05):
    """All pairwise Welch tests with Bonferroni correction (controls Type I error across comparisons)."""
    levels = sorted(df[group].dropna().unique())
    rows = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a = df.loc[df[group] == levels[i], value].dropna()
            b = df.loc[df[group] == levels[j], value].dropna()
            if len(a) > 1 and len(b) > 1:
                rows.append({"a": levels[i], "b": levels[j], "p": float(stats.ttest_ind(a, b, equal_var=False)[1])})
    out = pd.DataFrame(rows)
    if len(out):
        m = len(out)
        out["p_bonferroni"] = (out["p"] * m).clip(upper=1)
        out["significant"] = out["p_bonferroni"] < alpha
    return out


def chi_square(df, a, b):
    ct = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return {"test": "Chi-square independence", "a": a, "b": b, "chi2": float(chi2), "dof": int(dof),
            "p_value": float(p), "cramers_v": float(v)}


def correlation(df, cols, method="spearman"):
    return df[cols].corr(method=method).round(3)


def error_rates_simulation(n=60, effect=0.5, alpha=0.05, n_sim=2000, seed=42):
    """Estimate Type I error (no true effect) and Type II error / power (true standardised effect)."""
    rng = np.random.default_rng(seed)
    t1 = np.mean([stats.ttest_ind(rng.normal(0, 1, n), rng.normal(0, 1, n))[1] < alpha for _ in range(n_sim)])
    pw = np.mean([stats.ttest_ind(rng.normal(0, 1, n), rng.normal(effect, 1, n))[1] < alpha for _ in range(n_sim)])
    return {"n_per_group": n, "alpha": alpha, "effect_size_d": effect,
            "type_I_error_rate": float(t1), "power": float(pw), "type_II_error_rate": float(1 - pw)}
