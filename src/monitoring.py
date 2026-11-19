"""Module 7: simple model monitoring: Population Stability Index (PSI) drift check and log summary."""
import numpy as np, pandas as pd

def psi(expected, actual, bins=10):
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if len(e) < 20 or len(a) < 5: return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e); pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

def status(v):
    return "n/a" if np.isnan(v) else "stable" if v < 0.1 else "moderate drift" if v < 0.25 else "significant drift"

def drift_report(reference, current, features):
    rows = []
    for f in features:
        if f in reference and f in current and pd.api.types.is_numeric_dtype(reference[f]):
            v = psi(reference[f], current[f]); rows.append({"feature": f, "psi": v, "status": status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

def needs_retraining(report, max_drifted=3):
    return int((report["psi"] >= 0.25).sum()) >= max_drifted
