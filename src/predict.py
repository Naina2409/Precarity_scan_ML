"""Inference: validation, precarity index, ML probability and explanation for one worker record."""
import json, datetime as dt
import joblib, numpy as np, pandas as pd
from .config import (BOOL_COLS, MODELS_DIR, MONTH_COLS, PREDICTION_LOG, VALID_RANGES)
from .features import engineer
from .scoring import explain, precarity_index, risk_level

REQUIRED = ["city", "platform", "age", "months_experience", "hours_per_day", "days_per_week",
            "monthly_income", "dependents", "savings_months"] + BOOL_COLS
_cache = {}


def load_model(models_dir=MODELS_DIR):
    ptr = models_dir / "latest.json"
    if not ptr.exists():
        raise FileNotFoundError("No trained model found. Run: python scripts/train.py")
    info = json.loads(ptr.read_text())
    if _cache.get("version") != info["version"]:
        _cache["bundle"] = joblib.load(models_dir / info["file"]); _cache["version"] = info["version"]
    return _cache["bundle"]


def validate_record(rec):
    missing = [k for k in REQUIRED if rec.get(k) is None]
    if missing: raise ValueError(f"Missing fields: {missing}")
    for c, (lo, hi) in VALID_RANGES.items():
        if c in rec and rec[c] is not None and not (lo <= float(rec[c]) <= hi):
            raise ValueError(f"{c} must be between {lo} and {hi}")
    hist = rec.get("monthly_incomes")
    if hist is not None and (not 2 <= len(hist) <= 12 or any(float(v) < 0 for v in hist)):
        raise ValueError("monthly_incomes must hold 2-12 non-negative values")


def to_frame(rec):
    d = {k: rec[k] for k in REQUIRED}
    for c in BOOL_COLS: d[c] = int(bool(d[c]))
    hist = rec.get("monthly_incomes") or []
    hist = [float(v) for v in hist][-6:]
    for i, c in enumerate(MONTH_COLS):
        d[c] = hist[i - (6 - len(hist))] if len(hist) and i >= 6 - len(hist) else np.nan
    d["gender"] = rec.get("gender", "Unknown")
    return pd.DataFrame([d])


def predict_one(rec, models_dir=MODELS_DIR, log=True):
    validate_record(rec)
    bundle = load_model(models_dir)
    df = engineer(to_frame(rec))
    proba = float(bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1][0])
    idx = float(precarity_index(df).iloc[0])
    contrib = explain(df).iloc[0].sort_values(ascending=False)
    out = {"precarity_index": round(idx, 1), "risk_level": str(risk_level(pd.Series([idx])).iloc[0]),
           "stress_probability": round(proba, 3), "flagged": bool(proba >= bundle["threshold"]),
           "threshold": bundle["threshold"], "top_drivers": {k: round(float(v), 1) for k, v in contrib.head(3).items()},
           "all_components": {k: round(float(v), 1) for k, v in contrib.items()},
           "history_used": bool(rec.get("monthly_incomes")), "model_version": bundle["version"]}
    if log:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "model_version": bundle["version"],
                             "inputs": json.dumps({k: rec[k] for k in REQUIRED}, default=str),
                             "index": out["precarity_index"], "probability": out["stress_probability"]}])
        row.to_csv(PREDICTION_LOG, mode="a", header=not PREDICTION_LOG.exists(), index=False)
    return out
