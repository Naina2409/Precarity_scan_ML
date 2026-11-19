"""Drift check: compare logged predictions' inputs (or a CSV) with the training reference.
Usage: python scripts/monitor.py [--current data/new_data.csv]
With no --current it uses logs/predictions.csv (needs >= 5 logged predictions for a PSI value)."""
import argparse, json
import pandas as pd
import _common  # noqa
from src.config import MODELS_DIR, PREDICTION_LOG, MODEL_NUM
from src.features import engineer
from src.monitoring import drift_report, needs_retraining

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--current"); a = ap.parse_args()
    ver = json.loads((MODELS_DIR / "latest.json").read_text())["version"]
    ref = pd.read_csv(MODELS_DIR / f"reference_{ver}.csv")
    if a.current: cur = engineer(pd.read_csv(a.current))
    else:
        log = pd.read_csv(PREDICTION_LOG); cur = pd.DataFrame([json.loads(s) for s in log["inputs"]])
        cur = engineer(cur)   # logged inputs have no earnings history, so income_cv/trend are NaN (PSI skipped)
    rep = drift_report(ref, cur, MODEL_NUM); print(rep.to_string(index=False))
    print("RETRAIN RECOMMENDED" if needs_retraining(rep) else "No retraining needed")
