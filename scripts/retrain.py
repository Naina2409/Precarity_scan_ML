"""Retrain on current data (+ optional new labelled rows) and promote only if not worse than the live model.
Usage: python scripts/retrain.py [--new data/new_labelled.csv] [--tolerance 0.01]"""
import argparse, json
import pandas as pd
import _common  # noqa
from src.config import CLEAN_DATA, MODELS_DIR
from src.data_cleaner import clean_dataframe
from src.features import engineer
from src.models import MODEL_FEATURES, save_versioned, train_and_select

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--new"); ap.add_argument("--tolerance", type=float, default=0.01)
    a = ap.parse_args()
    frames = [pd.read_csv(CLEAN_DATA)]
    if a.new: frames.append(pd.read_csv(a.new))
    df, _ = clean_dataframe(pd.concat(frames, ignore_index=True))
    r = train_and_select(df)
    new_auc = float(r["test"]["roc_auc"])
    reg = MODELS_DIR / "registry.json"
    live_auc = json.loads(reg.read_text())[-1]["metrics"]["roc_auc"] if reg.exists() else 0.0
    if new_auc >= live_auc - a.tolerance:
        metrics = {k: round(float(v), 4) for k, v in r["test"].items() if isinstance(v, float)}
        e = save_versioned(r["pipeline"], metrics, {"model": r["best_name"], "threshold": r["threshold"]}, df,
                           engineer(df)[MODEL_FEATURES], threshold=r["threshold"], note="retrain")
        print(f"PROMOTED {e['version']}: AUC {new_auc:.3f} vs live {live_auc:.3f}")
    else:
        print(f"NOT promoted: AUC {new_auc:.3f} is worse than live {live_auc:.3f} by more than {a.tolerance}")
