"""Train, tune, evaluate and register a model version.  Usage: python scripts/train.py [--data path] [--no-tune]"""
import argparse
import pandas as pd
import _common  # noqa
from src.config import CLEAN_DATA
from src.data_cleaner import clean_dataframe
from src.features import engineer
from src.models import save_versioned, train_and_select, MODEL_FEATURES

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(CLEAN_DATA)); ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    df, rep = clean_dataframe(pd.read_csv(a.data))
    r = train_and_select(df, do_tune=not a.no_tune)
    t = r["test"]
    metrics = {k: round(float(v), 4) for k, v in t.items() if isinstance(v, float)}
    entry = save_versioned(r["pipeline"], metrics, {"model": r["best_name"], "threshold": r["threshold"]}, df,
                           engineer(df)[MODEL_FEATURES], threshold=r["threshold"], note=f"best={r['best_name']}")
    print(f"Registered {entry['version']} ({r['best_name']}), hold-out ROC-AUC={metrics['roc_auc']}, recall={metrics['recall']}, threshold={r['threshold']}")
