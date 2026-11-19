"""Create the synthetic dataset and a deliberately messy raw file (for the cleaning demo).
Usage: python scripts/generate_data.py [--n 600] [--noise 0.5]"""
import argparse
import _common  # noqa
from src.config import CLEAN_DATA, RAW_MESSY, SEED
from src.data_generator import make_clean, make_messy

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--noise", type=float, default=0.5); ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    CLEAN_DATA.parent.mkdir(exist_ok=True)
    clean = make_clean(a.n, a.seed, a.noise); clean.to_csv(CLEAN_DATA, index=False)
    make_messy(clean, a.seed).to_csv(RAW_MESSY, index=False)
    print(f"Wrote {CLEAN_DATA.name} ({len(clean)} rows, stress rate {clean['financial_stress'].mean():.1%}) and {RAW_MESSY.name}")
    print("NOTE: this is SYNTHETIC data. Replace with real survey data (docs/SURVEY_FORM.md) for real conclusions.")
