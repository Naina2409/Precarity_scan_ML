"""Feature engineering shared by scoring, modelling and the API."""
import numpy as np
import pandas as pd
from .config import MONTH_COLS, BENEFIT_COLS, WEEKS_PER_MONTH


def engineer(df):
    d = df.copy()
    d["weekly_hours"] = d["hours_per_day"] * d["days_per_week"]
    denom = (d["weekly_hours"] * WEEKS_PER_MONTH).replace(0, np.nan)
    d["income_per_hour"] = d["monthly_income"] / denom
    months = [c for c in MONTH_COLS if c in d.columns]
    if len(months) >= 2:
        m = d[months].astype(float)
        mean = m.mean(axis=1).replace(0, np.nan)
        d["income_cv"] = (m.std(axis=1, ddof=1) / mean).clip(0, 1.5)
        k = max(1, len(months) // 2)
        first = m.iloc[:, :k].mean(axis=1).replace(0, np.nan)
        last = m.iloc[:, -k:].mean(axis=1)
        d["income_trend"] = ((last - first) / first).clip(-1, 1)
    else:
        d["income_cv"] = np.nan
        d["income_trend"] = np.nan
    d["benefits_count"] = d[BENEFIT_COLS].sum(axis=1)
    return d
