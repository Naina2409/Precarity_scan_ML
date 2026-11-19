"""Module 3: simple time-series analysis of monthly earnings (trend, smoothing, forecast back-test)."""
import numpy as np, pandas as pd
from scipy.stats import linregress
from .config import MONTH_COLS


def monthly_by_platform(df):
    long = df.melt(id_vars=["platform"], value_vars=[c for c in MONTH_COLS if c in df], var_name="month", value_name="income")
    long["t"] = long["month"].str.extract(r"(\d+)").astype(int)
    return long.groupby(["platform", "t"])["income"].mean().unstack("platform")


def ses_forecast(series, alpha):
    level = series[0]
    for v in series[1:]:
        level = alpha * v + (1 - alpha) * level
    return level


def best_alpha(train):
    grid = np.linspace(0.1, 0.9, 9)
    def sse(a):
        lvl, e = train[0], 0.0
        for v in train[1:]:
            e += (v - lvl) ** 2; lvl = a * v + (1 - a) * lvl
        return e
    return float(min(grid, key=sse))


def analyse(df):
    tbl = monthly_by_platform(df)
    overall = tbl.mean(axis=1)
    slope = linregress(overall.index, overall.values)
    rows = []
    for plat in tbl.columns:
        s = tbl[plat].values
        a = best_alpha(s[:-1])
        rows.append({"platform": plat, "ses_alpha": a, "forecast_m6_ses": ses_forecast(s[:-1], a),
                     "forecast_m6_naive": s[-2], "actual_m6": s[-1], "next_month_forecast": ses_forecast(s, best_alpha(s))})
    bt = pd.DataFrame(rows)
    bt["abs_err_ses"] = (bt["forecast_m6_ses"] - bt["actual_m6"]).abs()
    bt["abs_err_naive"] = (bt["forecast_m6_naive"] - bt["actual_m6"]).abs()
    return {"table": tbl, "overall_trend_per_month": float(slope.slope), "trend_p_value": float(slope.pvalue),
            "backtest": bt, "mae_ses": float(bt["abs_err_ses"].mean()), "mae_naive": float(bt["abs_err_naive"].mean())}
