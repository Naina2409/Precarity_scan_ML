"""Synthetic data generator.

IMPORTANT: no real worker data is shipped with this project. This module simulates a plausible
gig-worker survey (including 6 months of earnings and a financial-stress outcome) so the full
data-science pipeline can be developed and tested. Replace it with real survey data
(see docs/SURVEY_FORM.md) before drawing any real-world conclusions.
"""
import numpy as np
import pandas as pd
from .config import SEED, MONTH_COLS, BOOL_COLS, TARGET

CITIES = {"Delhi": 1.05, "Mumbai": 1.15, "Bangalore": 1.10, "Chennai": 0.95,
          "Kolkata": 0.85, "Hyderabad": 1.00, "Pune": 1.00}
# platform: (hourly pay INR, earnings volatility sd, sampling probability)
PLATFORMS = {"Zomato": (110, 0.12, 0.22), "Swiggy": (105, 0.12, 0.22), "Blinkit": (95, 0.14, 0.12),
             "Zepto": (95, 0.14, 0.08), "Uber": (150, 0.20, 0.14), "Ola": (140, 0.22, 0.10),
             "Urban Company": (170, 0.25, 0.12)}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_clean(n=600, seed=SEED, noise=0.5):
    """noise = sd of unobserved factors in the stress model (higher = harder prediction problem)."""
    rng = np.random.default_rng(seed)
    plat_names = list(PLATFORMS)
    p = np.array([PLATFORMS[k][2] for k in plat_names]); p = p / p.sum()
    platform = rng.choice(plat_names, n, p=p)
    city = rng.choice(list(CITIES), n)
    age = np.clip(rng.normal(28, 5, n), 18, 55).astype(int)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.88, 0.11, 0.01])
    exp = np.clip(rng.gamma(2.0, 10, n), 1, 120).astype(int)
    hours = np.clip(np.round(rng.normal(9.5, 2.0, n) * 2) / 2, 4, 14)
    days = rng.choice([4, 5, 6, 7], n, p=[0.05, 0.20, 0.50, 0.25])
    dependents = np.clip(rng.poisson(1.4, n), 0, 5)
    multi = (rng.random(n) < 0.30).astype(int)

    pay = np.array([PLATFORMS[x][0] for x in platform])
    vol = np.array([PLATFORMS[x][1] for x in platform]) * rng.uniform(0.6, 1.5, n)
    cityf = np.array([CITIES[c] for c in city])
    base = (pay * cityf * (1 + 0.003 * np.minimum(exp, 60)) * hours * days * 4.345
            * rng.beta(8, 6.5, n) * rng.lognormal(0, 0.12, n) * (1 + 0.08 * multi))
    base = np.clip(base, 5000, 70000)

    months = np.zeros((n, 6))
    trend = rng.normal(0, 0.02, n)
    for t in range(6):
        months[:, t] = base * (1 + trend * t + vol * rng.standard_normal(n))
    months = np.clip(months, 0.2 * base[:, None], None)
    income = months.mean(axis=1)
    cv = months.std(axis=1, ddof=1) / months.mean(axis=1)

    health = (rng.random(n) < np.clip(0.08 + income / 150000, 0.02, 0.6)).astype(int)
    leave = (rng.random(n) < 0.04).astype(int)
    retire = (rng.random(n) < 0.05).astype(int)
    loan = (rng.random(n) < np.where(np.isin(platform, ["Uber", "Ola"]), 0.75, 0.55)).astype(int)
    savings = np.round(np.clip(rng.exponential(0.7, n) * income / 20000, 0, 8), 1)

    z = (-1.1 * (income - 20000) / 10000 + 2.2 * (cv - 0.15) + 0.45 * dependents - 0.7 * savings
         + 0.5 * loan + 0.35 * (hours * days > 60) - 0.4 * health + 0.3 * (1 - multi)
         + rng.normal(0, noise, n))
    lo, hi = -10.0, 10.0
    for _ in range(60):                       # calibrate intercept: ~30% prevalence
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > 0.30: hi = mid
        else: lo = mid
    y = (rng.random(n) < _sigmoid(z + (lo + hi) / 2)).astype(int)

    df = pd.DataFrame({
        "worker_id": [f"GW{i + 1:04d}" for i in range(n)], "age": age, "gender": gender,
        "city": city, "platform": platform, "months_experience": exp,
        "hours_per_day": hours, "days_per_week": days, "dependents": dependents,
        "monthly_income": np.round(income).astype(int)})
    for i, c in enumerate(MONTH_COLS):
        df[c] = np.round(months[:, i]).astype(int)
    df["has_health_insurance"] = health; df["has_paid_leave"] = leave; df["has_retirement"] = retire
    df["multiple_platforms"] = multi; df["has_work_loan"] = loan
    df["savings_months"] = savings; df[TARGET] = y
    return df


def make_messy(clean, seed=SEED):
    """Create a deliberately dirty copy (typos, units, missing, duplicates, invalid values)."""
    rng = np.random.default_rng(seed + 1)
    d = clean.copy()
    n = len(d)
    city_var = {"Delhi": ["delhi", "New Delhi", "DELHI"], "Bangalore": ["bengaluru", "Bangalore", "BANGALORE"],
                "Mumbai": ["mumbai", "Bombay"], "Chennai": ["chennai", "Madras"]}
    d["city"] = [rng.choice(city_var[c]) if c in city_var and rng.random() < 0.5 else c for c in d["city"]]
    plat_var = {"Zomato": ["zomato", "ZOMATO"], "Swiggy": ["swiggy", "Swigy"], "Uber": ["uber", "Uber Eats"]}
    d["platform"] = [rng.choice(plat_var[p]) if p in plat_var and rng.random() < 0.5 else p for p in d["platform"]]
    yn = lambda v: rng.choice(["Yes", "Y", "true"]) if v == 1 else rng.choice(["No", "N", "false"])
    for c in BOOL_COLS + [TARGET]:
        d[c] = [yn(v) for v in d[c]]
    d["monthly_income"] = [f"₹{v:,}" if rng.random() < 0.3 else v for v in d["monthly_income"]]
    for c in ["age", "savings_months", "dependents"] + MONTH_COLS[:3]:
        d.loc[rng.random(n) < 0.03, c] = np.nan
    d.loc[rng.random(n) < 0.01, "gender"] = ""
    d.loc[rng.choice(n, 6, replace=False), "days_per_week"] = 26      # invalid (the original bug)
    d.loc[rng.choice(n, 3, replace=False), "age"] = 0
    d = pd.concat([d, d.sample(8, random_state=seed)], ignore_index=True)   # duplicates
    d = d.rename(columns={"worker_id": "Worker ID", "hours_per_day": "Hours/Day",
                          "days_per_week": "Days per week", "monthly_income": "Monthly Income",
                          "has_health_insurance": "Health Insurance"})
    return d
