"""Data cleaning pipeline with an audit report.

Differences from the first version:
 * exact (normalised) column-name aliases instead of fuzzy substring matching;
 * out-of-range values are flagged and set to missing (not silently clipped or defaulted);
 * imputation is explicit (median / mode / row-mean for monthly earnings) and reported;
 * returns a report dict so every change is documented (useful for the project report).
"""
import re
import numpy as np
import pandas as pd
from .config import (BOOL_COLS, CAT_COLS, MONTH_COLS, NUM_BASE, TARGET, VALID_RANGES)

ALIASES = {
    "worker_id": ["worker_id", "id", "workerid", "employee_id"],
    "age": ["age", "worker_age"],
    "gender": ["gender", "sex"],
    "city": ["city", "location", "town"],
    "platform": ["platform", "app", "company", "gig_platform"],
    "months_experience": ["months_experience", "experience_months", "experience", "months_of_experience"],
    "hours_per_day": ["hours_per_day", "hours_day", "daily_hours", "hrs_per_day"],
    "days_per_week": ["days_per_week", "working_days", "days_week", "days_a_week"],
    "monthly_income": ["monthly_income", "income", "monthly_earnings", "earnings", "salary"],
    "dependents": ["dependents", "dependants", "family_dependents"],
    "savings_months": ["savings_months", "emergency_savings_months", "savings"],
    "has_health_insurance": ["has_health_insurance", "health_insurance", "insurance"],
    "has_paid_leave": ["has_paid_leave", "paid_leave", "leave"],
    "has_retirement": ["has_retirement", "retirement", "pension", "retirement_plan"],
    "multiple_platforms": ["multiple_platforms", "multi_platform", "works_multiple_platforms"],
    "has_work_loan": ["has_work_loan", "work_loan", "vehicle_loan", "loan"],
    TARGET: [TARGET, "financial_stress", "stress", "emergency_gap"],
}
for i, c in enumerate(MONTH_COLS, 1):
    ALIASES[c] = [c, f"income_month_{i}", f"month_{i}_income", f"m{i}"]

CITY_MAP = {"new_delhi": "Delhi", "delhi": "Delhi", "bengaluru": "Bangalore", "bangalore": "Bangalore",
            "bombay": "Mumbai", "mumbai": "Mumbai", "madras": "Chennai", "chennai": "Chennai",
            "calcutta": "Kolkata", "kolkata": "Kolkata", "hyderabad": "Hyderabad", "pune": "Pune"}
PLATFORM_MAP = {"zomato": "Zomato", "swiggy": "Swiggy", "swigy": "Swiggy", "blinkit": "Blinkit",
                "zepto": "Zepto", "uber": "Uber", "uber_eats": "Uber", "ola": "Ola",
                "urban_company": "Urban Company", "urbanclap": "Urban Company"}
TRUE = {"yes", "y", "true", "t", "1", "1.0", "has", "have"}
FALSE = {"no", "n", "false", "f", "0", "0.0", "none"}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def standardise_columns(df):
    lookup = {}
    for std, names in ALIASES.items():
        for n in names:
            lookup.setdefault(_norm(n), std)
    mapping, used = {}, set()
    for col in df.columns:
        std = lookup.get(_norm(col))
        if std and std not in used:
            mapping[col] = std; used.add(std)
    return df.rename(columns=mapping), mapping


def _to_number(s):
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype(str).str.replace(r"[₹$,]|rs\.?|inr", "", regex=True, case=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _to_bool(s):
    def f(v):
        if pd.isna(v): return np.nan
        k = str(v).strip().lower()
        return 1.0 if k in TRUE else 0.0 if k in FALSE else np.nan
    return s.map(f)


def clean_dataframe(df_in):
    """Return (clean_df, report)."""
    report = {"rows_in": int(len(df_in))}
    df, mapping = standardise_columns(df_in.copy())
    report["column_mapping"] = {str(k): v for k, v in mapping.items()}
    report["unmapped_columns"] = [c for c in df_in.columns if c not in mapping]

    for c in NUM_BASE + MONTH_COLS:
        if c in df: df[c] = _to_number(df[c])
    for c in BOOL_COLS + [TARGET]:
        if c in df: df[c] = _to_bool(df[c])
    for c in CAT_COLS:
        if c in df:
            df[c] = df[c].fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})
    if "city" in df: df["city"] = df["city"].map(lambda v: CITY_MAP.get(_norm(v), v.title() if v else "Unknown"))
    if "platform" in df: df["platform"] = df["platform"].map(lambda v: PLATFORM_MAP.get(_norm(v), v.title() if v else "Unknown"))
    if "gender" in df: df["gender"] = df["gender"].map(lambda v: v.title() if v else "Unknown")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if "worker_id" in df:
        df = df.drop_duplicates(subset="worker_id", keep="first").reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))

    out_of_range = {}
    for c, (lo, hi) in VALID_RANGES.items():
        if c in df:
            bad = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
            if bad.any():
                out_of_range[c] = int(bad.sum()); df.loc[bad, c] = np.nan
    report["out_of_range_set_to_missing"] = out_of_range

    months = [c for c in MONTH_COLS if c in df]
    for c in months:
        df.loc[df[c] < 0, c] = np.nan
    if months:   # fill a missing month with that worker's mean of the other months
        rowmean = df[months].mean(axis=1)
        for c in months: df[c] = df[c].fillna(rowmean)
        if "monthly_income" not in df: df["monthly_income"] = np.nan
        df["monthly_income"] = df["monthly_income"].fillna(rowmean).round()

    report["missing_before_imputation"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    imputed = {}
    for c in [c for c in NUM_BASE if c in df]:
        n = int(df[c].isna().sum())
        if n: df[c] = df[c].fillna(df[c].median()); imputed[c] = f"median ({n})"
    for c in [c for c in BOOL_COLS if c in df]:
        n = int(df[c].isna().sum())
        if n: df[c] = df[c].fillna(df[c].mode().iloc[0]); imputed[c] = f"mode ({n})"
    for c in BOOL_COLS:
        if c in df: df[c] = df[c].astype(int)
    if "worker_id" not in df:
        df.insert(0, "worker_id", [f"GW{i + 1:04d}" for i in range(len(df))])
    else:
        mask = df["worker_id"].isna() | (df["worker_id"].astype(str).str.strip() == "")
        df.loc[mask, "worker_id"] = [f"GW{i:04d}" for i in range(9000, 9000 + int(mask.sum()))]
    report["imputed"] = imputed
    report["target_missing"] = int(df[TARGET].isna().sum()) if TARGET in df else None
    if TARGET in df:
        df[TARGET] = df[TARGET].astype("Int64")
    report["rows_out"] = int(len(df))
    return df, report


def clean_csv_file(path):
    return clean_dataframe(pd.read_csv(path))
