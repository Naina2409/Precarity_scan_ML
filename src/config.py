"""Central configuration: paths, column groups, validation ranges and index weights."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

CLEAN_DATA = DATA_DIR / "sample_data.csv"          # cleaned dataset used for analysis
RAW_MESSY = DATA_DIR / "raw_survey_messy.csv"      # deliberately messy file to demo the cleaner
USERS_FILE = DATA_DIR / "users.csv"                # created at first setup (git-ignored)
SUBMISSIONS_FILE = DATA_DIR / "worker_submissions.csv"  # consented worker entries (git-ignored)
PREDICTION_LOG = LOGS_DIR / "predictions.csv"

SEED = 42
WEEKS_PER_MONTH = 4.345

MONTH_COLS = [f"income_m{i}" for i in range(1, 7)]
BOOL_COLS = ["has_health_insurance", "has_paid_leave", "has_retirement",
             "multiple_platforms", "has_work_loan"]
BENEFIT_COLS = ["has_health_insurance", "has_paid_leave", "has_retirement"]
NUM_BASE = ["age", "months_experience", "hours_per_day", "days_per_week",
            "monthly_income", "dependents", "savings_months"]
CAT_COLS = ["city", "platform", "gender"]
TARGET = "financial_stress"   # 1 = could not cover an emergency expense in the last 3 months

# Features given to the ML models. Gender is deliberately excluded (used only for the fairness audit).
MODEL_NUM = ["age", "months_experience", "hours_per_day", "days_per_week", "monthly_income",
             "dependents", "savings_months", "weekly_hours", "income_per_hour",
             "income_cv", "income_trend", "benefits_count"]
MODEL_BOOL = BOOL_COLS
MODEL_CAT = ["city", "platform"]
MODEL_FEATURES = MODEL_NUM + MODEL_BOOL + MODEL_CAT

# Valid ranges: values outside are treated as invalid (set to missing), never silently clipped.
VALID_RANGES = {
    "age": (16, 80), "months_experience": (0, 360), "hours_per_day": (1, 18),
    "days_per_week": (1, 7), "monthly_income": (0, 300000), "dependents": (0, 10),
    "savings_months": (0, 24),
}

# Precarity index (transparent, rule-based). Weights are an assumption: see sensitivity analysis.
LIVING_INCOME_BENCHMARK = 20000   # assumed monthly living-income benchmark in INR
INDEX_WEIGHTS = {
    "income_inadequacy": 0.25,
    "income_volatility": 0.20,
    "benefits_gap": 0.20,
    "platform_dependence": 0.15,
    "hours_strain": 0.10,
    "asset_risk": 0.10,
}
RISK_BINS = (45, 60)   # <45 Low, 45-60 Medium, >=60 High (assumption: recalibrate on real data)
