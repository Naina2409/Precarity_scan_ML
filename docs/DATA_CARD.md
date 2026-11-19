# Data card

**Status: the shipped dataset is SYNTHETIC.** `scripts/generate_data.py` simulates 600 gig workers (7 platforms, 7 cities,
6 months of earnings) with a financial-stress outcome generated from a latent model plus noise (`--noise`, default 0.5).
Findings from it describe the simulator, not real workers. Replace `data/sample_data.csv` with real survey data
(`docs/SURVEY_FORM.md`) and re-run `python scripts/run_analysis.py`.

`data/raw_survey_messy.csv` is the same data deliberately corrupted (typos, currency strings, Yes/No variants, missing values,
8 duplicates, 6 invalid `days_per_week`, 3 invalid ages) to demonstrate the cleaner.

| Column | Meaning | Valid range / values |
|---|---|---|
| worker_id | Anonymous id | |
| age, gender | Demographics (gender used only for the fairness audit, never as a model feature) | age 16-80 |
| city, platform | Location and main platform | |
| months_experience | Months doing gig work | 0-360 |
| hours_per_day, days_per_week | Working pattern | 1-18, 1-7 |
| dependents | People relying on the income | 0-10 |
| monthly_income | Mean monthly income, INR | 0-300000 |
| income_m1..income_m6 | Earnings of each of the last 6 months | >= 0 |
| has_health_insurance, has_paid_leave, has_retirement | Benefits (0/1) | |
| multiple_platforms, has_work_loan | 0/1 | |
| savings_months | Emergency savings in months of expenses | 0-24 |
| **financial_stress** | **Target.** 1 = could not cover an emergency expense (e.g. INR 10,000) in the last 3 months | 0/1 |

Derived: `weekly_hours`, `income_per_hour`, `income_cv` (std/mean of 6 months), `income_trend` (last 3 vs first 3 months), `benefits_count`.
