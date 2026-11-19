# Collecting real data (Google Form / Kobo / ODK)

**Consent text (first page):** "This survey is voluntary and anonymous. We do not ask for your name, phone or address. Answers are used
only for a university research project. You can stop at any time." Add a required checkbox. Do not collect Aadhaar, phone numbers or
exact addresses. Store the sheet privately; share only aggregated results.

| # | Question | Column |
|---|---|---|
| 1 | Age | age |
| 2 | Gender (Male/Female/Other/Prefer not to say) | gender |
| 3 | City | city |
| 4 | Main platform | platform |
| 5 | Months doing gig work | months_experience |
| 6 | Hours worked per day (average) | hours_per_day |
| 7 | Days worked per week | days_per_week |
| 8 | Earnings in each of the last 6 months (6 numeric fields) | income_m1..income_m6 |
| 9 | People who depend on your income | dependents |
| 10 | Do you have health insurance / paid leave / retirement plan? (3 yes/no) | has_health_insurance, has_paid_leave, has_retirement |
| 11 | Do you work on more than one platform? | multiple_platforms |
| 12 | Do you have a loan taken for work (vehicle, phone)? | has_work_loan |
| 13 | Emergency savings, in months of expenses | savings_months |
| 14 | In the last 3 months, could you not cover an unexpected expense of INR 10,000? (yes/no) | financial_stress |

Export as CSV and run it through the cleaner (the Streamlit "Data & cleaning" page or `clean_csv_file`); the column aliases cover
most Google-Form wording. Rule of thumb: about 10 positive outcomes per predictor, so for ~20 predictors and a ~30% stress rate
aim for roughly 600+ responses; fewer is workable but report wider uncertainty. Reach workers through several channels (not only
one WhatsApp group) to limit selection bias.
