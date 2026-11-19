# Analysis results (auto-generated)

> Data are SYNTHETIC unless you replaced `data/sample_data.csv` with real survey data.

## 1. Cleaning audit (raw_survey_messy.csv)

Rows in 608 -> out 600; duplicates removed 8; out-of-range set to missing {'age': 3, 'days_per_week': 6}; imputed {'age': 'median (27)', 'days_per_week': 'median (6)', 'dependents': 'median (25)', 'savings_months': 'median (17)'}

## 2. Descriptive statistics

| variable | count | mean | std | min | 25% | 50% | 75% | max | skew | kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|
| age | 600.0 | 27.617 | 4.885 | 18.0 | 24.0 | 28.0 | 31.0 | 42.0 | 0.069 | -0.302 |
| monthly_income | 600.0 | 17859.585 | 7789.072 | 4153.0 | 12199.500 | 16552.500 | 21903.750 | 56957.0 | 1.201 | 2.445 |
| income_per_hour | 600.0 | 73.300 | 25.572 | 23.841 | 54.206 | 69.158 | 88.032 | 171.172 | 0.728 | 0.515 |
| income_cv | 600.0 | 0.167 | 0.084 | 0.031 | 0.105 | 0.148 | 0.215 | 0.517 | 1.116 | 1.373 |
| weekly_hours | 600.0 | 56.127 | 13.594 | 20.0 | 45.0 | 56.0 | 66.0 | 98.0 | 0.191 | -0.212 |
| dependents | 600.0 | 1.343 | 1.121 | 0.0 | 1.0 | 1.0 | 2.0 | 5.0 | 0.791 | 0.406 |
| savings_months | 600.0 | 0.610 | 0.726 | 0.0 | 0.200 | 0.400 | 0.800 | 6.400 | 3.073 | 14.368 |
| benefits_count | 600.0 | 0.287 | 0.502 | 0.0 | 0.0 | 0.0 | 1.0 | 3.0 | 1.570 | 2.048 |

Mean income 17860 (95% bootstrap CI 17247 - 18485). Shapiro-Wilk p=6.22e-16 (normal at 5%: False).

## 3. Hypothesis tests

- **multi_platform_income**: test=Welch t-test, value=monthly_income, group=multiple_platforms, mean_group1=1.944e+04, mean_group0=1.714e+04, t=3.209, p_value=0.001467, mannwhitney_p=0.001515, cohens_d=0.2977, n1=188, n0=412
- **platform_income_anova**: test=One-way ANOVA, value=monthly_income, group=platform, F=22.91, p_value=2.311e-24, kruskal_p=3.105e-21, eta_squared=0.1882, k_groups=7
- **benefits_vs_platform_chi2**: test=Chi-square independence, a=any_benefit, b=platform, chi2=9.082, dof=6, p_value=0.169, cramers_v=0.123
- **stress_vs_platform_chi2**: test=Chi-square independence, a=financial_stress, b=platform, chi2=4.673, dof=6, p_value=0.5864, cramers_v=0.08825
- **error_rates**: n_per_group=60, alpha=0.05, effect_size_d=0.5, type_I_error_rate=0.052, power=0.7775, type_II_error_rate=0.2225


## 4. Precarity index

Mean 52.1; bands {'Medium': 342, 'Low': 138, 'High': 120}. Sensitivity (500 random weight draws): mean Spearman 0.91, workers changing band 33% on average.

## 5. Worker segments (K-Means)

| segment | monthly_income | income_cv | benefits_count | weekly_hours | dependents | savings_months | income_per_hour | size | stress_rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 26197.510 | 0.200 | 0.410 | 63.640 | 1.340 | 0.990 | 96.610 | 201 | 0.209 |
| 1 | 13659.280 | 0.150 | 0.220 | 52.340 | 1.350 | 0.420 | 61.560 | 399 | 0.373 |

k=2, silhouette 0.23; PCA first two components explain 48% of variance.

## 6. Time series

Average earnings trend 25 INR/month (p=0.725); month-6 back-test MAE: exponential smoothing 432 vs naive 572.

## 7. Machine learning (target = financial_stress)

n=600, prevalence 31.8%.

### 7.1 5-fold CV model comparison

| model | accuracy | precision | recall | f1 | roc_auc | train_roc_auc |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.733 | 0.640 | 0.431 | 0.505 | 0.709 | 0.790 |
| RandomForest | 0.698 | 0.551 | 0.268 | 0.359 | 0.691 | 1.0 |
| GradientBoosting | 0.706 | 0.575 | 0.353 | 0.433 | 0.677 | 0.997 |
| KNN | 0.681 | 0.594 | 0.111 | 0.177 | 0.661 | 0.765 |
| DecisionTree | 0.633 | 0.359 | 0.269 | 0.298 | 0.595 | 0.858 |

### 7.2 Selected model: LogisticRegression (threshold 0.31 = highest threshold keeping CV recall >= 0.70)

Hold-out: accuracy=0.617, precision=0.438, recall=0.737, f1=0.549, roc_auc=0.743, brier=0.187; confusion matrix [[46, 36], [10, 28]]

### 7.3 Imbalance strategies (CV)

| strategy | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| No handling | 0.697 | 0.555 | 0.225 | 0.314 | 0.687 |
| class_weight=balanced | 0.670 | 0.473 | 0.419 | 0.438 | 0.697 |
| SMOTE | 0.680 | 0.493 | 0.325 | 0.386 | 0.679 |

### 7.4 Permutation importance

| feature | importance | sd |
|---|---|---|
| dependents | 0.077 | 0.035 |
| income_cv | 0.040 | 0.015 |
| monthly_income | 0.030 | 0.018 |
| savings_months | 0.020 | 0.015 |
| income_per_hour | 0.020 | 0.026 |
| months_experience | 0.008 | 0.005 |
| multiple_platforms | 0.005 | 0.007 |
| weekly_hours | 0.001 | 0.002 |

### 7.5 Bias-variance: best tree depth by CV = 3; deeper trees overfit (train AUC -> 1.0, CV AUC falls)

### 7.6 Income regression

| model | cv_r2 | test_r2 | test_rmse | test_mae |
|---|---|---|---|---|
| LinearRegression | 0.501 | 0.544 | 5614.849 | 4133.173 |
| RandomForestRegressor | 0.355 | 0.420 | 6334.104 | 4511.708 |

### 7.7 Fairness audit (hold-out; small groups are unreliable)

| gender | n | actual_rate | predicted_rate | tpr | fpr | small_group |
|---|---|---|---|---|---|---|
| Female | 16 | 0.375 | 0.750 | 1.0 | 0.600 | True |
| Male | 101 | 0.317 | 0.485 | 0.688 | 0.391 | False |
| Other | 3 | 0.0 | 1.0 | nan | 1.0 | True |

| city | n | actual_rate | predicted_rate | tpr | fpr | small_group |
|---|---|---|---|---|---|---|
| Bangalore | 19 | 0.263 | 0.421 | 0.600 | 0.357 | True |
| Chennai | 9 | 0.333 | 1.0 | 1.0 | 1.0 | True |
| Delhi | 19 | 0.263 | 0.474 | 0.800 | 0.357 | True |
| Hyderabad | 24 | 0.375 | 0.542 | 0.889 | 0.333 | True |
| Kolkata | 10 | 0.400 | 0.700 | 0.500 | 0.833 | True |
| Mumbai | 18 | 0.333 | 0.444 | 0.833 | 0.250 | True |
| Pune | 21 | 0.286 | 0.476 | 0.500 | 0.467 | True |
