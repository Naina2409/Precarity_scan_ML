# Ethics, compliance and limitations

## Limitations (CO4)
- **Synthetic data:** the shipped results describe a simulator. Real conclusions need real data.
- **Small sample / uncertainty:** with a few hundred rows, hold-out metrics are noisy (hold-out = 120 rows). Report CV standard deviations and CIs, not single numbers.
- **Self-selection and recall bias:** survey respondents are not a random sample; earnings and "could not cover an expense" are self-reported.
- **Correlation, not causation:** importance scores and tests show association only.
- **Index weights are assumptions:** sensitivity analysis shows ~1/3 of workers can change risk band under plausible alternative weights; the 20,000 INR benchmark and the risk bands should be recalibrated on real data.
- **Threshold policy:** the screening threshold favours recall (>= 70% of stressed workers caught), which costs precision (more false alarms).
- **Fairness:** group sizes in a hold-out are tiny; the audit is only meaningful with real, larger data. Gender is excluded from model features, but proxies (city, platform) can still carry bias.
- **Operational:** SQLite/CSV storage, in-memory lockout counters and single-node deployment suit a course project, not production.

## Ethics and compliance (CO5)
- **Consent and minimisation:** worker submissions are stored only after an explicit consent tick, without names or contact details.
- **Privacy law:** worker data falls under India's Digital Personal Data Protection Act, 2023: state the purpose, keep only what is needed, allow deletion, secure storage.
- **Security:** salted PBKDF2 password hashes, lockout, no credentials in the repo, optional API key; add HTTPS and a real database before public use.
- **Responsible use:** the index and model are indicators for self-reflection and research; they must not be used to deny work, pay or credit to individuals.
- **Transparency:** every score shows its component contributions; the model version and training-data hash are recorded in `models/registry.json`.
