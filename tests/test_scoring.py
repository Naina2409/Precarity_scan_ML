import unittest
import helpers  # noqa
import numpy as np
from src.config import INDEX_WEIGHTS
from src.data_generator import make_clean
from src.features import engineer
from src.scoring import precarity_index, risk_level, sensitivity_analysis, explain, pca_weights


class TestScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_clean(200, seed=3)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(INDEX_WEIGHTS.values()), 1.0)

    def test_index_range(self):
        s = precarity_index(self.df)
        self.assertTrue(((s >= 0) & (s <= 100)).all())

    def test_lower_income_means_higher_index(self):
        a = self.df.iloc[[0]].copy(); b = a.copy()
        a["monthly_income"] = 40000; b["monthly_income"] = 8000
        self.assertGreater(precarity_index(b).iloc[0], precarity_index(a).iloc[0])

    def test_explain_sums_to_index(self):
        d = engineer(self.df)
        np.testing.assert_allclose(explain(d).sum(axis=1), precarity_index(d), atol=1e-9)

    def test_sensitivity_and_pca(self):
        r = sensitivity_analysis(self.df, n_draws=30)
        self.assertGreater(r["mean_spearman"], 0.5)
        self.assertAlmostEqual(sum(pca_weights(self.df)["weights"].values()), 1.0, places=1)

    def test_risk_levels(self):
        import pandas as pd
        self.assertEqual(list(risk_level(pd.Series([10, 50, 90]))), ["Low", "Medium", "High"])


if __name__ == "__main__":
    unittest.main()
