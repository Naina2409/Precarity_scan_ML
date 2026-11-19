import unittest
import helpers  # noqa
import numpy as np, pandas as pd
from src.stats_tests import bootstrap_ci, welch_ttest, error_rates_simulation, chi_square, cohens_d


class TestStats(unittest.TestCase):
    def test_bootstrap_contains_mean(self):
        x = np.random.default_rng(0).normal(50, 5, 300)
        r = bootstrap_ci(x, n_boot=500)
        self.assertLess(r["ci_low"], 50.5); self.assertGreater(r["ci_high"], 49.5)

    def test_welch_detects_difference(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"v": np.r_[rng.normal(10, 1, 100), rng.normal(12, 1, 100)], "g": [0] * 100 + [1] * 100})
        r = welch_ttest(df, "v", "g")
        self.assertLess(r["p_value"], 0.001); self.assertGreater(r["cohens_d"], 1)

    def test_type1_error_near_alpha(self):
        r = error_rates_simulation(n=40, n_sim=600)
        self.assertAlmostEqual(r["type_I_error_rate"], 0.05, delta=0.03)
        self.assertAlmostEqual(r["power"] + r["type_II_error_rate"], 1.0)

    def test_chi_square_independent(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"a": rng.integers(0, 2, 400), "b": rng.integers(0, 3, 400)})
        self.assertGreater(chi_square(df, "a", "b")["p_value"], 0.01)

    def test_cohens_d_zero(self):
        self.assertAlmostEqual(cohens_d([1, 2, 3], [1, 2, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
