import unittest
import helpers  # noqa
import numpy as np, pandas as pd
from src.monitoring import psi, status, drift_report, needs_retraining


class TestMonitoring(unittest.TestCase):
    def test_psi_same_vs_shifted(self):
        rng = np.random.default_rng(0); a = rng.normal(0, 1, 1000)
        self.assertLess(psi(a, rng.normal(0, 1, 1000)), 0.1)
        self.assertGreater(psi(a, rng.normal(1.5, 1, 1000)), 0.25)

    def test_report_and_retrain_flag(self):
        rng = np.random.default_rng(1)
        ref = pd.DataFrame({c: rng.normal(0, 1, 500) for c in "abcd"}); cur = ref + 2
        rep = drift_report(ref, cur, list("abcd"))
        self.assertTrue(needs_retraining(rep)); self.assertEqual(status(0.3), "significant drift")


if __name__ == "__main__":
    unittest.main()
