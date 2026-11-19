import unittest
import helpers  # noqa
import numpy as np
from src.data_cleaner import clean_dataframe, standardise_columns
from src.data_generator import make_clean, make_messy


class TestCleaner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = make_clean(200, seed=7)
        cls.messy = make_messy(cls.clean, seed=7)
        cls.out, cls.rep = clean_dataframe(cls.messy)

    def test_duplicates_removed(self):
        self.assertEqual(self.rep["duplicates_removed"], 8)
        self.assertEqual(len(self.out), 200)

    def test_invalid_values_flagged_not_kept(self):
        self.assertLessEqual(self.out["days_per_week"].max(), 7)
        self.assertGreaterEqual(self.out["age"].min(), 16)
        self.assertEqual(self.rep["out_of_range_set_to_missing"].get("days_per_week"), 6)

    def test_no_missing_and_standard_names(self):
        self.assertEqual(int(self.out.drop(columns=["financial_stress"]).isna().sum().sum()), 0)
        for c in ["worker_id", "hours_per_day", "monthly_income", "has_health_insurance"]:
            self.assertIn(c, self.out.columns)

    def test_categories_standardised(self):
        self.assertTrue(set(self.out["city"]) <= {"Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"})
        self.assertNotIn("Swigy", set(self.out["platform"]))

    def test_currency_strings_parsed(self):
        self.assertTrue(np.issubdtype(self.out["monthly_income"].dtype, np.number))

    def test_no_fuzzy_substring_mapping(self):
        import pandas as pd
        df = pd.DataFrame({"paid_bonus": [1], "average_rating": [4.0]})
        _, mp = standardise_columns(df)
        self.assertEqual(mp, {})   # old version mapped these to worker_id / age by substring


if __name__ == "__main__":
    unittest.main()
