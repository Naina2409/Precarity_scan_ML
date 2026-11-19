import tempfile, unittest
from pathlib import Path
import helpers  # noqa
from src.data_generator import make_clean
from src.features import engineer
from src.models import MODEL_FEATURES, save_versioned, train_and_select
from src import predict

REC = dict(city="Delhi", platform="Zomato", age=28, months_experience=24, hours_per_day=9, days_per_week=6,
           monthly_income=15000, dependents=2, savings_months=0.3, has_health_insurance=False, has_paid_leave=False,
           has_retirement=False, multiple_platforms=False, has_work_loan=True)


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = make_clean(300, seed=9); res = train_and_select(df, do_tune=False)
        cls.dir = Path(tempfile.mkdtemp())
        save_versioned(res["pipeline"], {"roc_auc": .7}, {}, df, engineer(df)[MODEL_FEATURES], models_dir=cls.dir, threshold=res["threshold"])

    def test_predict_shape(self):
        o = predict.predict_one(REC, self.dir, log=False)
        self.assertTrue(0 <= o["stress_probability"] <= 1); self.assertTrue(0 <= o["precarity_index"] <= 100)
        self.assertIn(o["risk_level"], {"Low", "Medium", "High"}); self.assertEqual(len(o["top_drivers"]), 3)

    def test_history_improves_volatility_input(self):
        a = predict.predict_one(REC, self.dir, log=False)
        b = predict.predict_one({**REC, "monthly_incomes": [5000, 25000, 8000, 22000]}, self.dir, log=False)
        self.assertFalse(a["history_used"]); self.assertTrue(b["history_used"])
        self.assertGreater(b["all_components"]["income_volatility"], a["all_components"]["income_volatility"])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError): predict.predict_one({**REC, "days_per_week": 26}, self.dir, log=False)
        bad = dict(REC); bad.pop("age")
        with self.assertRaises(ValueError): predict.predict_one(bad, self.dir, log=False)

    def test_missing_model(self):
        with self.assertRaises(FileNotFoundError): predict.load_model(Path(tempfile.mkdtemp()))


if __name__ == "__main__":
    unittest.main()
