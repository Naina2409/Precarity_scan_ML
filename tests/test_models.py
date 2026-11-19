import tempfile, unittest, json
from pathlib import Path
import helpers  # noqa
import numpy as np
from src.data_generator import make_clean
from src.features import engineer
from src.models import (MODEL_FEATURES, SMOTEClassifier, choose_threshold, get_xy, make_pipeline, save_versioned,
                        smote_resample, train_and_select, fairness_audit)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_clean(300, seed=5)
        cls.res = train_and_select(cls.df, do_tune=False)

    def test_smote_balances_classes(self):
        X = np.random.default_rng(0).normal(size=(100, 3)); y = np.r_[np.ones(20), np.zeros(80)]
        Xr, yr = smote_resample(X, y)
        self.assertEqual(int(yr.sum()), 80); self.assertEqual(len(Xr), 160)

    def test_smote_only_in_training_fold(self):
        X, y, _ = get_xy(self.df)
        est = SMOTEClassifier(RandomForestClassifier(30, random_state=0))
        s = cross_val_score(make_pipeline(est), X, y, cv=3, scoring="roc_auc")
        self.assertTrue(np.isfinite(s).all())

    def test_training_output(self):
        r = self.res
        self.assertGreater(r["test"]["roc_auc"], 0.55)
        self.assertTrue(0.1 <= r["threshold"] <= 0.8)
        self.assertEqual(len(r["cv_table"]), 5)

    def test_gender_not_a_feature(self):
        self.assertNotIn("gender", MODEL_FEATURES)

    def test_registry_and_versioning(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            e = save_versioned(self.res["pipeline"], {"roc_auc": 0.7}, {}, self.df, engineer(self.df)[MODEL_FEATURES], models_dir=t)
            self.assertTrue((t / e["file"]).exists())
            self.assertEqual(json.loads((t / "latest.json").read_text())["version"], e["version"])
            self.assertEqual(len(json.loads((t / "registry.json").read_text())), 1)

    def test_fairness_table(self):
        r = self.res
        fa = fairness_audit(r["d_test"], r["test"]["proba"], r["yte"], "city", r["threshold"])
        self.assertEqual(int(fa["n"].sum()), len(r["yte"]))


if __name__ == "__main__":
    unittest.main()
