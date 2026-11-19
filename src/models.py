"""Module 3 and 4: preprocessing pipeline, model comparison, tuning, imbalance handling,
interpretation, fairness audit, regression and model versioning."""
import hashlib, json, datetime as dt
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold,
                                     cross_validate, train_test_split, validation_curve)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .config import MODEL_BOOL, MODEL_CAT, MODEL_FEATURES, MODEL_NUM, MODELS_DIR, SEED, TARGET
from .features import engineer

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


# ---------- data -> X, y ----------
def get_xy(df):
    d = engineer(df)
    d = d[d[TARGET].notna()]
    return d[MODEL_FEATURES], d[TARGET].astype(int), d


def build_preprocessor(num=None, boolean=None, cat=None):
    num = MODEL_NUM if num is None else num
    boolean = MODEL_BOOL if boolean is None else boolean
    cat = MODEL_CAT if cat is None else cat
    parts = []
    if num: parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if boolean: parts.append(("bool", SimpleImputer(strategy="most_frequent"), boolean))
    if cat: parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts)


# ---------- imbalance: SMOTE implemented here (applied only inside fit, so CV stays leak-free) ----------
def smote_resample(X, y, k=5, seed=SEED):
    X, y = np.asarray(X, float), np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2: return X, y
    minority, majority_n = classes[counts.argmin()], counts.max()
    Xm = X[y == minority]
    n_new = majority_n - len(Xm)
    if len(Xm) < 2 or n_new <= 0: return X, y
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xm), n_new)
    pick = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
    synth = Xm[base] + rng.random((n_new, 1)) * (Xm[pick] - Xm[base])
    return np.vstack([X, synth]), np.concatenate([y, np.full(n_new, minority)])


class SMOTEClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, k_neighbors=5, random_state=SEED):
        self.estimator, self.k_neighbors, self.random_state = estimator, k_neighbors, random_state

    def fit(self, X, y):
        Xr, yr = smote_resample(X, y, self.k_neighbors, self.random_state)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X): return self.estimator_.predict(np.asarray(X, float))
    def predict_proba(self, X): return self.estimator_.predict_proba(np.asarray(X, float))


def make_pipeline(est):
    return Pipeline([("pre", build_preprocessor()), ("clf", est)])


def candidate_models(seed=SEED):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }


def cv_compare(X, y, models=None, seed=SEED):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for name, est in (models or candidate_models(seed)).items():
        r = cross_validate(make_pipeline(est), X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=-1)
        row = {"model": name}
        for m in SCORING:
            row[m] = r[f"test_{m}"].mean(); row[m + "_sd"] = r[f"test_{m}"].std()
        row["train_roc_auc"] = r["train_roc_auc"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def imbalance_comparison(X, y, seed=SEED):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rf = lambda **k: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, **k)
    setups = {"No handling": rf(), "class_weight=balanced": rf(class_weight="balanced"),
              "SMOTE": SMOTEClassifier(rf(), random_state=seed)}
    rows = []
    for name, est in setups.items():
        r = cross_validate(make_pipeline(est), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({"strategy": name, **{m: r[f"test_{m}"].mean() for m in SCORING}})
    return pd.DataFrame(rows)


def tune(X, y, seed=SEED):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    searches = {
        "LogisticRegression": GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000, random_state=seed)),
                                           {"clf__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="roc_auc", n_jobs=-1),
        "RandomForest": GridSearchCV(make_pipeline(RandomForestClassifier(random_state=seed, n_jobs=-1)),
                                     {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 8, None],
                                      "clf__min_samples_leaf": [1, 5]}, cv=cv, scoring="roc_auc", n_jobs=-1),
        "GradientBoosting": RandomizedSearchCV(make_pipeline(GradientBoostingClassifier(random_state=seed)),
                                               {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.03, 0.05, 0.1],
                                                "clf__max_depth": [2, 3, 4], "clf__subsample": [0.7, 1.0]},
                                               n_iter=8, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed),
    }
    out = {}
    for name, s in searches.items():
        s.fit(X, y)
        out[name] = {"cv_roc_auc": float(s.best_score_), "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in s.best_params_.items()},
                     "estimator": s.best_estimator_}
    return out


def evaluate(pipe, X, y, threshold=0.5):
    p = pipe.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}, "proba": p, "pred": pred}


def choose_threshold(pipe, X, y, target_recall=0.70, seed=SEED):
    """Operating policy: on out-of-fold training predictions, pick the highest threshold that still
    catches at least `target_recall` of financially stressed workers (best precision at that recall).
    Failing to flag a vulnerable worker is treated as costlier than a false alarm."""
    p = cross_val_predict(clone(pipe), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba", n_jobs=-1)[:, 1]
    best = 0.10
    for t in np.round(np.arange(0.10, 0.80, 0.01), 2):
        if recall_score(y, (p >= t).astype(int)) >= target_recall:
            best = float(t)
    return best


def bias_variance_curve(X, y, depths=range(1, 15), seed=SEED):
    pipe = make_pipeline(DecisionTreeClassifier(random_state=seed))
    tr, te = validation_curve(pipe, X, y, param_name="clf__max_depth", param_range=list(depths),
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=-1)
    return pd.DataFrame({"max_depth": list(depths), "train_auc": tr.mean(axis=1), "cv_auc": te.mean(axis=1)})


def importance(pipe, X, y, seed=SEED, n_repeats=10):
    r = permutation_importance(pipe, X, y, n_repeats=n_repeats, random_state=seed, scoring="roc_auc", n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "sd": r.importances_std}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)


def shap_values_if_available(pipe, X):
    """Optional: SHAP for tree models if the `shap` package is installed."""
    try:
        import shap
    except ImportError:
        return None
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X)
    return shap.TreeExplainer(clf).shap_values(Xt)


def fairness_audit(d_test, proba, y, group_col, threshold=0.5, min_n=30):
    pred = (proba >= threshold).astype(int)
    rows = []
    for g, idx in d_test.groupby(group_col).groups.items():
        pos = d_test.index.get_indexer(idx)
        yt, yp = np.asarray(y)[pos], pred[pos]
        tp = ((yt == 1) & (yp == 1)).sum(); fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum(); tn = ((yt == 0) & (yp == 0)).sum()
        rows.append({group_col: g, "n": len(pos), "actual_rate": yt.mean(), "predicted_rate": yp.mean(),
                     "tpr": tp / (tp + fn) if tp + fn else np.nan, "fpr": fp / (fp + tn) if fp + tn else np.nan,
                     "small_group": len(pos) < min_n})
    return pd.DataFrame(rows)


# ---------- regression (income) ----------
def income_regression(df, seed=SEED):
    d = engineer(df)
    feats = ["age", "months_experience", "hours_per_day", "days_per_week", "dependents", "platform", "city"]
    X, y = d[feats], d["monthly_income"]
    pre = lambda: build_preprocessor(["age", "months_experience", "hours_per_day", "days_per_week", "dependents"], [], ["platform", "city"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    rows = []
    for name, est in {"LinearRegression": LinearRegression(),
                      "RandomForestRegressor": RandomForestRegressor(200, min_samples_leaf=5, random_state=seed, n_jobs=-1)}.items():
        p = Pipeline([("pre", pre()), ("reg", est)])
        cvr2 = cross_validate(p, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2", n_jobs=-1)["test_score"].mean()
        p.fit(Xtr, ytr); pr = p.predict(Xte)
        rows.append({"model": name, "cv_r2": cvr2, "test_r2": r2_score(yte, pr),
                     "test_rmse": float(np.sqrt(mean_squared_error(yte, pr))), "test_mae": mean_absolute_error(yte, pr)})
    return pd.DataFrame(rows)


# ---------- versioned persistence ----------
def data_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def save_versioned(pipe, metrics, params, df, reference, models_dir=MODELS_DIR, note="", threshold=0.5):
    models_dir.mkdir(parents=True, exist_ok=True)
    version = dt.datetime.now(dt.timezone.utc).strftime("v%Y%m%d%H%M%S")
    path = models_dir / f"model_{version}.joblib"
    joblib.dump({"pipeline": pipe, "features": MODEL_FEATURES, "version": version, "threshold": threshold}, path)
    reference.sample(min(len(reference), 1000), random_state=SEED).to_csv(models_dir / f"reference_{version}.csv", index=False)
    entry = {"version": version, "file": path.name, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "data_hash": data_hash(df), "n_rows": int(len(df)), "metrics": metrics, "params": params,
             "sklearn_version": sklearn.__version__, "note": note}
    reg_path = models_dir / "registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else []
    registry.append(entry); reg_path.write_text(json.dumps(registry, indent=2))
    (models_dir / "latest.json").write_text(json.dumps({"version": version, "file": path.name}))
    return entry


def train_and_select(df, seed=SEED, do_tune=True):
    X, y, d = get_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    result = {"n": int(len(X)), "prevalence": float(y.mean()), "cv_table": cv_compare(Xtr, ytr, seed=seed)}
    if do_tune:
        tuned = tune(Xtr, ytr, seed)
        best_name = max(tuned, key=lambda k: tuned[k]["cv_roc_auc"])
        best = tuned[best_name]["estimator"]
        result["tuned"] = {k: {kk: vv for kk, vv in v.items() if kk != "estimator"} for k, v in tuned.items()}
    else:
        best_name = "RandomForest"; best = make_pipeline(candidate_models(seed)[best_name]).fit(Xtr, ytr)
    result["best_name"] = best_name
    result["threshold"] = choose_threshold(best, Xtr, ytr, seed=seed)
    result["test"] = evaluate(best, Xte, yte, threshold=result["threshold"])
    result["test_at_0.5"] = evaluate(best, Xte, yte, threshold=0.5)
    result["pipeline"] = best
    result.update({"Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "d_test": d.loc[Xte.index]})
    return result
