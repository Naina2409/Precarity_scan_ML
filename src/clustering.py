"""Module 3: unsupervised learning (PCA, K-Means, hierarchical clustering) for worker segments."""
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from .config import SEED
from .features import engineer

SEG_FEATURES = ["monthly_income", "income_cv", "benefits_count", "weekly_hours", "dependents", "savings_months", "income_per_hour"]


def prepare(df):
    d = engineer(df)
    X = d[SEG_FEATURES].copy()
    X = X.fillna(X.median())
    return d, pd.DataFrame(StandardScaler().fit_transform(X), columns=SEG_FEATURES, index=X.index)


def choose_k(Xs, k_range=range(2, 7), seed=SEED):
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=seed).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_)})
    return pd.DataFrame(rows)


def segment(df, k=None, seed=SEED):
    d, Xs = prepare(df)
    ks = choose_k(Xs, seed=seed)
    k = k or int(ks.loc[ks["silhouette"].idxmax(), "k"])
    km = KMeans(k, n_init=10, random_state=seed).fit(Xs)
    hc = AgglomerativeClustering(k, linkage="ward").fit(Xs)
    pca = PCA(n_components=min(5, Xs.shape[1])).fit(Xs)
    coords = pca.transform(Xs)[:, :2]
    d = d.copy(); d["segment"] = km.labels_
    profile = d.groupby("segment")[SEG_FEATURES].mean().round(2)
    profile["size"] = d.groupby("segment").size()
    if "financial_stress" in d and d["financial_stress"].notna().any():
        profile["stress_rate"] = d.groupby("segment")["financial_stress"].apply(lambda s: s.astype(float).mean()).round(3)
    return {"k": k, "k_table": ks, "labels": km.labels_, "profile": profile, "pca_coords": coords,
            "pca_explained": pca.explained_variance_ratio_.tolist(),
            "pca_loadings": pd.DataFrame(pca.components_[:2].T, index=SEG_FEATURES, columns=["PC1", "PC2"]).round(3),
            "ari_kmeans_vs_hier": float(adjusted_rand_score(km.labels_, hc.labels_)),
            "silhouette": float(silhouette_score(Xs, km.labels_)), "linkage": linkage(Xs.sample(min(len(Xs), 60), random_state=seed), "ward"),
            "data": d}
