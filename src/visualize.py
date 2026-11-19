"""Module 6: report figures (matplotlib / seaborn) saved to reports/figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from .config import FIG_DIR

sns.set_theme(style="whitegrid", context="notebook")

def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); return p.name

def income_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["monthly_income"], bins=30, kde=True, ax=ax[0]); ax[0].set_title("Monthly income distribution")
    order = df.groupby("platform")["monthly_income"].median().sort_values().index
    sns.boxplot(data=df, y="platform", x="monthly_income", order=order, ax=ax[1]); ax[1].set_title("Income by platform")
    return _save(fig, "01_income_distribution.png")

def correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation"); return _save(fig, "02_correlation_heatmap.png")

def pairplot(df, cols):
    g = sns.pairplot(df[cols + ["financial_stress"]].dropna().sample(min(len(df), 300), random_state=1),
                     hue="financial_stress", corner=True, plot_kws={"s": 12, "alpha": .6})
    FIG_DIR.mkdir(parents=True, exist_ok=True); g.savefig(FIG_DIR / "03_pairplot.png", dpi=110); plt.close("all"); return "03_pairplot.png"

def risk_distribution(index, levels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(index, bins=25, ax=ax[0]); ax[0].set_title("Precarity index (0-100)")
    levels.value_counts().reindex(["Low", "Medium", "High"]).plot.bar(ax=ax[1], color=["#4caf50", "#ffb300", "#e53935"])
    ax[1].set_title("Risk bands"); ax[1].set_xlabel("")
    return _save(fig, "04_risk_distribution.png")

def model_comparison(cv):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(cv["model"], cv["roc_auc"], xerr=cv["roc_auc_sd"], color="#1f77b4"); ax.set_xlim(0.4, 1)
    ax.set_title("5-fold CV ROC-AUC by model"); ax.invert_yaxis(); return _save(fig, "05_model_comparison.png")

def roc_and_confusion(test, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(test["roc_curve"]["fpr"], test["roc_curve"]["tpr"], label=f"{name} (AUC={test['roc_auc']:.2f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(); ax[0].set_title("ROC (hold-out)")
    sns.heatmap(np.array(test["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[1],
                xticklabels=["Not stressed", "Stressed"], yticklabels=["Not stressed", "Stressed"])
    ax[1].set_title("Confusion matrix at chosen threshold"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")
    return _save(fig, "06_roc_confusion.png")

def feature_importance(imp):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top["feature"], top["importance"], xerr=top["sd"])
    ax.set_title("Permutation importance (drop in ROC-AUC)"); return _save(fig, "07_feature_importance.png")

def bias_variance(bv):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bv["max_depth"], bv["train_auc"], "o-", label="Train"); ax.plot(bv["max_depth"], bv["cv_auc"], "s-", label="CV")
    ax.set_xlabel("Decision-tree depth"); ax.set_ylabel("ROC-AUC"); ax.legend(); ax.set_title("Bias-variance trade-off (overfitting as depth grows)")
    return _save(fig, "08_bias_variance.png")

def clustering_figs(seg):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    kt = seg["k_table"]; ax[0].plot(kt["k"], kt["silhouette"], "o-", color="tab:green"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Silhouette")
    a2 = ax[0].twinx(); a2.plot(kt["k"], kt["inertia"], "s--", color="tab:blue"); a2.set_ylabel("Inertia"); ax[0].set_title("Choosing k")
    sc = ax[1].scatter(seg["pca_coords"][:, 0], seg["pca_coords"][:, 1], c=seg["labels"], cmap="viridis", s=14)
    ax[1].set_title(f"PCA view of segments (k={seg['k']})"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    n1 = _save(fig, "09_clustering.png")
    fig, ax = plt.subplots(figsize=(9, 4)); dendrogram(seg["linkage"], ax=ax, no_labels=True); ax.set_title("Hierarchical clustering (Ward, 60-worker sample)")
    return n1, _save(fig, "10_dendrogram.png")

def timeseries_fig(tbl):
    fig, ax = plt.subplots(figsize=(8, 4)); tbl.plot(ax=ax, marker="o", linewidth=1.2); ax.plot(tbl.mean(axis=1), "k--", linewidth=2.2, label="All")
    ax.set_xlabel("Month"); ax.set_ylabel("Mean income (INR)"); ax.set_title("Monthly mean earnings by platform"); ax.legend(fontsize=7, ncol=2)
    return _save(fig, "11_timeseries.png")

def fairness_fig(fa, col):
    fig, ax = plt.subplots(figsize=(7, 3.8)); fa.set_index(col)[["actual_rate", "predicted_rate", "tpr"]].plot.bar(ax=ax)
    ax.set_title(f"Fairness audit by {col}"); ax.set_ylim(0, 1.05); return _save(fig, f"12_fairness_{col}.png")
