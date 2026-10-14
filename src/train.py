"""Train, tune, evaluate and save the churn model.

Run from the project root:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.preprocessing import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES, TARGET, add_features, load_data,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
# Business assumption (relative units): losing a customer costs 10x more than making a retention offer.
COST_MISSED_CHURNER = 500
COST_RETENTION_OFFER = 50

sns.set_theme(style="whitegrid")


def build_pipeline(estimator, scale=False):
    pre = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", pre),
        ("model", estimator),
    ])


def candidate_models():
    return {
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
            {"model__C": [0.1, 1, 10]},
        ),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1)),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [5, 10]},
        ),
        "Gradient Boosting": (
            build_pipeline(HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE)),
            {"model__learning_rate": [0.03, 0.1], "model__max_depth": [3, None],
             "model__max_iter": [150, 300]},
        ),
    }


def save_eda_figures(df):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    df[TARGET].map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Target balance"); ax.set_ylabel("Customers"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_target_balance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.values, y=rate.index, ax=ax, color="#C44E52")
        ax.set_title(f"Churn rate by {col}"); ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_churn_by_category.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
        sns.histplot(data=df.assign(Status=df[TARGET].map({0: "Stayed", 1: "Churned"})),
                     x=col, hue="Status", bins=30, kde=True, stat="density", common_norm=False, ax=ax)
        ax.set_title(f"{col} by churn status")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_numeric_distributions.png", dpi=150); plt.close(fig)


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def business_cost(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return fn * COST_MISSED_CHURNER + (tp + fp) * COST_RETENTION_OFFER


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, churn rate {df[TARGET].mean():.1%}")
    save_eda_figures(df)

    X, y = df[RAW_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Compare tuned models using cross-validated ROC-AUC on the training set only
    results, fitted = {}, {}
    for name, (pipe, grid) in candidate_models().items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        results[name] = {"cv_roc_auc": round(search.best_score_, 4), "best_params": search.best_params_}
        fitted[name] = search.best_estimator_
        print(f"{name:20s} CV ROC-AUC = {search.best_score_:.4f}  {search.best_params_}")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # 2) Choose the decision threshold from out-of-fold training predictions (no test leakage)
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid_t = np.linspace(0.05, 0.95, 91)
    costs = [business_cost(y_train, oof, t) for t in grid_t]
    threshold = float(grid_t[int(np.argmin(costs))])
    print(f"Cost-optimal threshold (from CV): {threshold:.2f}")

    # 3) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    at_default = evaluate(y_test, proba, 0.5)
    at_tuned = evaluate(y_test, proba, threshold)
    baseline_acc = round(max(y_test.mean(), 1 - y_test.mean()), 4)
    cost_default, cost_tuned = business_cost(y_test, proba, 0.5), business_cost(y_test, proba, threshold)
    cost_none = int(y_test.sum() * COST_MISSED_CHURNER)  # do nothing: every churner is lost
    print("Test @0.50:", at_default)
    print(f"Test @{threshold:.2f}:", at_tuned)

    # 4) Figures
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(results); vals = [results[n]["cv_roc_auc"] for n in names]
    sns.barplot(x=vals, y=names, ax=ax, color="#4C72B0")
    ax.set_xlim(min(vals) - 0.03, max(vals) + 0.02); ax.set_xlabel("5-fold CV ROC-AUC")
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center")
    ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_roc_curve.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axes, [0.5, threshold]):
        ConfusionMatrixDisplay.from_predictions(
            y_test, (proba >= t).astype(int), display_labels=["Stayed", "Churned"],
            cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix @ threshold {t:.2f}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(grid_t, costs, color="#C44E52")
    ax.axvline(threshold, ls="--", color="k", alpha=0.6, label=f"chosen = {threshold:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total business cost (CV, train set)")
    ax.set_title("Threshold vs business cost"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "07_threshold_cost.png", dpi=150); plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": RAW_FEATURES, "importance": imp.importances_mean})
              .sort_values("importance", ascending=False))
    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp_df.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#55A868")
    ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("Permutation importance (test set)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150); plt.close(fig)
    print("\nTop drivers:\n", imp_df.head(6).to_string(index=False))

    # 5) Save model + metrics
    joblib.dump({"pipeline": best, "threshold": threshold, "features": RAW_FEATURES}, MODEL_PATH)
    metrics = {
        "dataset": {"rows": int(df.shape[0]), "churn_rate": round(float(y.mean()), 4),
                    "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "baseline_majority_accuracy": baseline_acc,
        "models": results, "best_model": best_name,
        "test_default_threshold": at_default, "test_tuned_threshold": at_tuned,
        "business": {"cost_missed_churner": COST_MISSED_CHURNER, "cost_retention_offer": COST_RETENTION_OFFER,
                     "test_cost_no_model": cost_none, "test_cost_default": int(cost_default),
                     "test_cost_tuned": int(cost_tuned)},
        "top_features": imp_df.head(8).round(4).to_dict("records"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {MODEL_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
