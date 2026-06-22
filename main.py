"""
5G User Prediction — multi-model training and AUC evaluation.

Supported models: LightGBM, Logistic Regression, Random Forest.
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Font fallback for CJK labels on Windows (report figures may use Chinese elsewhere)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- Paths and reproducibility ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
N_SPLITS = 5  # folds for optional cross-validation helper
TEST_SIZE = 0.2


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """Load CSV and split columns into features (X), label (y), and column groups."""
    df = pd.read_csv(path)
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    num_cols = [c for c in df.columns if c.startswith("num_")]
    feature_cols = cat_cols + num_cols

    X = df[feature_cols].copy()
    y = df["target"].astype(int)
    return X, y, cat_cols, num_cols


def run_eda(df_path: Path, y: pd.Series, cat_cols: list, num_cols: list) -> None:
    """Generate EDA figures for the assignment report."""
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Target class distribution (highly imbalanced)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts()
    sns.barplot(
        x=counts.index.map({0: "Non-5G", 1: "5G User"}),
        y=counts.values,
        ax=ax,
        palette="Set2",
    )
    ax.set_title("Target Distribution (Class Imbalance)")
    ax.set_ylabel("Sample Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(y) * 100:.2f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "target_distribution.png", dpi=150)
    plt.close()

    # Pearson correlation of numerical features with target
    df = pd.read_csv(df_path, usecols=num_cols + ["target"])
    corr = df.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
    top_corr = corr.head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_corr.values]
    top_corr.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Top-15 Numerical Features Correlated with Target")
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig(fig_dir / "num_feature_correlation.png", dpi=150)
    plt.close()

    # Positive rate (5G share) for the first 8 categorical features
    df_cat = pd.read_csv(df_path, usecols=cat_cols[:8] + ["target"])
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for i, col in enumerate(cat_cols[:8]):
        rate = df_cat.groupby(col)["target"].mean()
        rate.head(20).plot(kind="bar", ax=axes[i], color="#2ecc71", width=0.8)
        axes[i].set_title(f"{col}: 5G Rate by Category")
        axes[i].tick_params(axis="x", rotation=45, labelsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "cat_feature_target_rate.png", dpi=150)
    plt.close()


def prepare_for_sklearn(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict]:
    """
    Encode categoricals with LabelEncoder and scale numerical columns.
    Used by Logistic Regression and Random Forest (sklearn tree API).
    """
    X_tr = X_train.copy()
    X_va = X_val.copy()
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_tr[col], X_va[col]]).astype(str)
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_va[col] = le.transform(X_va[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
    X_va[num_cols] = scaler.transform(X_va[num_cols])

    return X_tr, X_va, scaler, encoders


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list,
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    """Train LightGBM with native categorical features and positive-class weighting."""
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    # Up-weight the minority (5G) class: neg_count / pos_count
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
    )
    y_pred = model.predict_proba(X_val)[:, 1]
    return model, y_pred


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
) -> tuple[LogisticRegression, np.ndarray, StandardScaler, dict]:
    """Train a balanced Logistic Regression baseline on encoded features."""
    X_tr, X_va, scaler, encoders = prepare_for_sklearn(X_train, X_val, cat_cols, num_cols)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        C=0.1,
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict_proba(X_va)[:, 1]
    return model, y_pred, scaler, encoders


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
) -> tuple[RandomForestClassifier, np.ndarray]:
    """Train Random Forest with balanced class weights on encoded features."""
    X_tr, X_va, _, _ = prepare_for_sklearn(X_train, X_val, cat_cols, num_cols)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict_proba(X_va)[:, 1]
    return model, y_pred


def plot_roc_curves(results: dict, y_val: pd.Series) -> None:
    """Plot ROC curves for all models on the validation set."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_pred in results.items():
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        score = roc_auc_score(y_val, y_pred)
        ax.plot(fpr, tpr, label=f"{name} (AUC={score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "roc_curves.png", dpi=150)
    plt.close()


def plot_feature_importance(model: lgb.LGBMClassifier, feature_names: list) -> None:
    """Plot top-20 LightGBM feature importances."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    top = importance.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#9b59b6")
    ax.set_title("LightGBM Top-20 Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "feature_importance.png", dpi=150)
    plt.close()


def cross_validate_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list,
) -> float:
    """Optional helper: stratified K-fold mean AUC for LightGBM."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        _, y_pred = train_lightgbm(X_train, y_train, X_val, y_val, cat_cols)
        scores.append(roc_auc_score(y_val, y_pred))

    return float(np.mean(scores))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    print("=" * 60)
    print("5G User Prediction — Training & Evaluation")
    print("=" * 60)

    X, y, cat_cols, num_cols = load_data(DATA_PATH)
    print(f"Dataset: {len(X):,} samples, {X.shape[1]} features")
    print(f"Positive rate: {y.mean():.4f} ({y.sum():,} / {len(y):,})")

    run_eda(DATA_PATH, y, cat_cols, num_cols)
    print("EDA figures saved to outputs/figures/")

    # Stratified split preserves the ~1.3% positive rate in both splits
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    feature_names = list(X.columns)

    results = {}
    metrics = {}

    print("\n[1/3] Training LightGBM ...")
    lgb_model, pred_lgb = train_lightgbm(X_train, y_train, X_val, y_val, cat_cols)
    auc_lgb = roc_auc_score(y_val, pred_lgb)
    results["LightGBM"] = pred_lgb
    metrics["LightGBM"] = {"AUC": auc_lgb}
    print(f"  Validation AUC: {auc_lgb:.4f}")

    print("\n[2/3] Training Logistic Regression ...")
    lr_model, pred_lr, _, _ = train_logistic_regression(
        X_train, y_train, X_val, cat_cols, num_cols
    )
    auc_lr = roc_auc_score(y_val, pred_lr)
    results["Logistic Regression"] = pred_lr
    metrics["Logistic Regression"] = {"AUC": auc_lr}
    print(f"  Validation AUC: {auc_lr:.4f}")

    print("\n[3/3] Training Random Forest ...")
    rf_model, pred_rf = train_random_forest(X_train, y_train, X_val, cat_cols, num_cols)
    auc_rf = roc_auc_score(y_val, pred_rf)
    results["Random Forest"] = pred_rf
    metrics["Random Forest"] = {"AUC": auc_rf}
    print(f"  Validation AUC: {auc_rf:.4f}")

    plot_roc_curves(results, y_val)
    plot_feature_importance(lgb_model, feature_names)

    # Report metrics for the best model at threshold 0.5
    best_name = max(metrics, key=lambda k: metrics[k]["AUC"])
    best_pred = results[best_name]
    best_auc = metrics[best_name]["AUC"]
    y_pred_label = (best_pred >= 0.5).astype(int)

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_val, y_pred_label, target_names=["Non-5G", "5G"]))

    # Persist metrics and validation predictions
    summary = {
        "data_size": len(X),
        "positive_rate": float(y.mean()),
        "validation_metrics": {k: {"AUC": float(v["AUC"])} for k, v in metrics.items()},
        "best_model": best_name,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {
            "y_true": y_val.values,
            **{f"pred_{k}": v for k, v in results.items()},
        }
    ).to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
