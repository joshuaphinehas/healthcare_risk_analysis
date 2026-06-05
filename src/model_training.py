"""
model_training.py
-----------------
Trains multiple classifiers, handles class imbalance via SMOTE,
performs hyperparameter tuning, evaluates models, and saves the best one.
"""

import os
import warnings
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASSES = ["Low Risk", "Medium Risk", "High Risk"]


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA SPLITTING
# ═════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2):
    """
    Split into train / validation / test sets (60 / 20 / 20).
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size * 2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═════════════════════════════════════════════════════════════════════════════
# 2.  CLASS IMBALANCE — SMOTE
# ═════════════════════════════════════════════════════════════════════════════

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    Only applied to the training set to prevent data leakage.
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE → {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITIONS & HYPERPARAMETER GRIDS
# ═════════════════════════════════════════════════════════════════════════════

def get_models_and_params() -> dict:
    """Return a dict of (model, param_grid) pairs for GridSearchCV."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 8], "min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            {"max_depth": [5, 10, None], "min_samples_split": [2, 5]},
        ),
        "XGBoost": (
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            ),
            {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING WITH GRIDSEARCH + CV
# ═════════════════════════════════════════════════════════════════════════════

def train_all_models(X_train, y_train, X_val, y_val, cv_folds: int = 5) -> dict:
    """
    Train every classifier with GridSearchCV, evaluate on validation set,
    and return a results dictionary.
    """
    models_params = get_models_and_params()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, (model, params) in models_params.items():
        print(f"\n  Training: {name} …", end=" ")
        gs = GridSearchCV(model, params, cv=cv, scoring="f1_weighted", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        y_pred = best.predict(X_val)
        y_prob = best.predict_proba(X_val)
        y_val_bin = label_binarize(y_val, classes=[0, 1, 2])

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        roc = roc_auc_score(y_val_bin, y_prob, multi_class="ovr", average="weighted")

        results[name] = {
            "model": best,
            "best_params": gs.best_params_,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        print(f"Done | Acc={acc:.3f} | F1={f1:.3f} | AUC={roc:.3f}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str) -> None:
    path = os.path.join(REPORTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(results: dict) -> None:
    """Bar chart comparing all models on key metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(results.keys())
    data = {m: [results[n][m] for n in names] for m in metrics}

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    colors = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12"]
    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        bars = ax.bar(names, data[metric], color=colors[i], alpha=0.85)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for bar, val in zip(bars, data[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8)
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_model_comparison")


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Heatmap confusion matrix for the best model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    _save(fig, "09_confusion_matrix")


def plot_roc_curves(y_true, y_prob, model_name: str) -> None:
    """Multi-class ROC curves (one-vs-rest) for the best model."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ["#2ECC71", "#F39C12", "#E74C3C"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    _save(fig, "10_roc_curves")


def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Bar chart of feature importances (tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} has no feature_importances_ — skipping.")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="#3498DB", alpha=0.85,
    )
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    _save(fig, "11_feature_importance")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  SELECT & SAVE BEST MODEL
# ═════════════════════════════════════════════════════════════════════════════

def select_best_model(results: dict) -> tuple:
    """Select the model with the highest weighted F1 on the validation set."""
    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    print(f"\n✔  Best model: {best_name}  (F1={best['f1']:.4f})")
    return best_name, best["model"]


def save_model(model, scaler, feature_names: list) -> None:
    """Persist model, scaler, and feature names to disk."""
    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print("  Model, scaler & feature names saved.")


def load_artifacts():
    """Load the saved model artifacts."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


# ═════════════════════════════════════════════════════════════════════════════
# 7.  HEALTHCARE-SPECIFIC NOTE — WHY RECALL MATTERS
# ═════════════════════════════════════════════════════════════════════════════

RECALL_NOTE = """
Why Recall Is Critical in Healthcare Risk Prediction
─────────────────────────────────────────────────────
In a healthcare setting, a False Negative (predicting Low Risk when the
patient is actually High Risk) is far more dangerous than a False Positive.

• A missed High-Risk patient may not receive timely intervention, leading to
  preventable hospitalisation or death.
• A patient wrongly flagged Medium/High Risk only undergoes extra diagnostic
  tests — an inconvenience, not a catastrophe.

Therefore, we optimise for HIGH RECALL on the High Risk class, even if it
slightly reduces overall precision. Practically:
  - Use class_weight='balanced' or SMOTE to correct imbalance.
  - Monitor per-class recall in the classification report.
  - Consider a lower probability threshold (e.g. 0.4 instead of 0.5) to
    catch more true positives.
"""