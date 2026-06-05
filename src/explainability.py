"""
explainability.py
-----------------
SHAP-based explainability for the best trained model.
Provides global (dataset-level) and local (patient-level) explanations.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed. Run: pip install shap")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(REPORTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Build the correct SHAP explainer depending on the model type.

    Tree-based (RF, XGB, DT) → TreeExplainer  (fast, exact)
    Others (LR, SVM)         → KernelExplainer (model-agnostic, slower)
    """
    if not SHAP_AVAILABLE:
        return None

    model_name = type(model).__name__
    tree_models = ("RandomForestClassifier", "XGBClassifier", "DecisionTreeClassifier",
                   "GradientBoostingClassifier")

    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
    else:
        # Use a small background sample for speed
        background = shap.sample(X_background, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(explainer, X: pd.DataFrame):
    """Compute SHAP values for the given dataset."""
    if explainer is None:
        return None
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, class_index: int = 2) -> None:
    """
    SHAP summary (beeswarm) plot for the HIGH RISK class (index 2).
    Shows which features push predictions toward High Risk.
    """
    if shap_values is None or not SHAP_AVAILABLE:
        print("  SHAP not available — skipping summary plot.")
        return

    # shap_values may be a list (one per class) or a 3-D array
    if isinstance(shap_values, list):
        sv = shap_values[class_index]
    else:
        sv = shap_values[:, :, class_index]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X, show=False, max_display=15, plot_type="dot")
    plt.title("SHAP Summary — High Risk Class", fontsize=13, fontweight="bold")
    _save(plt.gcf(), "12_shap_summary")


def plot_shap_bar(shap_values, X: pd.DataFrame, class_index: int = 2) -> None:
    """Mean absolute SHAP bar chart (global feature importance)."""
    if shap_values is None or not SHAP_AVAILABLE:
        return

    if isinstance(shap_values, list):
        sv = shap_values[class_index]
    else:
        sv = shap_values[:, :, class_index]

    mean_abs = np.abs(sv).mean(axis=0)
    feature_importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(feature_importance)))
    feature_importance.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Top Features — Mean |SHAP| for High Risk", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    _save(fig, "13_shap_bar")


def explain_single_patient(
    explainer,
    shap_values,
    X: pd.DataFrame,
    patient_index: int = 0,
    class_index: int = 2,
) -> dict:
    """
    Return a human-readable explanation for a single patient.

    Returns
    -------
    dict with:
        features      : feature values for the patient
        shap_contrib  : SHAP contribution per feature (for High Risk class)
        top_factors   : top 5 risk-increasing factors
        top_protectors: top 3 risk-decreasing factors
    """
    if shap_values is None or not SHAP_AVAILABLE:
        return {}

    if isinstance(shap_values, list):
        sv = shap_values[class_index][patient_index]
    else:
        sv = shap_values[patient_index, :, class_index]

    patient_row = X.iloc[patient_index]
    contributions = pd.Series(sv, index=X.columns).sort_values()

    explanation = {
        "features": patient_row.to_dict(),
        "shap_contrib": contributions.to_dict(),
        "top_factors": contributions.nlargest(5).to_dict(),        # push toward High Risk
        "top_protectors": contributions.nsmallest(3).to_dict(),    # push away from High Risk
    }
    return explanation


def print_patient_explanation(explanation: dict, predicted_class: str) -> None:
    """Pretty-print a patient's risk explanation."""
    if not explanation:
        return
    print(f"\n  ── Patient Risk Explanation ──────────────────────────")
    print(f"  Predicted class: {predicted_class}")
    print(f"\n  Top risk-INCREASING factors (push toward High Risk):")
    for feat, val in explanation["top_factors"].items():
        print(f"    {feat:<30} SHAP = +{val:.4f}")
    print(f"\n  Top risk-DECREASING factors (push away from High Risk):")
    for feat, val in explanation["top_protectors"].items():
        print(f"    {feat:<30} SHAP = {val:.4f}")
    print("  ─────────────────────────────────────────────────────\n")