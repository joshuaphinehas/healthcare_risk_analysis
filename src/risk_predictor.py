"""
risk_predictor.py
-----------------
Reusable predict_risk() function that returns predicted class,
probability score, and risk category for any patient input.
"""

import numpy as np
import pandas as pd

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {"Low Risk": "#2ECC71", "Medium Risk": "#F39C12", "High Risk": "#E74C3C"}


def predict_risk(model, scaler, feature_names: list, patient_data: dict) -> dict:
    """
    Classify a single patient's health risk.

    Parameters
    ----------
    model         : trained sklearn/xgb classifier
    scaler        : fitted StandardScaler
    feature_names : list of features the model was trained on
    patient_data  : dict of raw (unscaled) feature values

    Returns
    -------
    dict:
        predicted_class  : 0 / 1 / 2
        predicted_label  : 'Low Risk' / 'Medium Risk' / 'High Risk'
        probability      : probability of the predicted class
        class_probs      : probabilities for all three classes
        risk_category    : 'Low' / 'Medium' / 'High'  (threshold-based)
    """
    # ── Build input DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame([patient_data])

    # Ensure columns match training order; fill missing with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # ── Scale the features that the scaler was fitted on ──────────────────────
    scale_cols = [c for c in ["age", "bmi", "blood_pressure", "cholesterol", "glucose",
                               "lifestyle_risk_score"] if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_class = int(model.predict(df)[0])
    class_probs = model.predict_proba(df)[0]
    pred_prob = float(class_probs[pred_class])
    high_risk_prob = float(class_probs[2])          # probability of High Risk

    # ── Probability-based risk category ──────────────────────────────────────
    if high_risk_prob < 0.30:
        risk_category = "Low"
    elif high_risk_prob < 0.70:
        risk_category = "Medium"
    else:
        risk_category = "High"

    return {
        "predicted_class": pred_class,
        "predicted_label": RISK_LABELS[pred_class],
        "probability": pred_prob,
        "high_risk_probability": high_risk_prob,
        "class_probs": {
            "Low Risk": float(class_probs[0]),
            "Medium Risk": float(class_probs[1]),
            "High Risk": float(class_probs[2]),
        },
        "risk_category": risk_category,
        "risk_color": RISK_COLORS[RISK_LABELS[pred_class]],
    }


def batch_predict(model, scaler, feature_names: list, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run predict_risk on every row of a DataFrame.
    Returns the original DataFrame with extra columns appended.
    """
    results = [
        predict_risk(model, scaler, feature_names, row.to_dict())
        for _, row in df_raw.iterrows()
    ]
    out = df_raw.copy()
    out["predicted_label"] = [r["predicted_label"] for r in results]
    out["high_risk_probability"] = [r["high_risk_probability"] for r in results]
    out["risk_category"] = [r["risk_category"] for r in results]
    return out