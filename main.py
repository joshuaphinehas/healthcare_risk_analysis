"""
main.py
-------
End-to-end pipeline:
  1. Generate synthetic dataset
  2. Preprocess & engineer features
  3. Train & evaluate all models
  4. Save the best model + scaler
  5. Run SHAP explainability
  6. Demo predict_risk()
"""

import os
import sys

# Allow relative imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from sklearn.metrics import classification_report

from src.data_generator import generate_healthcare_dataset
from src.preprocessing import run_preprocessing_pipeline
from src.model_training import (
    split_data, apply_smote, train_all_models,
    plot_model_comparison, plot_confusion_matrix,
    plot_roc_curves, plot_feature_importance,
    select_best_model, save_model, RECALL_NOTE,
)
from src.explainability import (
    get_shap_explainer, compute_shap_values,
    plot_shap_summary, plot_shap_bar,
    explain_single_patient, print_patient_explanation,
)
from src.risk_predictor import predict_risk


def main() -> None:
    print("\n" + "=" * 70)
    print("   INTELLIGENT HEALTHCARE RISK STRATIFICATION SYSTEM")
    print("=" * 70)

    # ── STEP 1 : Generate & Save Dataset ──────────────────────────────────────
    print("\n[STEP 1] Generating synthetic dataset …")
    os.makedirs("data", exist_ok=True)
    raw_df = generate_healthcare_dataset(n_samples=5000)
    raw_df.to_csv("data/healthcare_data.csv", index=False)
    print(f"  Dataset saved → data/healthcare_data.csv  ({raw_df.shape})")

    # ── STEP 2 : Preprocess + EDA + Feature Engineering ──────────────────────
    print("\n[STEP 2] Preprocessing pipeline …")
    df_clean, scaler, scale_cols = run_preprocessing_pipeline(raw_df)
    df_clean.to_csv("data/healthcare_clean.csv", index=False)
    print(f"  Clean dataset saved → data/healthcare_clean.csv  ({df_clean.shape})")

    # ── STEP 3 : Split ────────────────────────────────────────────────────────
    print("\n[STEP 3] Splitting data …")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # ── STEP 4 : SMOTE ────────────────────────────────────────────────────────
    print("\n[STEP 4] Applying SMOTE …")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── STEP 5 : Train All Models ─────────────────────────────────────────────
    print("\n[STEP 5] Training models with GridSearchCV …")
    results = train_all_models(X_train_res, y_train_res, X_val, y_val)

    # ── STEP 6 : Compare & Select Best ────────────────────────────────────────
    print("\n[STEP 6] Comparing models …")
    plot_model_comparison(results)
    best_name, best_model = select_best_model(results)

    # ── STEP 7 : Evaluate on Test Set ─────────────────────────────────────────
    print("\n[STEP 7] Final evaluation on held-out test set …")
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)
    print(classification_report(y_test, y_test_pred,
                                 target_names=["Low Risk", "Medium Risk", "High Risk"]))
    plot_confusion_matrix(y_test, y_test_pred, best_name)
    plot_roc_curves(y_test, y_test_prob, best_name)
    plot_feature_importance(best_model, list(X_test.columns), best_name)

    print("\n" + RECALL_NOTE)

    # ── STEP 8 : Save Artifacts ────────────────────────────────────────────────
    print("\n[STEP 8] Saving model artifacts …")
    save_model(best_model, scaler, list(X_train.columns))

    # ── STEP 9 : SHAP Explainability ──────────────────────────────────────────
    print("\n[STEP 9] Running SHAP explainability …")
    sample_X = X_test.sample(min(300, len(X_test)), random_state=42)
    explainer = get_shap_explainer(best_model, sample_X)
    shap_values = compute_shap_values(explainer, sample_X)
    plot_shap_summary(shap_values, sample_X)
    plot_shap_bar(shap_values, sample_X)

    # Explain a random high-risk patient
    high_risk_mask = y_test.values == 2
    if high_risk_mask.any():
        hr_idx = sample_X.index.get_indexer([X_test[high_risk_mask].index[0]])[0]
        if hr_idx >= 0:
            explanation = explain_single_patient(explainer, shap_values, sample_X, hr_idx)
            print_patient_explanation(explanation, "High Risk")

    # ── STEP 10 : Demo predict_risk() ─────────────────────────────────────────
    print("\n[STEP 10] Demo — predict_risk() on a sample patient")
    from src.model_training import load_artifacts
    model, saved_scaler, feature_names = load_artifacts()

    sample_patient = {
        "age": 65, "sex": 1, "bmi": 32.5, "blood_pressure": 155,
        "cholesterol": 260, "glucose": 135, "smoking": 1,
        "alcohol_intake": 2, "physical_activity": 0,
        "diabetes": 1, "heart_disease": 0,
        "age_group": 2, "bmi_category": 3, "condition_count": 1,
        "lifestyle_risk_score": 6.0, "bp_category": 3, "glucose_category": 2,
    }
    result = predict_risk(model, saved_scaler, feature_names, sample_patient)
    print(f"\n  Predicted Label  : {result['predicted_label']}")
    print(f"  Risk Category    : {result['risk_category']}")
    print(f"  P(High Risk)     : {result['high_risk_probability']:.3f}")
    print(f"  All Probs        : {result['class_probs']}")

    print("\n" + "=" * 70)
    print("  Pipeline complete!  All plots saved to reports/")
    print("  Run the Streamlit app:  streamlit run app/streamlit_app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()