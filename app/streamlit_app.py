"""
streamlit_app.py
----------------
Professional Streamlit web application for the
Intelligent Healthcare Risk Stratification System.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Path setup so src/ imports work ──────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthRisk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b2d 0%, #1a2f4a 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0eaff !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
}

/* Main background */
.main { background-color: #f7f9fc; }

/* Risk Cards */
.risk-card {
    border-radius: 16px;
    padding: 28px 32px;
    margin: 12px 0;
    color: white;
    font-family: 'Syne', sans-serif;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}
.risk-low    { background: linear-gradient(135deg, #11998e, #38ef7d); }
.risk-medium { background: linear-gradient(135deg, #f7971e, #ffd200); color: #1a1a1a !important; }
.risk-high   { background: linear-gradient(135deg, #cb2d3e, #ef473a); }

.risk-card h2 { font-size: 2rem; margin: 0 0 6px 0; }
.risk-card p  { font-size: 1rem; margin: 0; opacity: 0.9; }

/* Metric cards */
.metric-box {
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #3a86ff;
}
.metric-box .val { font-size: 2rem; font-weight: 700; color: #3a86ff; font-family: 'Syne', sans-serif; }
.metric-box .lbl { font-size: 0.85rem; color: #888; margin-top: 4px; }

/* Info box */
.info-box {
    background: #eef4ff;
    border-left: 5px solid #3a86ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    color: #2c3e50;
}

/* Section header */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f1b2d;
    border-bottom: 3px solid #3a86ff;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Prob bar */
.prob-row { margin: 8px 0; }
.prob-label { font-size: 0.9rem; color: #444; margin-bottom: 3px; }

/* Separator */
hr { border: none; border-top: 1px solid #e0e8f0; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load model, scaler and feature names from disk (cached)."""
    models_dir = os.path.join(ROOT, "models")
    try:
        model        = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        scaler       = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
        return model, scaler, feature_names, None
    except FileNotFoundError:
        return None, None, None, (
            "⚠️  Model files not found in `models/` folder.\n\n"
            "Please run `python main.py` first to train and save the model."
        )


def engineer_input_features(raw: dict) -> dict:
    """
    Apply the same feature engineering used in training so the
    patient input dict has all required columns.
    """
    d = dict(raw)

    # age_group  (Young=0, Middle-aged=1, Senior=2 after LabelEncoder)
    if d["age"] <= 35:
        d["age_group"] = 0
    elif d["age"] <= 55:
        d["age_group"] = 1
    else:
        d["age_group"] = 2

    # bmi_category (Obese=0, Normal=1, Overweight=2, Underweight=3 after LE)
    if d["bmi"] < 18.5:
        d["bmi_category"] = 3
    elif d["bmi"] < 25:
        d["bmi_category"] = 1
    elif d["bmi"] < 30:
        d["bmi_category"] = 2
    else:
        d["bmi_category"] = 0

    # condition_count
    d["condition_count"] = d["diabetes"] + d["heart_disease"]

    # lifestyle_risk_score
    d["lifestyle_risk_score"] = (
        d["smoking"] * 2
        + d["alcohol_intake"]
        + (2 - d["physical_activity"])
    )

    # bp_category (Low=0, Normal=1, Pre-hypertension=2, Hypertension=3 after LE)
    bp = d["blood_pressure"]
    if bp < 90:
        d["bp_category"] = 0
    elif bp < 120:
        d["bp_category"] = 1
    elif bp < 140:
        d["bp_category"] = 2
    else:
        d["bp_category"] = 3

    # glucose_category (Diabetic=0, Normal=1, Pre-diabetic=2 after LE)
    g = d["glucose"]
    if g <= 100:
        d["glucose_category"] = 1
    elif g <= 126:
        d["glucose_category"] = 2
    else:
        d["glucose_category"] = 0

    return d


def run_prediction(model, scaler, feature_names, raw_input: dict) -> dict:
    """Scale inputs and return prediction results."""
    patient = engineer_input_features(raw_input)

    df = pd.DataFrame([patient])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    scale_cols = [c for c in ["age", "bmi", "blood_pressure", "cholesterol",
                               "glucose", "lifestyle_risk_score"] if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    pred_class  = int(model.predict(df)[0])
    class_probs = model.predict_proba(df)[0]
    labels      = ["Low Risk", "Medium Risk", "High Risk"]
    high_prob   = float(class_probs[2])

    risk_cat = "Low" if high_prob < 0.30 else ("Medium" if high_prob < 0.70 else "High")

    return {
        "predicted_class": pred_class,
        "predicted_label": labels[pred_class],
        "risk_category":   risk_cat,
        "high_risk_prob":  high_prob,
        "probs": {
            "Low Risk":    float(class_probs[0]),
            "Medium Risk": float(class_probs[1]),
            "High Risk":   float(class_probs[2]),
        },
    }


def probability_bar(label: str, prob: float, color: str):
    """Render a styled probability bar using st.progress."""
    st.markdown(f"<div class='prob-label'><b>{label}</b>  —  {prob*100:.1f}%</div>",
                unsafe_allow_html=True)
    st.progress(prob)


def load_report_image(filename: str):
    path = os.path.join(ROOT, "reports", filename)
    return path if os.path.exists(path) else None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 HealthRisk AI")
    st.markdown("*Intelligent Patient Risk Stratification*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Risk Prediction", "📊 EDA & Reports",
         "🤖 Model Insights", "ℹ️ About"],
    )
    st.markdown("---")
    st.markdown("""
**Risk Levels**
🟢 **Low Risk** — P(High) < 0.30  
🟡 **Medium Risk** — 0.30 – 0.70  
🔴 **High Risk** — P(High) > 0.70
""")
    st.markdown("---")
    st.caption("Powered by XGBoost · SHAP · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (all pages may need it)
# ─────────────────────────────────────────────────────────────────────────────
model, scaler, feature_names, model_error = load_model_artifacts()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("# 🏥 Intelligent Healthcare Risk Stratification")
    st.markdown("### *Predict patient risk levels using clinical and lifestyle data*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("5 Models", "Trained & Compared"),
        ("12 Features", "Clinical Inputs"),
        ("3 Classes", "Low / Medium / High"),
        ("SHAP", "Explainable AI"),
    ]
    for col, (val, lbl) in zip([col1, col2, col3, col4], metrics):
        col.markdown(f"""
        <div class="metric-box">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 What This System Does")
        st.markdown("""
- Analyses **patient demographics**, biometrics, and lifestyle
- Predicts whether a patient is **Low / Medium / High Risk**
- Shows the **probability** of each risk level
- Explains **why** using SHAP feature contributions
- Helps clinicians **prioritise** high-risk patients for early intervention
        """)

    with c2:
        st.markdown("### 🧬 Features Used")
        features_df = pd.DataFrame({
            "Feature": ["Age", "Sex", "BMI", "Blood Pressure", "Cholesterol",
                        "Glucose", "Smoking", "Alcohol Intake", "Physical Activity",
                        "Diabetes", "Heart Disease"],
            "Type": ["Demographic", "Demographic", "Biometric", "Biometric", "Biometric",
                     "Biometric", "Lifestyle", "Lifestyle", "Lifestyle",
                     "Condition", "Condition"],
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
<div class="info-box">
💡 <b>How to use:</b> Go to <b>🔬 Risk Prediction</b> in the sidebar, fill in the patient details, and click <b>Predict Risk</b>.
</div>
""", unsafe_allow_html=True)

    if model_error:
        st.error(model_error)
    else:
        st.success("✅ Model loaded and ready for predictions.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: RISK PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Risk Prediction":
    st.markdown("# 🔬 Patient Risk Prediction")
    st.markdown("Enter the patient's clinical details below.")

    if model_error:
        st.error(model_error)
        st.stop()

    st.markdown("---")

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    with st.form("patient_form"):
        st.markdown("<div class='section-header'>👤 Demographics</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age    = c1.number_input("Age (years)", min_value=18, max_value=90, value=45)
        sex    = c2.selectbox("Sex", ["Male", "Female"])
        bmi    = c3.number_input("BMI", min_value=15.0, max_value=55.0, value=26.5, step=0.1)

        st.markdown("<div class='section-header'>🩺 Clinical Biometrics</div>", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        bp          = c4.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        cholesterol = c5.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=195)
        glucose     = c6.number_input("Glucose (mg/dL)", min_value=60, max_value=350, value=98)

        st.markdown("<div class='section-header'>🏃 Lifestyle Factors</div>", unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        smoking           = c7.selectbox("Smoking", ["No", "Yes"])
        alcohol_intake    = c8.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])
        physical_activity = c9.selectbox("Physical Activity", ["Inactive", "Moderate", "Active"])

        st.markdown("<div class='section-header'>💊 Existing Conditions</div>", unsafe_allow_html=True)
        c10, c11 = st.columns(2)
        diabetes     = c10.selectbox("Diabetes", ["No", "Yes"])
        heart_disease = c11.selectbox("Heart Disease", ["No", "Yes"])

        st.markdown("---")
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    # ── PREDICTION ────────────────────────────────────────────────────────────
    if submitted:
        raw_input = {
            "age":               age,
            "sex":               1 if sex == "Male" else 0,
            "bmi":               bmi,
            "blood_pressure":    bp,
            "cholesterol":       cholesterol,
            "glucose":           glucose,
            "smoking":           1 if smoking == "Yes" else 0,
            "alcohol_intake":    ["None", "Moderate", "Heavy"].index(alcohol_intake),
            "physical_activity": ["Inactive", "Moderate", "Active"].index(physical_activity),
            "diabetes":          1 if diabetes == "Yes" else 0,
            "heart_disease":     1 if heart_disease == "Yes" else 0,
        }

        with st.spinner("Analysing patient data …"):
            result = run_prediction(model, scaler, feature_names, raw_input)

        st.markdown("---")
        st.markdown("## 📋 Prediction Results")

        # ── Risk Card ─────────────────────────────────────────────────────────
        label    = result["predicted_label"]
        cat      = result["risk_category"]
        css_cls  = f"risk-{cat.lower()}"
        emoji    = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[cat]
        advice   = {
            "Low":    "Patient shows low risk. Maintain healthy lifestyle and routine checkups.",
            "Medium": "Patient shows moderate risk. Lifestyle intervention and regular monitoring advised.",
            "High":   "Patient is HIGH RISK. Immediate clinical evaluation and intervention recommended.",
        }[cat]

        st.markdown(f"""
<div class="risk-card {css_cls}">
    <h2>{emoji} {label}</h2>
    <p>{advice}</p>
</div>""", unsafe_allow_html=True)

        # ── Probability Breakdown ──────────────────────────────────────────────
        st.markdown("### 📊 Probability Breakdown")
        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            colors = {"Low Risk": "#2ECC71", "Medium Risk": "#F39C12", "High Risk": "#E74C3C"}
            for lbl, prob in result["probs"].items():
                probability_bar(lbl, prob, colors[lbl])

        with col_p2:
            fig, ax = plt.subplots(figsize=(4, 4))
            probs  = list(result["probs"].values())
            labels = list(result["probs"].keys())
            clrs   = [colors[l] for l in labels]
            wedges, texts, autotexts = ax.pie(
                probs, labels=labels, autopct="%1.1f%%",
                colors=clrs, startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2),
            )
            for at in autotexts:
                at.set_fontsize(10)
                at.set_fontweight("bold")
            ax.set_title("Risk Distribution", fontsize=11, fontweight="bold", pad=10)
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Patient Summary Table ──────────────────────────────────────────────
        st.markdown("### 🗂️ Patient Summary")
        summary = pd.DataFrame({
            "Parameter": ["Age", "Sex", "BMI", "Blood Pressure", "Cholesterol",
                          "Glucose", "Smoking", "Alcohol", "Physical Activity",
                          "Diabetes", "Heart Disease"],
            "Value": [
                f"{age} yrs", sex, f"{bmi:.1f}", f"{bp} mmHg",
                f"{cholesterol} mg/dL", f"{glucose} mg/dL",
                smoking, alcohol_intake, physical_activity,
                diabetes, heart_disease,
            ],
            "Status": [
                "⚠️ Senior" if age > 60 else "✅ Normal",
                "—",
                "⚠️ Obese" if bmi > 30 else ("⚠️ Overweight" if bmi > 25 else "✅ Normal"),
                "🔴 High" if bp > 140 else ("⚠️ Pre-hyp." if bp > 120 else "✅ Normal"),
                "🔴 High" if cholesterol > 240 else "✅ Normal",
                "🔴 Diabetic" if glucose > 126 else ("⚠️ Pre-diabetic" if glucose > 100 else "✅ Normal"),
                "🔴 Yes" if smoking == "Yes" else "✅ No",
                "🔴 Heavy" if alcohol_intake == "Heavy" else "✅ OK",
                "🔴 Inactive" if physical_activity == "Inactive" else "✅ Active",
                "🔴 Yes" if diabetes == "Yes" else "✅ No",
                "🔴 Yes" if heart_disease == "Yes" else "✅ No",
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Clinical Recommendations ───────────────────────────────────────────
        st.markdown("### 💡 Clinical Recommendations")
        recs = []
        if age > 60:           recs.append("• Schedule annual cardiovascular screening for senior patient.")
        if bmi > 30:           recs.append("• Recommend weight management programme (BMI in obese range).")
        if bp > 140:           recs.append("• Evaluate for hypertension treatment — BP above 140 mmHg.")
        if cholesterol > 240:  recs.append("• Consider statin therapy — cholesterol elevated above 240 mg/dL.")
        if glucose > 126:      recs.append("• Refer to endocrinology — glucose suggests diabetes.")
        if smoking == "Yes":   recs.append("• Enrol in smoking cessation programme.")
        if physical_activity == "Inactive": recs.append("• Prescribe structured exercise routine (150 min/week).")
        if diabetes == "Yes":  recs.append("• Ongoing diabetes management and HbA1c monitoring.")
        if heart_disease == "Yes": recs.append("• Cardiology follow-up and medication review.")

        if recs:
            for r in recs:
                st.markdown(r)
        else:
            st.success("✅ Patient shows no major clinical red flags. Maintain current healthy habits.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & REPORTS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA & Reports":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("Visual insights generated during model training.")
    st.markdown("---")

    report_images = [
        ("01_risk_distribution.png",  "Risk Category Distribution"),
        ("02_age_distribution.png",   "Age Distribution"),
        ("03_bmi_by_risk.png",        "BMI by Risk Category"),
        ("04_correlation_heatmap.png","Correlation Heatmap"),
        ("05_smoking_impact.png",     "Smoking Impact on Risk"),
        ("06_cholesterol_patterns.png","Cholesterol by Risk Level"),
        ("07_feature_histograms.png", "All Feature Distributions"),
    ]

    found_any = False
    for i in range(0, len(report_images), 2):
        c1, c2 = st.columns(2)
        for col, (fname, title) in zip([c1, c2], report_images[i:i+2]):
            path = load_report_image(fname)
            if path:
                found_any = True
                col.markdown(f"**{title}**")
                col.image(path, use_column_width=True)
            else:
                col.info(f"'{fname}' not found. Run `python main.py` to generate plots.")

    if not found_any:
        st.warning("No report images found. Please run `python main.py` first.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Insights":
    st.markdown("# 🤖 Model Insights & Explainability")
    st.markdown("---")

    tabs = st.tabs(["📈 Model Comparison", "🎯 Confusion Matrix",
                    "📉 ROC Curves", "🏆 Feature Importance", "🔍 SHAP Explainability"])

    images = {
        "comparison":   "08_model_comparison.png",
        "confusion":    "09_confusion_matrix.png",
        "roc":          "10_roc_curves.png",
        "importance":   "11_feature_importance.png",
        "shap_summary": "12_shap_summary.png",
        "shap_bar":     "13_shap_bar.png",
    }

    with tabs[0]:
        p = load_report_image(images["comparison"])
        if p: st.image(p, use_column_width=True)
        else: st.info("Run `python main.py` to generate this chart.")
        st.markdown("""
**Five models were trained and compared:**
- Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM
- GridSearchCV tuned hyperparameters using 5-fold stratified cross-validation
- Best model selected by weighted F1-score
        """)

    with tabs[1]:
        p = load_report_image(images["confusion"])
        if p: st.image(p, use_column_width=True)
        else: st.info("Run `python main.py` to generate this chart.")
        st.markdown("""
**Reading the Confusion Matrix:**
- Diagonal = correct predictions  
- Off-diagonal = misclassifications  
- Most critical: High Risk patients predicted as Low Risk (bottom-left cell)
        """)

    with tabs[2]:
        p = load_report_image(images["roc"])
        if p: st.image(p, use_column_width=True)
        else: st.info("Run `python main.py` to generate this chart.")
        st.markdown("**AUC closer to 1.0 = better discrimination between risk classes.**")

    with tabs[3]:
        p = load_report_image(images["importance"])
        if p: st.image(p, use_column_width=True)
        else: st.info("Run `python main.py` to generate this chart.")
        st.markdown("**Longer bars = feature has more influence on the prediction.**")

    with tabs[4]:
        col1, col2 = st.columns(2)
        for col, key, title in [
            (col1, "shap_summary", "SHAP Summary (Beeswarm)"),
            (col2, "shap_bar",     "Mean |SHAP| — Global Importance"),
        ]:
            p = load_report_image(images[key])
            if p:
                col.markdown(f"**{title}**")
                col.image(p, use_column_width=True)
            else:
                col.info("Run `python main.py` to generate SHAP plots.")

        st.markdown("---")
        st.markdown("""
### 🔍 How to Read SHAP Plots

**Beeswarm Plot (Summary)**
- Each dot = one patient
- **Red dots** = high feature value; **Blue dots** = low feature value
- Dots on the **right** push the prediction toward High Risk
- Dots on the **left** reduce High Risk probability

**Bar Chart (Global Importance)**
- Shows average impact of each feature across ALL patients
- Top features are the most influential drivers of High Risk predictions

**Why SHAP matters in healthcare:**
SHAP turns the "black box" model into a transparent system that clinicians
can trust — for each patient you can see exactly *which* factors contributed
most to their risk classification.
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
## Intelligent Healthcare Risk Stratification System

This project demonstrates a **production-grade AI/ML pipeline** applied to
healthcare risk prediction. It covers the full data science workflow:

| Stage | What happens |
|-------|-------------|
| 📥 Data Generation | Synthetic 5,000-patient dataset with realistic correlations |
| 🧹 Cleaning | Duplicate removal, missing value imputation, IQR outlier removal |
| 📊 EDA | 7 publication-quality visualisations saved to `reports/` |
| ⚙️ Feature Engineering | age_group, bmi_category, lifestyle_risk_score, etc. |
| ⚖️ Class Balancing | SMOTE applied to training set only |
| 🤖 Modelling | 5 classifiers + GridSearchCV hyperparameter tuning |
| 📈 Evaluation | Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, ROC |
| 🔍 Explainability | SHAP TreeExplainer — global and per-patient explanations |
| 🚦 Risk Stratification | Probability thresholds: Low < 0.30 < Medium < 0.70 < High |
| 🌐 Deployment | This Streamlit app — runs locally or on Streamlit Cloud |

---

### 🛠️ Tech Stack
`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `XGBoost`  
`imbalanced-learn (SMOTE)` · `SHAP` · `matplotlib` · `seaborn` · `Streamlit` · `joblib`

---

### ⚠️ Disclaimer
This tool is for **educational and research purposes only**.  
It is **not** a medical device and should not replace clinical judgement.
    """)

    st.markdown("---")
    st.markdown("""
<div class="info-box">
📌 <b>Quick Start:</b><br>
1. Run <code>python main.py</code> to train the model<br>
2. Run <code>streamlit run app/streamlit_app.py</code> to launch this app<br>
3. Go to <b>🔬 Risk Prediction</b> and enter patient details
</div>
""", unsafe_allow_html=True)