"""
preprocessing.py
----------------
Data cleaning, EDA, and feature engineering for the healthcare risk system.
All functions are reusable and well-commented.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Palette consistent with the project theme ────────────────────────────────
PALETTE = {"Low Risk": "#2ECC71", "Medium Risk": "#F39C12", "High Risk": "#E74C3C"}
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def basic_info(df: pd.DataFrame) -> None:
    """Print dataset shape, dtypes, and a statistical summary."""
    print("=" * 60)
    print(f"  DATASET SHAPE : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 60)
    print("\n── COLUMN TYPES ──")
    print(df.dtypes)
    print("\n── STATISTICAL SUMMARY ──")
    print(df.describe(include="all").T.to_string())
    print("\n── MISSING VALUES ──")
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    print(pd.concat([miss, miss_pct], axis=1, keys=["Count", "%"]).query("Count > 0"))


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and report how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed: {before - len(df)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical  → median (robust to outliers)
    - Categorical → mode
    """
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values imputed.")
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Remove outliers using the IQR method for the specified columns.
    Rows where a value is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are dropped.
    """
    before = len(df)
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    df = df.reset_index(drop=True)
    print(f"Outliers removed: {before - len(df)} rows")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary/ordinal categoricals; one-hot for nominal."""
    # sex → binary
    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])          # Male=1, Female=0

    # target → ordinal integer
    risk_map = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    df["target"] = df["target"].map(risk_map)

    return df


def scale_features(df: pd.DataFrame, num_cols: list):
    """StandardScale numerical features. Returns (scaled_df, scaler)."""
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


# ═════════════════════════════════════════════════════════════════════════════
# 2.  EDA  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str) -> None:
    path = os.path.join(REPORTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Count-plot of the three risk categories."""
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Low Risk", "Medium Risk", "High Risk"]
    raw = df.copy()
    raw["risk_label"] = raw["target"].map({0: "Low Risk", 1: "Medium Risk", 2: "High Risk"})
    counts = raw["risk_label"].value_counts().reindex(order)
    bars = ax.bar(order, counts.values, color=[PALETTE[o] for o in order], width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Risk Category Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Patient Count")
    ax.set_xlabel("Risk Category")
    _save(fig, "01_risk_distribution")


def plot_age_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of patient age."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["age"] if "age" in df.columns else df["age"],
            bins=30, color="#3498DB", edgecolor="white", alpha=0.85)
    ax.set_title("Age Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    _save(fig, "02_age_distribution")


def plot_bmi_by_risk(df: pd.DataFrame) -> None:
    """Box-plot of BMI across risk categories."""
    raw = df.copy()
    raw["risk_label"] = raw["target"].map({0: "Low Risk", 1: "Medium Risk", 2: "High Risk"})
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Low Risk", "Medium Risk", "High Risk"]
    for i, cat in enumerate(order):
        data = raw[raw["risk_label"] == cat]["bmi"].dropna()
        ax.boxplot(data, positions=[i], patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[cat], alpha=0.7),
                   medianprops=dict(color="black", linewidth=2))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(order)
    ax.set_title("BMI Distribution by Risk Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("BMI")
    _save(fig, "03_bmi_by_risk")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlation among numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(fig, "04_correlation_heatmap")


def plot_smoking_impact(df: pd.DataFrame) -> None:
    """Stacked bar chart: smoking status vs risk level."""
    raw = df.copy()
    raw["risk_label"] = raw["target"].map({0: "Low Risk", 1: "Medium Risk", 2: "High Risk"})
    ct = pd.crosstab(raw["smoking"], raw["risk_label"], normalize="index") * 100
    ct.index = ["Non-Smoker", "Smoker"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[PALETTE["Low Risk"], PALETTE["Medium Risk"], PALETTE["High Risk"]],
            edgecolor="white")
    ax.set_title("Smoking Impact on Risk (% within group)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.legend(title="Risk", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=0)
    _save(fig, "05_smoking_impact")


def plot_cholesterol_patterns(df: pd.DataFrame) -> None:
    """Histogram of cholesterol with risk-level overlay."""
    raw = df.copy()
    raw["risk_label"] = raw["target"].map({0: "Low Risk", 1: "Medium Risk", 2: "High Risk"})
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat in ["Low Risk", "Medium Risk", "High Risk"]:
        subset = raw[raw["risk_label"] == cat]["cholesterol"].dropna()
        ax.hist(subset, bins=30, alpha=0.55, label=cat, color=PALETTE[cat])
    ax.axvline(240, color="black", linestyle="--", label="High Threshold (240)")
    ax.set_title("Cholesterol Distribution by Risk Level", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cholesterol (mg/dL)")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, "06_cholesterol_patterns")


def plot_feature_histograms(df: pd.DataFrame) -> None:
    """Grid of histograms for all numeric features."""
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "target"]
    n = len(num_cols)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, rows * 3.5))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=25, color="#5DADE2", edgecolor="white", alpha=0.85)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "07_feature_histograms")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed features on top of the cleaned dataset.

    New features
    ────────────
    age_group            : binned age (Young / Middle-aged / Senior)
    bmi_category         : WHO BMI classification (Underweight / Normal /
                           Overweight / Obese)
    condition_count      : number of existing conditions (diabetes + heart_disease)
    lifestyle_risk_score : composite lifestyle risk (smoking + alcohol + inactivity)
    bp_category          : blood-pressure category
    glucose_category     : glucose category (Normal / Pre-diabetic / Diabetic)
    """
    df = df.copy()

    # age_group
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 35, 55, 100],
        labels=["Young", "Middle-aged", "Senior"],
    ).astype(str)
    df["age_group"] = LabelEncoder().fit_transform(df["age_group"])

    # bmi_category
    bmi_bins = [0, 18.5, 24.9, 29.9, 100]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels).astype(str)
    df["bmi_category"] = LabelEncoder().fit_transform(df["bmi_category"])

    # condition_count
    df["condition_count"] = df["diabetes"] + df["heart_disease"]

    # lifestyle_risk_score (higher = riskier)
    df["lifestyle_risk_score"] = (
        df["smoking"] * 2
        + df["alcohol_intake"]
        + (2 - df["physical_activity"])        # inactivity increases risk
    )

    # bp_category
    df["bp_category"] = pd.cut(
        df["blood_pressure"],
        bins=[0, 90, 120, 140, 300],
        labels=["Low", "Normal", "Pre-hypertension", "Hypertension"],
    ).astype(str)
    df["bp_category"] = LabelEncoder().fit_transform(df["bp_category"])

    # glucose_category
    df["glucose_category"] = pd.cut(
        df["glucose"],
        bins=[0, 100, 126, 500],
        labels=["Normal", "Pre-diabetic", "Diabetic"],
    ).astype(str)
    df["glucose_category"] = LabelEncoder().fit_transform(df["glucose_category"])

    return df


def run_preprocessing_pipeline(raw_df: pd.DataFrame):
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    df_clean   : cleaned + encoded + feature-engineered DataFrame
    scaler     : fitted StandardScaler (needed for inference)
    num_cols   : list of numeric columns that were scaled
    """
    print("\n[1] Basic Info")
    basic_info(raw_df)

    print("\n[2] Removing Duplicates")
    df = remove_duplicates(raw_df)

    print("\n[3] Handling Missing Values")
    df = handle_missing_values(df)

    print("\n[4] Removing Outliers (IQR)")
    outlier_cols = ["bmi", "blood_pressure", "cholesterol", "glucose"]
    df = remove_outliers_iqr(df, outlier_cols)

    print("\n[5] Encoding Categoricals")
    df = encode_categoricals(df)

    print("\n[6] Feature Engineering")
    df = engineer_features(df)

    print("\n[7] Generating EDA Plots")
    plot_target_distribution(df)
    plot_age_distribution(df)
    plot_bmi_by_risk(df)
    plot_correlation_heatmap(df)
    plot_smoking_impact(df)
    plot_cholesterol_patterns(df)
    plot_feature_histograms(df)

    print("\n[8] Scaling Numerical Features")
    num_cols = [
        "age", "bmi", "blood_pressure", "cholesterol", "glucose",
        "lifestyle_risk_score",
    ]
    df, scaler = scale_features(df, num_cols)

    print(f"\nPreprocessing complete. Final shape: {df.shape}")
    return df, scaler, num_cols