"""
data_generator.py
-----------------
Generates a synthetic healthcare dataset for the risk stratification system.
This simulates realistic patient data with correlations between features and risk.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)


def generate_healthcare_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate a synthetic healthcare dataset with realistic correlations.

    Args:
        n_samples: Number of patient records to generate.

    Returns:
        pd.DataFrame: Synthetic healthcare dataset.
    """
    # ── Demographics ──────────────────────────────────────────────────────────
    age = np.random.normal(loc=50, scale=15, size=n_samples).clip(18, 90).astype(int)
    sex = np.random.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])

    # ── Biometrics (correlated with age) ─────────────────────────────────────
    bmi = np.random.normal(loc=27 + age * 0.03, scale=5, size=n_samples).clip(16, 50)
    blood_pressure = np.random.normal(
        loc=120 + age * 0.3, scale=15, size=n_samples
    ).clip(80, 200).astype(int)
    cholesterol = np.random.normal(
        loc=200 + age * 0.4, scale=30, size=n_samples
    ).clip(120, 350).astype(int)
    glucose = np.random.normal(
        loc=100 + age * 0.2, scale=20, size=n_samples
    ).clip(60, 300).astype(int)

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    smoking = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    alcohol_intake = np.random.choice([0, 1, 2], size=n_samples, p=[0.50, 0.35, 0.15])
    physical_activity = np.random.choice([0, 1, 2], size=n_samples, p=[0.35, 0.40, 0.25])

    # ── Existing Conditions ───────────────────────────────────────────────────
    diabetes_prob = 0.05 + (glucose > 126) * 0.40 + (bmi > 30) * 0.15
    diabetes = (np.random.random(n_samples) < diabetes_prob).astype(int)

    heart_disease_prob = (
        0.03
        + (cholesterol > 240) * 0.20
        + (blood_pressure > 140) * 0.20
        + (smoking == 1) * 0.15
        + (age > 60) * 0.10
    )
    heart_disease = (np.random.random(n_samples) < heart_disease_prob.clip(0, 0.95)).astype(int)

    # ── Composite Risk Score → Target ─────────────────────────────────────────
    risk_score = (
        (age > 60) * 2
        + (bmi > 30) * 1.5
        + (blood_pressure > 140) * 2
        + (cholesterol > 240) * 1.5
        + (glucose > 126) * 2
        + smoking * 1.5
        + (alcohol_intake == 2) * 1
        + (physical_activity == 0) * 1
        + diabetes * 3
        + heart_disease * 3
        + np.random.normal(0, 0.5, n_samples)  # noise
    )

    # Assign risk categories
    target = pd.cut(
        risk_score,
        bins=[-np.inf, 4, 8, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi.round(1),
            "blood_pressure": blood_pressure,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "smoking": smoking,
            "alcohol_intake": alcohol_intake,
            "physical_activity": physical_activity,
            "diabetes": diabetes,
            "heart_disease": heart_disease,
            "target": target,
        }
    )

    # ── Inject realistic missing values (~3%) ─────────────────────────────────
    for col in ["bmi", "cholesterol", "glucose", "blood_pressure"]:
        mask = np.random.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    # ── Inject a few duplicates (~1%) ─────────────────────────────────────────
    dup_idx = np.random.choice(df.index, size=int(n_samples * 0.01), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


if __name__ == "__main__":
    df = generate_healthcare_dataset()
    df.to_csv("../data/healthcare_data.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df["target"].value_counts())