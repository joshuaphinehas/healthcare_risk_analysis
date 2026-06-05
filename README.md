# Intelligent Healthcare Risk Stratification System

## Live Demo

Streamlit App: https://healthcareriskanalysis.streamlit.app

## GitHub Repository

Repository: https://github.com/joshuaphinehas/healthcare_risk_analysis

## Project Overview

The Intelligent Healthcare Risk Stratification System predicts whether a patient belongs to the Low Risk, Medium Risk, or High Risk category based on healthcare parameters such as age, BMI, blood pressure, cholesterol, glucose level, smoking status, diabetes, and heart disease history.

## Features

* Data preprocessing and cleaning
* Missing value handling
* Outlier detection and removal
* Feature engineering
* Exploratory Data Analysis (EDA)
* Class balancing using SMOTE
* Multiple machine learning models
* XGBoost-based risk prediction
* SHAP explainability
* Interactive Streamlit web application

## Model Performance

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 88.1%    | 0.881    |
| Decision Tree       | 83.2%    | 0.833    |
| Random Forest       | 88.7%    | 0.887    |
| XGBoost             | 88.9%    | 0.889    |
| SVM                 | 86.7%    | 0.868    |

Best Model: XGBoost

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib
* Seaborn

## How to Run Locally

```bash
git clone https://github.com/joshuaphinehas/healthcare_risk_analysis.git
cd healthcare_risk_analysis
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
