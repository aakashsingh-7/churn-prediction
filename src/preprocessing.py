"""Data loading, cleaning and feature engineering shared by training and the app."""
from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# 'gender' is deliberately excluded: it is a protected attribute and carries no real signal here.
NUMERIC_BASE = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    *ADDON_COLS, "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_FEATURES = NUMERIC_BASE + ["n_addons"]  # n_addons is engineered below
RAW_FEATURES = NUMERIC_BASE + CATEGORICAL_FEATURES  # what a user/app must supply


def load_data(path):
    """Read the Telco CSV and return a clean frame with a 0/1 Churn target."""
    df = pd.read_csv(Path(path))
    # TotalCharges is blank for customers with tenure 0 (brand-new): they have been billed nothing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    assert df["TotalCharges"].notna().all(), "Unexpected missing TotalCharges"
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=[ID_COL])


def add_features(X):
    """Engineer features. Works on a raw-feature DataFrame (used inside the model pipeline)."""
    X = X.copy()
    X["n_addons"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
    return X
