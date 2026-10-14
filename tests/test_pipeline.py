from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import RAW_FEATURES, add_features, load_data

ROOT = Path(__file__).resolve().parent.parent


def test_load_data_is_clean():
    df = load_data(ROOT / "data" / "telco_churn.csv")
    assert df.isna().sum().sum() == 0
    assert set(df["Churn"].unique()) == {0, 1}
    assert "customerID" not in df.columns


def test_add_features_counts_addons():
    df = load_data(ROOT / "data" / "telco_churn.csv")[RAW_FEATURES].head(50)
    out = add_features(df)
    assert out["n_addons"].between(0, 6).all()


def test_model_predicts_valid_probabilities():
    bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
    df = load_data(ROOT / "data" / "telco_churn.csv")[RAW_FEATURES].head(100)
    proba = bundle["pipeline"].predict_proba(df)[:, 1]
    assert proba.shape == (100,)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert 0 < bundle["threshold"] < 1
