"""Streamlit app: predict customer churn risk and explore what-if retention scenarios."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import RAW_FEATURES  # noqa: F401  (also makes the pickled pipeline importable)

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return bundle["pipeline"], bundle["threshold"], metrics


pipeline, threshold, metrics = load_artifacts()


def predict(row):
    return float(pipeline.predict_proba(pd.DataFrame([row])[RAW_FEATURES])[0, 1])


def risk_band(p):
    if p >= threshold:
        return "🔴 High risk"
    if p >= threshold / 2:
        return "🟡 Medium risk"
    return "🟢 Low risk"


st.title("📉 Customer Churn Predictor")
st.caption("Telecom customer churn model (Gradient Boosting). Enter a customer profile to estimate churn risk.")

tab_predict, tab_model = st.tabs(["Predict", "About the model"])

with tab_predict:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0, step=1.0)
    with c2:
        st.subheader("Services")
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        if internet == "No":
            addon_opts, addon_default = ["No internet service"], 0
        else:
            addon_opts, addon_default = ["No", "Yes"], 0
        security = st.selectbox("Online security", addon_opts, index=addon_default)
        backup = st.selectbox("Online backup", addon_opts, index=addon_default)
        protection = st.selectbox("Device protection", addon_opts, index=addon_default)
        support = st.selectbox("Tech support", addon_opts, index=addon_default)
        tv = st.selectbox("Streaming TV", addon_opts, index=addon_default)
        movies = st.selectbox("Streaming movies", addon_opts, index=addon_default)
    with c3:
        st.subheader("Household")
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])

    customer = {
        "SeniorCitizen": 1 if senior == "Yes" else 0, "tenure": tenure,
        "MonthlyCharges": monthly, "TotalCharges": round(monthly * tenure, 2),  # estimated
        "Partner": partner, "Dependents": dependents, "PhoneService": phone,
        "MultipleLines": multi if phone == "Yes" else "No phone service",
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }

    prob = predict(customer)
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Risk level", risk_band(prob))
    st.progress(min(prob, 1.0))
    st.caption(f"High-risk cutoff = {threshold:.2f} (chosen to minimise expected business cost).")

    st.subheader("What-if: retention levers")
    scenarios = []
    if contract == "Month-to-month":
        scenarios += [("Move to a one-year contract", {"Contract": "One year"}),
                      ("Move to a two-year contract", {"Contract": "Two year"})]
    if internet != "No" and support == "No":
        scenarios.append(("Add tech support", {"TechSupport": "Yes"}))
    if internet != "No" and security == "No":
        scenarios.append(("Add online security", {"OnlineSecurity": "Yes"}))
    if payment == "Electronic check":
        scenarios.append(("Switch to automatic card payment", {"PaymentMethod": "Credit card (automatic)"}))
    if scenarios:
        rows = []
        for label, change in scenarios:
            new_p = predict({**customer, **change})
            rows.append({"Scenario": label, "New churn probability": f"{new_p:.1%}",
                         "Change": f"{(new_p - prob) * 100:+.1f} pts"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("These are model associations, not proven causal effects. Test offers before rolling them out.")
    else:
        st.info("No obvious retention levers for this profile.")

with tab_model:
    t = metrics["test_tuned_threshold"]
    d = metrics["test_default_threshold"]
    b = metrics["business"]
    a, bcol, c, dcol = st.columns(4)
    a.metric("ROC-AUC (test)", f"{t['roc_auc']:.3f}")
    bcol.metric("Recall @ cutoff", f"{t['recall']:.1%}")
    c.metric("Precision @ cutoff", f"{t['precision']:.1%}")
    saving = 1 - b["test_cost_tuned"] / b["test_cost_no_model"]
    dcol.metric("Cost vs no model", f"-{saving:.0%}")
    st.write(f"Best model: **{metrics['best_model']}**, trained on {metrics['dataset']['train_rows']:,} customers "
             f"and tested on {metrics['dataset']['test_rows']:,}. Default 0.5 cutoff: "
             f"recall {d['recall']:.1%}, precision {d['precision']:.1%}.")
    fig_dir = ROOT / "reports" / "figures"
    for name in ["08_feature_importance.png", "05_roc_curve.png", "06_confusion_matrices.png"]:
        if (fig_dir / name).exists():
            st.image(str(fig_dir / name))
