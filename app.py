from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, roc_auc_score


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "customer_churn_model.pkl"
DEFAULT_DATA_PATH = APP_DIR / "Telco-Customer-Churn.csv"


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df


st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
st.title("Telecom Customer Churn Prediction")

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    model_exists = Path(model_path).exists()
    if not model_exists:
        st.warning("Model file not found. Train one with `python train.py`.")

model = load_model(model_path) if model_exists else None

tab_predict, tab_batch, tab_eval = st.tabs(["Single prediction", "Batch prediction", "Evaluate / retrain"])

with tab_predict:
    st.subheader("Single customer prediction")
    st.write("Fill the form and get churn probability (Yes).")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        phone_service = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment_method = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col3:
        online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
        monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=1000.0)

    input_df = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    if st.button("Predict", type="primary", disabled=model is None):
        input_df = coerce_total_charges(input_df)
        proba = None
        if hasattr(model, "predict_proba"):
            classes = list(getattr(model.named_steps.get("model", model), "classes_", []))
            probs = model.predict_proba(input_df)
            if "Yes" in classes:
                proba = float(probs[:, classes.index("Yes")][0])
            elif probs.shape[1] > 1:
                proba = float(probs[:, 1][0])
        pred = model.predict(input_df)[0] if model is not None else "Unknown"

        left, right = st.columns([1, 2])
        with left:
            st.metric("Prediction", pred)
            if proba is not None:
                st.metric("Churn probability (Yes)", f"{proba:.2%}")
        with right:
            st.dataframe(input_df, use_container_width=True)

with tab_batch:
    st.subheader("Batch prediction")
    st.write("Upload a CSV with the same columns as the dataset (excluding `Churn` is fine).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None and model is not None:
        df = pd.read_csv(uploaded)
        df = coerce_total_charges(df)
        features = ensure_feature_columns(df)

        preds = model.predict(features)
        out = df.copy()
        out["Churn_Prediction"] = preds

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)
            classes = list(getattr(model.named_steps.get("model", model), "classes_", []))
            if "Yes" in classes:
                out["Churn_Prob_Yes"] = probs[:, classes.index("Yes")]
            elif probs.shape[1] > 1:
                out["Churn_Prob_Yes"] = probs[:, 1]

        st.dataframe(out.head(50), use_container_width=True)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="churn_predictions.csv")

with tab_eval:
    st.subheader("Evaluate model (optional)")
    st.write("Evaluate the current model on a labeled dataset (must include `Churn`).")

    eval_choice = st.radio("Evaluation data source", ["Use default dataset", "Upload CSV"], horizontal=True)
    eval_df = None

    if eval_choice == "Use default dataset":
        if DEFAULT_DATA_PATH.exists():
            eval_df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            st.info("Default dataset not found next to the app. Upload one instead.")
    else:
        eval_upload = st.file_uploader("Upload labeled CSV (includes Churn)", type=["csv"], key="eval_csv")
        if eval_upload is not None:
            eval_df = pd.read_csv(eval_upload)

    if eval_df is not None and model is not None:
        eval_df = coerce_total_charges(eval_df)
        if "Churn" not in eval_df.columns:
            st.error("Your evaluation CSV must include a `Churn` column.")
        else:
            y_true = eval_df["Churn"].astype(str)
            X_eval = ensure_feature_columns(eval_df)
            y_pred = model.predict(X_eval)
            st.text("Classification report:")
            st.code(classification_report(y_true, y_pred, digits=4))

            if hasattr(model, "predict_proba") and y_true.nunique() > 1:
                probs = model.predict_proba(X_eval)
                classes = list(getattr(model.named_steps.get("model", model), "classes_", []))
                if "Yes" in classes:
                    y_score = probs[:, classes.index("Yes")]
                else:
                    y_score = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(y_true))
                auc = roc_auc_score((y_true == "Yes").astype(int), y_score)
                st.metric("ROC-AUC", f"{auc:.4f}")

    st.divider()
    st.subheader("Retrain model (saves over model path)")
    st.write("This retrains using scikit-learn with the same preprocessing as `train.py`.")

    retrain_upload = st.file_uploader("Upload training CSV (includes Churn)", type=["csv"], key="train_csv")
    if st.button("Retrain now", disabled=retrain_upload is None):
        from train import build_pipeline  # local import to avoid slowing app start

        train_df = pd.read_csv(retrain_upload)
        train_df = coerce_total_charges(train_df)
        if "Churn" not in train_df.columns:
            st.error("Training CSV must include a `Churn` column.")
        else:
            y = train_df["Churn"].astype(str)
            X = ensure_feature_columns(train_df)
            pipe = build_pipeline(X)
            pipe.fit(X, y)
            joblib.dump(pipe, model_path)
            st.success(f"Saved retrained model to `{model_path}`. Reload the page to use it.")