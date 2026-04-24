import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _load_telco_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = [c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] if c in X.columns]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a churn prediction pipeline on Telco data.")
    parser.add_argument(
        "--data",
        type=str,
        default="Telco-Customer-Churn.csv",
        help="Path to Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="customer_churn_model.pkl",
        help="Output model path (joblib pickle).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    df = _load_telco_csv(data_path)

    if "Churn" not in df.columns:
        raise ValueError("Expected a 'Churn' column in the dataset.")

    drop_cols = [c for c in ["customerID"] if c in df.columns]
    X = df.drop(columns=["Churn"] + drop_cols)
    y = df["Churn"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    auc = None
    if hasattr(pipe, "predict_proba") and y_test.nunique() > 1:
        proba = pipe.predict_proba(X_test)
        classes = list(getattr(pipe.named_steps["model"], "classes_", []))
        if "Yes" in classes:
            yes_idx = classes.index("Yes")
            y_score = proba[:, yes_idx]
        else:
            y_score = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(y_test))
        auc = roc_auc_score((y_test == "Yes").astype(int), y_score)

    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC:  {auc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"\nSaved model to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

