

import os
import json
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # --- Cleaning ---
    # TotalCharges has 11 blank strings (customers with 0 tenure -> no charges yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop ID column, encode target
    df = df.drop(columns=["customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    best_name, best_pipeline, best_score = None, None, -np.inf

    for name, clf in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", clf)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

        results[name] = {
            "test_accuracy": round(accuracy_score(y_test, preds), 4),
            "test_precision": round(precision_score(y_test, preds), 4),
            "test_recall": round(recall_score(y_test, preds), 4),
            "test_f1": round(f1_score(y_test, preds), 4),
            "test_roc_auc": round(roc_auc_score(y_test, probs), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
        }

        print(f"{name}: roc_auc={results[name]['test_roc_auc']}, "
              f"f1={results[name]['test_f1']}, "
              f"cv_roc_auc={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if cv_scores.mean() > best_score:
            best_name, best_pipeline, best_score = name, pipeline, cv_scores.mean()

    print(f"\nBest model: {best_name} (cv_roc_auc={best_score:.4f})")

    # Feature importance (works for both linear coef and tree importances)
    feature_names = preprocessor.get_feature_names_out()
    model = best_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        importances = np.zeros(len(feature_names))

    feature_importance = dict(
        sorted(
            zip(feature_names, importances.round(4)),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:15]
    )

    joblib.dump(best_pipeline, os.path.join(model_dir, "churn_model.joblib"))

    metadata = {
        "best_model": best_name,
        "raw_feature_names": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_options": {
            col: sorted(X[col].dropna().unique().tolist()) for col in categorical_features
        },
        "feature_importance_top15": {k: float(v) for k, v in feature_importance.items()},
        "metrics": results,
        "target_description": "Whether the customer churned (left the company) - 1 = Yes, 0 = No",
        "churn_rate": float(y.mean().round(4)),
    }

    with open(os.path.join(model_dir, "churn_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model and metadata to {model_dir}")
    print(f"Churn rate in dataset: {metadata['churn_rate']:.1%}")


if __name__ == "__main__":
    main()
