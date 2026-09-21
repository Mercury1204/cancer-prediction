"""
Breast Cancer Diagnostic Prediction Pipeline

Benchmarks Logistic Regression, SVC, Decision Tree, and Random Forest.
Optimizes the primary model for Recall (Sensitivity) to minimize Type II errors.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

DATA_URL = "https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_cancer_pipeline.joblib")


def load_and_preprocess_data(url: str):
    """Ingest, audit, and encode the WDBC dataset."""
    df = pd.read_csv(url)

    # Strip unnecessary identification markers or trailing unindexed columns
    cols_to_drop = [col for col in ["id", "Unnamed: 32"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Encode Diagnosis: M (Malignant) -> 1, B (Benign) -> 0
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def evaluate_model(name: str, model, X_test, y_test):
    """Evaluate a fitted model and print a structured performance summary."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n--- {name} Performance ---")
    print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f} | F1-Score:  {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(f"Confusion: TN={tn}, FP={fp}, FN={fn} (Type II), TP={tp}")

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[1/4] Ingesting and auditing WDBC dataset...")
    X, y = load_and_preprocess_data(DATA_URL)

    # Stratified 80:20 split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(
        f"Dataset split: {X_train.shape[0]} training samples, "
        f"{X_test.shape[0]} testing samples."
    )

    # Model dictionary with leak-free StandardScaler integration
    candidate_pipelines = {
        "Logistic Regression (L2)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Support Vector Classifier (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "Decision Tree (Pruned)": Pipeline(
            [("clf", DecisionTreeClassifier(max_depth=4, random_state=42))]
        ),
        "Random Forest Ensemble": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100, max_depth=5, random_state=42
                    ),
                )
            ]
        ),
    }

    print("\n[2/4] Benchmarking baseline architectures...")
    for name, pipeline in candidate_pipelines.items():
        pipeline.fit(X_train, y_train)
        evaluate_model(name, pipeline, X_test, y_test)

    print(
        "\n[3/4] Tuning SVC Hyperparameters via GridSearchCV "
        "(Scoring='recall')..."
    )
    svc_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ]
    )

    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.1],
        "clf__kernel": ["rbf", "linear"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        svc_pipeline, param_grid, cv=cv, scoring="recall", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Optimal Parameters: {grid.best_params_}")
    print(f"Best CV Recall:     {grid.best_score_:.4f}")

    # Final holdout evaluation of the optimized estimator
    best_model = grid.best_estimator_
    print("\n[4/4] Final Evaluation on Unseen Test Partition:")
    evaluate_model("Optimized SVC Pipeline", best_model, X_test, y_test)

    # Persist the end-to-end pipeline (scaler + estimator)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nTrained pipeline serialized to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
