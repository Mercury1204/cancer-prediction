"""
evaluate.py — Model Evaluation Utilities

Generates confusion matrices, classification reports, ROC-AUC scores,
and structured performance summaries for any fitted sklearn estimator.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(name: str, model, X_test, y_test, verbose: bool = True) -> dict:
    """
    Evaluate a fitted model on the test partition.

    Parameters
    ----------
    name : str
        Human-readable model identifier.
    model : sklearn estimator or Pipeline
        Fitted model supporting .predict() and optionally .predict_proba().
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        Ground-truth labels.
    verbose : bool
        If True, print results to stdout.

    Returns
    -------
    dict
        Dictionary of metric name -> value.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    if verbose:
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
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_classification_report(model, X_test, y_test):
    """Print a full sklearn classification report with labeled classes."""
    y_pred = model.predict(X_test)
    print(
        classification_report(
            y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]
        )
    )
