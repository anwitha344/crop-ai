"""
Evaluation utilities for both the Random Forest baseline and the full
CNN-LSTM model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute accuracy, precision, recall, F1, and (if probabilities are
    provided and both classes are present) ROC-AUC.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None

    return metrics


def print_metrics(metrics, title="Model Evaluation"):
    print(title)
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    else:
        print("  ROC-AUC:   N/A (only one class present at current dataset size)")
