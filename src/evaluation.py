from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_cn2(model, X: np.ndarray, y_true: np.ndarray) -> dict:
    y_pred = model.predict(X)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(model.class_names),
        digits=4,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": report,
    }
