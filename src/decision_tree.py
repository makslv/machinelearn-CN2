from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def export_tree_ascii(model: DecisionTreeClassifier, feature_names: list[str]) -> str:
    return export_text(model, feature_names=feature_names)


def describe_decision_path(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    x: np.ndarray,
) -> str:
    """Людинозчитуваний шлях від кореня до листка для одного вектора ознак (непреривні значення)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    tree_ = model.tree_
    node = 0
    steps: list[str] = []
    while tree_.children_left[node] != -1:
        f_idx = int(tree_.feature[node])
        thr = float(tree_.threshold[node])
        fname = feature_names[f_idx]
        if x[f_idx] <= thr:
            steps.append(f"{fname} ≤ {thr:g}")
            node = int(tree_.children_left[node])
        else:
            steps.append(f"{fname} > {thr:g}")
            node = int(tree_.children_right[node])
    values = tree_.value[node][0]
    pred = int(np.argmax(values))
    steps.append(f"leaf: {class_names[pred]}")
    return " → ".join(steps)


def evaluate_decision_tree(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "predictions": y_pred
    }