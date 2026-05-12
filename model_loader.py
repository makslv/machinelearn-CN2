"""Train-state for CN2 + Decision Tree used by web_app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.cn2 import BIN_LABELS, CN2Classifier  # noqa: E402
from src.data_preprocessing import load_iris_split  # noqa: E402
from src.decision_tree import train_decision_tree  # noqa: E402

_state: dict[str, Any] | None = None


def _dataset_iris() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    X_train, X_test, y_train, y_test = load_iris_split()
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    feature_names = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]
    class_names = ["Setosa", "Versicolor", "Virginica"]
    return X, y, feature_names, class_names


def _dataset_weather() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    # temperature C, humidity %, wind km/h, pressure hPa; target: 1=Play, 0=Skip
    X = np.array(
        [
            [30, 85, 12, 1008], [28, 90, 18, 1005], [24, 60, 8, 1015], [22, 55, 6, 1018],
            [26, 70, 10, 1011], [20, 50, 7, 1019], [32, 88, 20, 1004], [18, 45, 5, 1021],
            [27, 65, 11, 1010], [25, 58, 9, 1014], [31, 92, 21, 1003], [19, 48, 6, 1020],
            [23, 52, 7, 1017], [29, 80, 14, 1009], [21, 54, 6, 1019], [33, 95, 24, 1002],
            [17, 42, 4, 1022], [24, 57, 8, 1016], [26, 75, 12, 1010], [22, 53, 7, 1018],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int64)
    feature_names = ["Temperature", "Humidity", "Wind speed", "Pressure"]
    class_names = ["Skip", "Play"]
    return X, y, feature_names, class_names


def _dataset_loan() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    # income k$, debt ratio, credit score, employment years; target: 1=Approve, 0=Reject
    X = np.array(
        [
            [25, 0.55, 520, 1], [40, 0.35, 640, 3], [75, 0.20, 720, 8], [62, 0.28, 690, 6],
            [18, 0.60, 500, 0], [55, 0.30, 670, 5], [90, 0.18, 760, 10], [33, 0.45, 610, 2],
            [48, 0.32, 655, 4], [70, 0.25, 705, 7], [22, 0.58, 510, 1], [65, 0.22, 715, 8],
            [38, 0.40, 625, 3], [80, 0.19, 740, 9], [28, 0.50, 560, 2], [58, 0.27, 680, 6],
            [45, 0.34, 650, 4], [95, 0.15, 780, 12], [30, 0.47, 590, 2], [72, 0.24, 710, 7],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=np.int64)
    feature_names = ["Income (k$)", "Debt ratio", "Credit score", "Employment years"]
    class_names = ["Reject", "Approve"]
    return X, y, feature_names, class_names


def _load_dataset(kind: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    key = (kind or "iris").strip().lower()
    if key == "iris":
        return _dataset_iris()
    if key == "weather":
        return _dataset_weather()
    if key == "loan":
        return _dataset_loan()
    raise ValueError("Unknown dataset. Use iris, weather, or loan.")


def _train_state(
    *,
    dataset: str = "iris",
    test_size: float = 0.25,
    random_state: int = 42,
    cn2_params: dict[str, Any] | None = None,
    tree_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    X, y, feature_names, class_names = _load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    cp = cn2_params or {}
    clf = CN2Classifier(
        n_bins=int(cp.get("n_bins", 3)),
        beam_width=int(cp.get("beam_width", 6)),
        max_conditions=int(cp.get("max_conditions", 5)),
        min_rule_coverage=int(cp.get("min_rule_coverage", 4)),
        random_state=random_state,
        test_fraction=test_size,
    )
    clf.feature_names = list(feature_names)
    clf.feature_names_short = [n[:12] for n in feature_names]
    clf.class_names = list(class_names)
    clf.fit_arrays(X_train, y_train)

    tp = tree_params or {}
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=int(tp.get("max_depth", 4)),
        min_samples_leaf=int(tp.get("min_samples_leaf", 2)),
        random_state=random_state,
    )
    tree.fit(X_train, y_train)
    ytr_t, yte_t = tree.predict(X_train), tree.predict(X_test)
    ytr_c, yte_c = clf.predict(X_train), clf.predict(X_test)
    return {
        "dataset": dataset,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "classifier": clf,
        "tree": tree,
        "meta": {
            "dataset": dataset,
            "n_samples_train": int(X_train.shape[0]),
            "n_samples_test": int(X_test.shape[0]),
            "n_classes": int(len(class_names)),
            "n_features": int(X_train.shape[1]),
            "class_names": class_names,
            "feature_names": feature_names,
            "cn2_train_accuracy": float(accuracy_score(y_train, ytr_c)),
            "cn2_test_accuracy": float(accuracy_score(y_test, yte_c)),
            "tree_train_accuracy": float(accuracy_score(y_train, ytr_t)),
            "tree_test_accuracy": float(accuracy_score(y_test, yte_t)),
            "cn2_n_rules": int(len(clf.rules_)),
            "tree_depth": int(tree.get_depth()),
            "tree_n_leaves": int(tree.get_n_leaves()),
            "cn2_params": {
                "n_bins": clf.n_bins,
                "beam_width": clf.beam_width,
                "max_conditions": clf.max_conditions,
                "min_rule_coverage": clf.min_rule_coverage,
            },
            "tree_params": {
                "criterion": "entropy",
                "max_depth": int(tp.get("max_depth", 4)),
                "min_samples_leaf": int(tp.get("min_samples_leaf", 2)),
            },
        },
    }


def ensure_state() -> dict[str, Any]:
    global _state
    if _state is None:
        _state = _train_state(dataset="iris")
    return _state


def retrain(**kwargs: Any) -> dict[str, Any]:
    global _state
    _state = _train_state(**kwargs)
    return _state


def get_classifier() -> CN2Classifier:
    return ensure_state()["classifier"]


def get_decision_tree() -> DecisionTreeClassifier:
    return ensure_state()["tree"]


def get_fit_meta() -> dict[str, Any]:
    return dict(ensure_state()["meta"])


def get_train_test() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = ensure_state()
    return s["X_train"], s["X_test"], s["y_train"], s["y_test"]


__all__ = [
    "BIN_LABELS",
    "ensure_state",
    "get_classifier",
    "get_decision_tree",
    "get_fit_meta",
    "get_train_test",
    "retrain",
]
