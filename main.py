
from __future__ import annotations

import os

import pandas as pd
from src.visualization import plot_tree_confusion_matrix
from src.decision_tree import train_decision_tree, evaluate_decision_tree
from src.cn2 import CN2Classifier
from src.data_preprocessing import load_iris_split
from src.evaluation import evaluate_cn2
from src.visualization import plot_accuracy_bars, plot_confusion_matrix, plot_rule_counts

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

X_train, X_test, y_train, y_test = load_iris_split()

model = CN2Classifier()
model.fit_arrays(X_train, y_train)

train_metrics = evaluate_cn2(model, X_train, y_train)
test_metrics = evaluate_cn2(model, X_test, y_test)

print("CN2 train accuracy:", train_metrics["accuracy"])
print("CN2 test accuracy:", test_metrics["accuracy"])
print("\nClassification report (test):\n", test_metrics["classification_report"])

rules_lines = model.export_rules_plain()
with open("outputs/tables/cn2_rules.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(f"{i+1}. {line}" for i, line in enumerate(rules_lines)))

pd.DataFrame(
    {
        "metric": ["train_accuracy", "test_accuracy", "n_rules"],
        "value": [
            train_metrics["accuracy"],
            test_metrics["accuracy"],
            len(model.rules_),
        ],
    }
).to_csv("outputs/tables/cn2_metrics.csv", index=False)

pd.DataFrame(
    {
        "rule_index": range(1, len(rules_lines) + 1),
        "rule": rules_lines,
    }
).to_csv("outputs/tables/cn2_rules.csv", index=False)

plot_confusion_matrix(test_metrics["confusion_matrix"], list(model.class_names))
plot_accuracy_bars(train_metrics["accuracy"], test_metrics["accuracy"])
plot_rule_counts(len(model.rules_))

print("\nSaved: outputs/tables/cn2_rules.txt, cn2_metrics.csv, cn2_rules.csv")
print("Saved: outputs/plots/confusion_matrix.png, cn2_accuracy.png, cn2_rule_count.png")
# Decision Tree
tree_model = train_decision_tree(X_train, y_train)
tree_results = evaluate_decision_tree(tree_model, X_test, y_test)

print("\n=== Decision Tree ===")
print("Accuracy:", tree_results["accuracy"])
print("Confusion matrix:\n", tree_results["confusion_matrix"])
print(tree_results["report"])
plot_tree_confusion_matrix(
    tree_results["confusion_matrix"],
    "outputs/plots/tree_confusion_matrix.png"
)