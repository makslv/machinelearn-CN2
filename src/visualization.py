from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import os

TEXT = "#EAEAF0"
TICKS = "#CFCFE6"
GRID = "#6B6B8C"
BG = "#050505"
ACCENT = "#B026FF"



def plot_tree_confusion_matrix(cm, output_path):
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", cbar=True)

    plt.title("Decision Tree — Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

    plt.savefig(output_path, dpi=300)  
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str = "outputs/plots/confusion_matrix.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Purples", colorbar=False, text_kw={"color": TEXT})
    ax.set_title("CN2 — confusion matrix (test set)", color=TEXT, fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted", color=TEXT)
    ax.set_ylabel("True", color=TEXT)
    ax.tick_params(colors=TICKS)
    for spine in ax.spines.values():
        spine.set_color(TICKS)
    plt.tight_layout()
    plt.savefig(path, dpi=300, facecolor=BG)
    plt.close()


def plot_accuracy_bars(train_acc: float, test_acc: float, path: str = "outputs/plots/cn2_accuracy.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)
    xs = np.arange(2)
    vals = [train_acc, test_acc]
    bars = ax.bar(xs, vals, color=[ACCENT, "#00FFE0"], width=0.55, edgecolor=TICKS, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Train accuracy", "Test accuracy"], color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_title("CN2 — accuracy (Iris)", color=TEXT, fontsize=15, fontweight="bold")
    ax.tick_params(colors=TICKS)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=GRID)
    for spine in ax.spines.values():
        spine.set_color(TICKS)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", color=TEXT, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300, facecolor=BG)
    plt.close()


def plot_rule_counts(n_rules: int, path: str = "outputs/plots/cn2_rule_count.png") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh([0], [n_rules], color=ACCENT, height=0.35, edgecolor=TICKS)
    ax.set_yticks([0])
    ax.set_yticklabels(["Ordered rules"], color=TEXT)
    ax.set_xlabel("Count (incl. default rule)", color=TEXT)
    ax.set_title("CN2 — induced rule list size", color=TEXT, fontsize=15, fontweight="bold")
    ax.tick_params(colors=TICKS)
    ax.text(n_rules + 0.05, 0, str(n_rules), va="center", color=TEXT, fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(TICKS)
    ax.grid(True, axis="x", linestyle="--", alpha=0.25, color=GRID)
    plt.tight_layout()
    plt.savefig(path, dpi=300, facecolor=BG)
    plt.close()

