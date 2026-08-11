"""
Visualization utilities — confusion matrices, training curves, etc.
"""

import numpy as np
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = False,
):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : (C, C) array
    class_names : optional list of class label strings
    save_path : if provided, save figure to this path
    normalize : if True, show row-normalised percentages
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib/seaborn not installed — skipping plot")
        return

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = ".2f"
    else:
        fmt = "d"

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """
    Plot training & validation loss/accuracy curves.

    Parameters
    ----------
    history : dict with keys like "train_loss", "val_loss", "train_f1", "val_f1"
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Loss ---
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- F1 / Accuracy ---
    ax = axes[1]
    if "train_f1" in history:
        ax.plot(history["train_f1"], label="Train F1")
    if "val_f1" in history:
        ax.plot(history["val_f1"], label="Val F1")
    if "train_acc" in history:
        ax.plot(history["train_acc"], label="Train Acc")
    if "val_acc" in history:
        ax.plot(history["val_acc"], label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("F1 / Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_class_distribution(
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
):
    """Bar chart of class counts."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plot")
        return

    classes, counts = np.unique(y, return_counts=True)
    if class_names is None:
        class_names = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, counts,
                  color=sns.color_palette("viridis", len(classes)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right")

    # Annotate bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
