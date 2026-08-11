"""
Evaluation metrics for multiclass DDoS classification.

Provides per-class and macro/weighted F1, precision, recall,
plus confusion matrix computation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict:
    """
    Compute per-class & aggregate classification metrics.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1,
        per_class (list of dicts with precision, recall, f1, support)
        confusion_matrix (C×C array)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class.append({
            "class": c,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        })

    # Macro F1
    macro_f1 = np.mean([m["f1"] for m in per_class])

    # Weighted F1
    total = sum(m["support"] for m in per_class)
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_class) / total
        if total > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def print_classification_report(
    metrics: Dict,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Pretty-print a classification report.

    Returns the report as a string.
    """
    lines = []
    header = f"{'Class':>20s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'Support':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for m in metrics["per_class"]:
        name = (
            class_names[m["class"]]
            if class_names and m["class"] < len(class_names)
            else str(m["class"])
        )
        lines.append(
            f"{name:>20s}  {m['precision']:7.4f}  {m['recall']:7.4f}  "
            f"{m['f1']:7.4f}  {m['support']:8d}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"{'Accuracy':>20s}  {'':>7s}  {'':>7s}  {metrics['accuracy']:7.4f}  "
        f"{sum(m['support'] for m in metrics['per_class']):8d}"
    )
    lines.append(
        f"{'Macro F1':>20s}  {'':>7s}  {'':>7s}  {metrics['macro_f1']:7.4f}")
    lines.append(
        f"{'Weighted F1':>20s}  {'':>7s}  {'':>7s}  {metrics['weighted_f1']:7.4f}"
    )

    report = "\n".join(lines)
    print(report)
    return report
