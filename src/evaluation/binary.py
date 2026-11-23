"""Binary anomaly-detection metrics and evaluation visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from cl5g.models.detection import _validate_binary_inputs


def evaluate_binary(
    y_true: np.ndarray,
    anomaly_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate binary metrics at a fixed, previously selected threshold."""
    labels, probabilities = _validate_binary_inputs(y_true, anomaly_probability)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric.")
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be finite and between 0 and 1.")
    predictions = probabilities >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = _f_score(precision, recall, beta=1.0)
    f2 = _f_score(precision, recall, beta=2.0)
    classification = classification_report(
        labels,
        predictions.astype(np.int8),
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "class_wise": {
            "Normal": _class_summary(classification["Normal"]),
            "Anomaly": _class_summary(classification["Anomaly"]),
        },
        "classification_report": _json_safe(classification),
    }


def write_threshold_comparison(
    rows: list[dict[str, float | int]],
    selected_threshold: float,
    output_directory: Path,
) -> None:
    """Persist validation-only threshold comparisons as CSV and a figure."""
    output_directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_directory / "validation_threshold_comparison.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 5))
    for metric in ("precision", "recall", "f2", "false_positive_rate"):
        axis.plot(table["threshold"], table[metric], label=metric.replace("_", " "))
    axis.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"selected threshold = {selected_threshold:.4f}",
    )
    axis.set(
        title="Validation threshold trade-off (test set not used)",
        xlabel="Anomaly probability threshold",
        ylabel="Metric",
        ylim=(0.0, 1.02),
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_directory / "validation_threshold_comparison.png", dpi=150)
    plt.close(fig)


def write_test_figures(
    y_true: np.ndarray,
    anomaly_probability: np.ndarray,
    threshold: float,
    summary: dict[str, Any],
    output_directory: Path,
) -> None:
    """Write final-test confusion, precision-recall, and ROC figures."""
    output_directory.mkdir(parents=True, exist_ok=True)
    labels, probabilities = _validate_binary_inputs(y_true, anomaly_probability)
    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        title=f"Test confusion matrix (threshold={threshold:.4f})",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Normal", "Anomaly"),
        yticklabels=("Normal", "Anomaly"),
    )
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_directory / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label="test precision-recall curve")
    axis.scatter(
        summary["recall"],
        summary["precision"],
        color="red",
        label="selected operating point",
    )
    axis.set(
        title=f"Test precision-recall curve (AP={summary['average_precision']:.4f})",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_directory / "test_precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    axis.plot(false_positive_rate, true_positive_rate, label="test ROC curve")
    axis.plot((0.0, 1.0), (0.0, 1.0), linestyle="--", color="gray")
    axis.scatter(
        summary["false_positive_rate"],
        summary["recall"],
        color="red",
        label="selected operating point",
    )
    axis.set(
        title=f"Test ROC curve (AUC={summary['roc_auc']:.4f})",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_directory / "test_roc_curve.png", dpi=150)
    plt.close(fig)


def _class_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": float(value["precision"]),
        "recall": float(value["recall"]),
        "f1_score": float(value["f1-score"]),
        "support": int(value["support"]),
    }


def _f_score(precision: float, recall: float, *, beta: float) -> float:
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    return (
        0.0
        if denominator == 0.0
        else (1.0 + beta_squared) * precision * recall / denominator
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
