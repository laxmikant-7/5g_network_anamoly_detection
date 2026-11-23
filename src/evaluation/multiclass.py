"""Evaluation and error analysis for multiclass anomaly diagnosis."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from cl5g.data.telecomts import TAXONOMY_LABELS

ANOMALY_CLASSES = tuple(label for label in TAXONOMY_LABELS if label != "Normal")


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_order: tuple[str, ...] = ANOMALY_CLASSES,
) -> dict[str, Any]:
    """Calculate metrics on classes with actual truth support only."""
    truth = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    allowed = set(class_order)
    if (
        truth.ndim != 1
        or predicted.ndim != 1
        or truth.size == 0
        or truth.size != predicted.size
        or any(label not in allowed for label in truth)
        or any(label not in allowed for label in predicted)
    ):
        raise ValueError("Labels and predictions must be aligned supported class vectors.")

    supported = [label for label in class_order if np.any(truth == label)]
    if not supported:
        raise ValueError("Evaluation data must contain at least one supported class.")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=supported,
        zero_division=0,
    )
    confusion = confusion_matrix(truth, predicted, labels=class_order)
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(supported)
    }
    for label in class_order:
        if label not in per_class:
            per_class[label] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
            }

    confusions = [
        {
            "actual": class_order[row],
            "predicted": class_order[column],
            "count": int(confusion[row, column]),
        }
        for row in range(len(class_order))
        for column in range(len(class_order))
        if row != column and confusion[row, column] > 0
    ]
    confusions.sort(key=lambda item: (-item["count"], item["actual"], item["predicted"]))

    return {
        "sample_count": int(truth.size),
        "supported_classes": supported,
        "missing_classes": [label for label in class_order if label not in supported],
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            np.average(f1, weights=support) if support.sum() else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(class_order),
            "counts": confusion.astype(int).tolist(),
        },
        "most_frequent_confusions": confusions[:20],
    }


def compare_validation_test(
    validation: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    """Summarize evaluation differences where both partitions support a class."""
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    )
    common_classes = sorted(
        set(validation["supported_classes"]) & set(test["supported_classes"])
    )
    return {
        "validation_minus_test": {
            metric: float(validation[metric] - test[metric]) for metric in metrics
        },
        "per_class_recall_validation_minus_test": {
            label: float(
                validation["per_class"][label]["recall"]
                - test["per_class"][label]["recall"]
            )
            for label in common_classes
        },
        "classes_missing_from_validation": validation["missing_classes"],
        "classes_missing_from_test": test["missing_classes"],
        "test_exposure_note": (
            "This test partition was previously used for binary-detector "
            "evaluation and is therefore a historical holdout for diagnosis; "
            "diagnosis results are not an untouched pristine-test estimate."
        ),
    }
