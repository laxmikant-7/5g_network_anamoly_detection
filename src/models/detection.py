"""Reproducible candidate estimators and binary-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


class ArrayInputClassifier(ClassifierMixin, BaseEstimator):
    """Keep ordered DataFrame inputs while fitting libraries with name limits."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(
        self,
        features: object,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ArrayInputClassifier:
        matrix = self._matrix(features, check_order=False)
        if isinstance(features, pd.DataFrame):
            self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = matrix.shape[1]
        if sample_weight is None:
            self.estimator_ = self.estimator.fit(matrix, labels)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if (
                weights.ndim != 1
                or weights.size != matrix.shape[0]
                or not np.isfinite(weights).all()
                or np.any(weights <= 0.0)
            ):
                raise ValueError("sample_weight must be positive finite values per row.")
            self.estimator_ = self.estimator.fit(
                matrix,
                labels,
                sample_weight=weights,
            )
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.estimator_.classes_

    def predict_proba(self, features: object) -> np.ndarray:
        return self.estimator_.predict_proba(self._matrix(features, check_order=True))

    def predict(self, features: object) -> np.ndarray:
        return self.estimator_.predict(self._matrix(features, check_order=True))

    def _matrix(self, features: object, *, check_order: bool) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            if (
                check_order
                and hasattr(self, "feature_names_in_")
                and not np.array_equal(
                    np.asarray(features.columns, dtype=object),
                    self.feature_names_in_,
                )
            ):
                raise ValueError("Feature names or order do not match model training.")
            values = features.to_numpy(dtype=np.float32)
        else:
            values = np.asarray(features, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] == 0
            or not np.isfinite(values).all()
            or (
                hasattr(self, "n_features_in_")
                and values.shape[1] != self.n_features_in_
            )
        ):
            raise ValueError("Model input must be a finite 2D feature matrix.")
        return values


@dataclass(frozen=True)
class DetectionCandidate:
    """One named estimator and its JSON-serializable training configuration."""

    name: str
    estimator: Any
    configuration: dict[str, Any]


def build_detection_candidates(
    y_train: np.ndarray,
    *,
    seed: int,
) -> list[DetectionCandidate]:
    """Build the baseline, unweighted, and class-weighted model candidates."""
    raw_labels = np.asarray(y_train)
    if (
        raw_labels.ndim != 1
        or raw_labels.size == 0
        or not np.isin(raw_labels, (0, 1)).all()
        or set(np.unique(raw_labels)) != {0, 1}
    ):
        raise ValueError("y_train must contain both binary classes encoded as 0 and 1.")
    labels = raw_labels.astype(np.int8)
    negative_count = int(np.count_nonzero(labels == 0))
    positive_count = int(np.count_nonzero(labels == 1))
    imbalance_ratio = negative_count / positive_count
    candidates = [
        DetectionCandidate(
            name="dummy_prior",
            estimator=DummyClassifier(strategy="prior"),
            configuration={"estimator": "DummyClassifier", "strategy": "prior"},
        ),
        DetectionCandidate(
            name="random_forest_unweighted",
            estimator=RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight=None,
                n_jobs=-1,
                random_state=seed,
            ),
            configuration={
                "estimator": "RandomForestClassifier",
                "n_estimators": 250,
                "min_samples_leaf": 2,
                "class_weight": None,
                "random_state": seed,
            },
        ),
        DetectionCandidate(
            name="random_forest_balanced",
            estimator=RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
            configuration={
                "estimator": "RandomForestClassifier",
                "n_estimators": 250,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": seed,
            },
        ),
    ]
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    if XGBClassifier is not None:
        shared_xgboost_parameters = {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": seed,
            "verbosity": 0,
        }
        for name, scale_pos_weight in (
            ("xgboost_unweighted", 1.0),
            ("xgboost_weighted", imbalance_ratio),
        ):
            parameters = {
                **shared_xgboost_parameters,
                "scale_pos_weight": scale_pos_weight,
            }
            candidates.append(
                DetectionCandidate(
                    name=name,
                    estimator=ArrayInputClassifier(XGBClassifier(**parameters)),
                    configuration={
                        "estimator": "XGBClassifier",
                        **parameters,
                    },
                )
            )
    return candidates


def threshold_comparison(
    y_true: np.ndarray,
    anomaly_probability: np.ndarray,
) -> tuple[float, list[dict[str, float | int]]]:
    """Choose the validation threshold that maximizes F2.

    Ties are resolved by recall, precision, lower false-positive rate, then
    the higher threshold. Decisions use ``probability >= threshold``.
    """
    labels, probabilities = _validate_binary_inputs(y_true, anomaly_probability)
    thresholds = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    decisions = probabilities[:, np.newaxis] >= thresholds[np.newaxis, :]
    positive = labels[:, np.newaxis] == 1
    negative = ~positive
    true_positive = np.count_nonzero(decisions & positive, axis=0)
    false_positive = np.count_nonzero(decisions & negative, axis=0)
    false_negative = np.count_nonzero(~decisions & positive, axis=0)
    true_negative = np.count_nonzero(~decisions & negative, axis=0)

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    false_positive_rate = _safe_divide(
        false_positive,
        false_positive + true_negative,
    )
    f1 = _f_score(precision, recall, beta=1.0)
    f2 = _f_score(precision, recall, beta=2.0)
    rows = [
        {
            "threshold": float(threshold),
            "accuracy": float((tp + tn) / labels.size),
            "precision": float(p),
            "recall": float(r),
            "f1": float(one),
            "f2": float(two),
            "false_positive_rate": float(fpr),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
        }
        for threshold, p, r, one, two, fpr, tp, fp, fn, tn in zip(
            thresholds,
            precision,
            recall,
            f1,
            f2,
            false_positive_rate,
            true_positive,
            false_positive,
            false_negative,
            true_negative,
            strict=True,
        )
    ]
    best_index = min(
        range(len(rows)),
        key=lambda index: (
            -rows[index]["f2"],
            -rows[index]["recall"],
            -rows[index]["precision"],
            rows[index]["false_positive_rate"],
            -rows[index]["threshold"],
        ),
    )
    return float(rows[best_index]["threshold"]), rows


def _validate_binary_inputs(
    y_true: np.ndarray,
    anomaly_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = np.asarray(y_true)
    probabilities = np.asarray(anomaly_probability, dtype=np.float64)
    if (
        raw_labels.ndim != 1
        or probabilities.ndim != 1
        or raw_labels.size == 0
        or raw_labels.size != probabilities.size
        or not np.isin(raw_labels, (0, 1)).all()
    ):
        raise ValueError("Labels and probabilities must be aligned non-empty binary vectors.")
    labels = raw_labels.astype(np.int8)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("Anomaly probabilities must be finite values between 0 and 1.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both binary classes.")
    return labels, probabilities


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _f_score(
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    return np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator != 0,
    )
