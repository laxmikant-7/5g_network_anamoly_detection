"""Anomaly-type diagnosis estimators and string-label inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from cl5g.models.detection import ArrayInputClassifier


class StringLabelClassifier(ClassifierMixin, BaseEstimator):
    """Encode string targets for training and return original labels at inference."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(
        self,
        features: object,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> StringLabelClassifier:
        raw_labels = np.asarray(labels, dtype=object)
        if (
            raw_labels.ndim != 1
            or raw_labels.size == 0
            or any(not isinstance(label, str) or not label.strip() for label in raw_labels)
        ):
            raise ValueError("Diagnosis labels must be a non-empty vector of strings.")
        self.classes_ = np.asarray(sorted(set(raw_labels)), dtype=object)
        if self.classes_.size < 2:
            raise ValueError("Diagnosis training requires at least two classes.")
        class_index = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([class_index[label] for label in raw_labels], dtype=np.int32)
        self.estimator_ = ArrayInputClassifier(self.estimator).fit(
            features,
            encoded,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, features: object) -> np.ndarray:
        encoded = self.estimator_.predict(features)
        return self.classes_[np.asarray(encoded, dtype=np.int64)]

    def predict_proba(self, features: object) -> np.ndarray:
        return self.estimator_.predict_proba(features)


@dataclass(frozen=True)
class DiagnosisCandidate:
    """One named multiclass estimator and its training configuration."""

    name: str
    estimator: StringLabelClassifier
    configuration: dict[str, Any]
    balanced_sample_weight: bool = False


def build_diagnosis_candidates(
    labels: np.ndarray,
    *,
    seed: int,
) -> list[DiagnosisCandidate]:
    """Build unweighted and class-balanced tree and optional XGBoost candidates."""
    raw_labels = np.asarray(labels, dtype=object)
    if (
        raw_labels.ndim != 1
        or raw_labels.size == 0
        or any(not isinstance(label, str) or not label.strip() for label in raw_labels)
    ):
        raise ValueError("Training labels must be a non-empty vector of strings.")
    classes, counts = np.unique(raw_labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("Diagnosis training requires at least two anomaly categories.")
    class_counts = {
        str(label): int(count)
        for label, count in zip(classes, counts, strict=True)
    }
    forest_parameters = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": seed,
    }
    candidates = [
        DiagnosisCandidate(
            name="random_forest_unweighted",
            estimator=StringLabelClassifier(RandomForestClassifier(**forest_parameters)),
            configuration={
                "estimator": "RandomForestClassifier",
                **forest_parameters,
                "class_weight": None,
            },
        ),
        DiagnosisCandidate(
            name="random_forest_balanced",
            estimator=StringLabelClassifier(
                RandomForestClassifier(
                    **forest_parameters,
                    class_weight="balanced",
                )
            ),
            configuration={
                "estimator": "RandomForestClassifier",
                **forest_parameters,
                "class_weight": "balanced",
            },
        ),
        DiagnosisCandidate(
            name="extra_trees_balanced",
            estimator=StringLabelClassifier(
                ExtraTreesClassifier(
                    **forest_parameters,
                    class_weight="balanced",
                )
            ),
            configuration={
                "estimator": "ExtraTreesClassifier",
                **forest_parameters,
                "class_weight": "balanced",
            },
        ),
    ]
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    if XGBClassifier is not None:
        parameters = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "num_class": int(classes.size),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": seed,
            "verbosity": 0,
        }
        candidates.append(
            DiagnosisCandidate(
                name="xgboost_balanced",
                estimator=StringLabelClassifier(XGBClassifier(**parameters)),
                configuration={
                    "estimator": "XGBClassifier",
                    **parameters,
                    "class_weighting": "inverse class frequency sample weights",
                    "training_class_counts": class_counts,
                },
                balanced_sample_weight=True,
            )
        )
    return candidates


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Give each category equal aggregate weight without resampling examples."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if counts.size < 2:
        raise ValueError("At least two classes are required for balanced weights.")
    return (len(inverse) / (counts.size * counts[inverse])).astype(np.float64)


def fit_diagnosis(
    model: StringLabelClassifier,
    features: pd.DataFrame,
    anomaly_types: pd.Series,
) -> StringLabelClassifier:
    """Fit the supplied diagnosis estimator on labeled feature rows.

    Raises:
        TypeError: If an argument has the wrong type.
    """
    if not isinstance(model, StringLabelClassifier):
        raise TypeError("model must be a StringLabelClassifier.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if not isinstance(anomaly_types, pd.Series):
        raise TypeError("anomaly_types must be a pandas Series.")
    if len(features) != len(anomaly_types):
        raise ValueError("features and anomaly_types must have equal row counts.")
    model.fit(features, anomaly_types.to_numpy(dtype=object))
    return model


def predict_anomaly_type(model: object, features: pd.DataFrame) -> pd.Series:
    """Predict anomaly types with a fitted diagnosis model.

    Raises:
        TypeError: If an argument has the wrong type.
    """
    if model is None:
        raise TypeError("model is required.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("model must provide predict().")
    values = np.asarray(predict(features), dtype=object)
    if values.ndim != 1 or len(values) != len(features):
        raise ValueError("model must return one diagnosis for every feature row.")
    return pd.Series(values, index=features.index, name="anomaly_type")
