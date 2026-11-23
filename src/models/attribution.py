"""SHAP explanations grouped from engineered features to source KPI channels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


EXPLANATION_CAVEAT = (
    "Feature importance and SHAP values describe model associations; they do "
    "not establish physical causality."
)
AGGREGATION_METHOD = (
    "KPI importance is the sum of absolute SHAP contributions for all "
    "engineered features mapped to that KPI. Signed contributions are also "
    "summed separately to show net direction."
)


class KPIAttributionExplainer:
    """Create local TreeSHAP explanations and aggregate them by original KPI."""

    def __init__(
        self,
        feature_configuration: Mapping[str, Any],
        *,
        top_feature_count: int = 10,
    ) -> None:
        feature_order = feature_configuration.get("feature_order")
        feature_definitions = feature_configuration.get("features")
        if (
            not isinstance(feature_order, list)
            or not feature_order
            or not isinstance(feature_definitions, list)
        ):
            raise ValueError("A valid ordered feature configuration is required.")
        if isinstance(top_feature_count, bool) or top_feature_count < 1:
            raise ValueError("top_feature_count must be a positive integer.")
        definitions = {
            definition.get("name"): definition
            for definition in feature_definitions
            if isinstance(definition, Mapping)
        }
        if set(feature_order) != set(definitions):
            raise ValueError("Feature definitions do not match feature_order.")
        self.feature_order = tuple(feature_order)
        self.feature_to_kpi = {
            name: definitions[name].get("source_kpi", definitions[name].get("kpi"))
            for name in self.feature_order
        }
        if any(not isinstance(kpi, str) or not kpi for kpi in self.feature_to_kpi.values()):
            raise ValueError("Every feature must map to an original source KPI.")
        self.kpi_order = tuple(feature_configuration["kpi_names"])
        self.top_feature_count = top_feature_count

    def explain_prediction(
        self,
        features: pd.DataFrame,
        *,
        model: object,
        model_kind: str,
        predicted_class: object,
        probability: float,
    ) -> dict[str, Any]:
        """Explain one class output and group its feature contributions by KPI."""
        values, base_value = self.shap_contributions(
            features,
            model=model,
            predicted_class=predicted_class,
        )
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float, np.number))
            or not np.isfinite(probability)
            or probability < 0.0
            or probability > 1.0
        ):
            raise ValueError("probability must be a finite value between 0 and 1.")

        row = features.iloc[0].to_numpy(dtype=np.float64)
        feature_rows = [
            {
                "feature": name,
                "value": float(row[index]),
                "contribution": float(values[index]),
                "absolute_contribution": float(abs(values[index])),
                "source_kpi": self.feature_to_kpi[name],
            }
            for index, name in enumerate(self.feature_order)
        ]
        feature_rows.sort(
            key=lambda item: (-item["absolute_contribution"], item["feature"])
        )
        kpi_scores = self.aggregate_by_kpi(values)
        ranked_kpis = sorted(
            self.kpi_order,
            key=lambda kpi: (-kpi_scores[kpi]["importance"], self.kpi_order.index(kpi)),
        )
        return {
            "model_kind": model_kind,
            "method": "TreeSHAP",
            "predicted_class": _json_scalar(predicted_class),
            "prediction_probability": float(probability),
            "baseline_value": float(base_value),
            "output_value": float(base_value + np.sum(values)),
            "aggregation_method": AGGREGATION_METHOD,
            "interpretation": EXPLANATION_CAVEAT,
            "top_features": feature_rows[: self.top_feature_count],
            "top_kpis": [
                {
                    "kpi": kpi,
                    **kpi_scores[kpi],
                }
                for kpi in ranked_kpis
                if kpi_scores[kpi]["importance"] > 0.0
            ],
        }

    def shap_contributions(
        self,
        features: pd.DataFrame,
        *,
        model: object,
        predicted_class: object,
    ) -> tuple[np.ndarray, float]:
        """Return one row of class-specific SHAP values and its base value."""
        if not isinstance(features, pd.DataFrame) or len(features) != 1:
            raise TypeError("features must be a one-row pandas DataFrame.")
        self._validate_feature_frame(features)
        class_values, estimator = _unwrap_estimator(model, predicted_class)
        class_index = _class_index(class_values, predicted_class)
        try:
            import shap

            explainer = shap.TreeExplainer(estimator)
            raw_values = explainer.shap_values(features)
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"TreeSHAP could not explain this model: {exc}") from exc
        contributions = _select_shap_values(
            raw_values,
            class_index=class_index,
            sample_count=1,
            feature_count=len(self.feature_order),
            class_count=len(class_values),
        )
        baseline = _select_base_value(
            explainer.expected_value,
            class_index=class_index,
            class_count=len(class_values),
        )
        if not np.isfinite(contributions).all():
            raise RuntimeError("TreeSHAP returned non-finite attribution values.")
        return contributions[0], baseline

    def shap_values(
        self,
        features: pd.DataFrame,
        *,
        model: object,
        predicted_class: object,
    ) -> np.ndarray:
        """Return class-specific TreeSHAP values for an ordered feature table."""
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise TypeError("features must be a non-empty pandas DataFrame.")
        self._validate_feature_frame(features)
        matrix = features.to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("Feature values must be finite.")
        classes, estimator = _unwrap_estimator(model, predicted_class)
        class_index = _class_index(classes, predicted_class)
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError("SHAP is required to produce model explanations.") from exc
        try:
            explainer = shap.TreeExplainer(estimator)
            raw_values = explainer.shap_values(features)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"TreeSHAP could not explain this model: {exc}") from exc
        contributions = _select_shap_values(
            raw_values,
            class_index=class_index,
            sample_count=len(features),
            feature_count=len(self.feature_order),
            class_count=len(classes),
        )
        if not np.isfinite(contributions).all():
            raise RuntimeError("TreeSHAP returned non-finite attribution values.")
        return contributions

    def _validate_feature_frame(self, features: pd.DataFrame) -> None:
        if tuple(features.columns) != self.feature_order:
            raise ValueError("Feature names or order do not match the saved contract.")

    def aggregate_by_kpi(self, contributions: np.ndarray) -> dict[str, dict[str, float]]:
        """Group contributions using sum-absolute importance and signed sum."""
        values = np.asarray(contributions, dtype=np.float64)
        if values.shape != (len(self.feature_order),) or not np.isfinite(values).all():
            raise ValueError("SHAP contributions do not match the feature order.")
        scores = {
            kpi: {"importance": 0.0, "signed_contribution": 0.0}
            for kpi in self.kpi_order
        }
        for feature, contribution in zip(self.feature_order, values, strict=True):
            kpi = self.feature_to_kpi[feature]
            scores[kpi]["importance"] += float(abs(contribution))
            scores[kpi]["signed_contribution"] += float(contribution)
        return scores


def _unwrap_estimator(model: object, predicted_class: object) -> tuple[list[object], object]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Explained model must expose classes_.")
    class_values = list(classes)
    if not class_values:
        raise ValueError("Explained model has no fitted classes.")
    _class_index(class_values, predicted_class)

    estimator = model
    if estimator.__class__.__name__ == "StringLabelClassifier":
        estimator = getattr(estimator, "estimator_", None)
    if estimator is not None and estimator.__class__.__name__ == "ArrayInputClassifier":
        estimator = getattr(estimator, "estimator_", None)
    if not (
        callable(getattr(estimator, "predict_proba", None))
        and hasattr(estimator, "estimators_")
    ):
        raise ValueError(
            "TreeSHAP currently requires a fitted scikit-learn tree ensemble."
        )
    return class_values, estimator


def _class_index(classes: list[object], label: object) -> int:
    matches = [
        index
        for index, value in enumerate(classes)
        if value == label or str(value).strip().lower() == str(label).strip().lower()
    ]
    if len(matches) != 1:
        raise ValueError("Predicted class does not match exactly one model class.")
    return matches[0]


def _select_shap_values(
    raw_values: object,
    *,
    class_index: int,
    sample_count: int,
    feature_count: int,
    class_count: int,
) -> np.ndarray:
    if isinstance(raw_values, list):
        if class_index >= len(raw_values):
            raise ValueError("SHAP class output does not match model classes.")
        selected = np.asarray(raw_values[class_index], dtype=np.float64)
    else:
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim == 3:
            if values.shape == (sample_count, feature_count, class_count):
                selected = values[:, :, class_index]
            elif values.shape == (sample_count, class_count, feature_count):
                selected = values[:, class_index, :]
            else:
                raise ValueError("SHAP output dimensions do not match model features/classes.")
        elif values.ndim == 2 and values.shape == (sample_count, feature_count):
            selected = values
        elif values.ndim == 1 and values.size == feature_count:
            selected = values[np.newaxis, :]
        else:
            raise ValueError("SHAP output must contain one contribution per feature.")
    if selected.shape == (feature_count,) and sample_count == 1:
        selected = selected[np.newaxis, :]
    if selected.shape != (sample_count, feature_count):
        raise ValueError("SHAP output must contain one contribution per feature per sample.")
    return selected


def _select_base_value(
    raw_value: object,
    *,
    class_index: int,
    class_count: int,
) -> float:
    if isinstance(raw_value, list):
        if class_index >= len(raw_value):
            raise ValueError("SHAP base values do not match model classes.")
        selected = raw_value[class_index]
    else:
        values = np.asarray(raw_value, dtype=np.float64).reshape(-1)
        if values.size == 1:
            selected = values[0]
        elif values.size == class_count:
            selected = values[class_index]
        else:
            raise ValueError("SHAP base values do not match model classes.")
    result = float(np.asarray(selected, dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("SHAP base value is non-finite.")
    return result


def _json_scalar(value: object) -> Any:
    return value.item() if isinstance(value, np.generic) else value
