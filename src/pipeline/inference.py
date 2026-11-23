"""Saved-artifact inference for one 128-step, 18-channel network sample.

Detection and diagnosis estimators, preprocessing, attribution, and temporal
localization are joblib artifacts. The feature JSON declares the raw and
processed KPI orders, optional categorical domains, and ordered feature
definitions. Metadata JSON declares ``model_version`` and may include a
``decision_threshold``; when present, inference compares the anomaly
probability with that threshold, while legacy metadata continues to use
``predict()``. Attribution and temporal localization artifacts are optional
and produce null response fields when absent; they are never inferred from
unrelated model outputs.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cl5g.common.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    ProjectError,
    SchemaValidationError,
)
from cl5g.common.paths import display_path
from cl5g.data.telecomts import ALLOWED_PROTOCOL_VALUES, CATEGORICAL_KPIS, KPI_NAMES
from cl5g.data.telecomts_preprocessing import (
    ENCODED_FEATURE_NAMES,
    PROTOCOL_CATEGORIES,
)
from cl5g.features.extraction import (
    extract_features,
    validate_feature_configuration,
)
from cl5g.pipeline.config import AppConfig

TIME_STEPS = 128
KPI_CHANNELS = 18
_ANOMALY_LABELS = {"1", "true", "anomaly", "anomalous", "abnormal", "attack", "positive"}


@dataclass(frozen=True)
class _DetectionOutput:
    """Prepared features and binary detection values for one sample."""

    features: pd.DataFrame
    original_sequence: np.ndarray
    processed: np.ndarray
    anomaly_index: int
    probability: float
    is_anomaly: bool
    model_version: str


class InferenceError(ProjectError):
    """A saved artifact cannot be used to produce a valid inference result."""


class InferencePipeline:
    """Load saved artifacts and make predictions without fitting any estimators."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._loaded: dict[str, Any] = {}
        self._feature_spec: dict[str, Any] | None = None
        self._model_version: str | None = None
        self._model_metadata: dict[str, Any] | None = None

    def predict(
        self,
        sample: object,
        *,
        sampling_rate_hz: float = 10.0,
        kpi_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Return a result for one numeric sample with optional protocol strings."""
        output = self._run_detection(sample, kpi_names=kpi_names)
        if not output.is_anomaly:
            return {
                **_binary_response(output),
                "model_versions": self._model_versions(),
                "predicted_anomaly_type": None,
                "diagnosis_confidence": None,
                "diagnosis_probabilities": None,
                "important_kpis": None,
                "attributions": None,
                "temporal_localization": None,
                "explanation": (
                    f"No anomaly detected by model version {output.model_version}."
                ),
            }

        diagnosis = self._load_joblib(
            "diagnosis",
            self.config.diagnosis.artifact,
            required=True,
        )
        feature_spec = self._load_feature_spec()
        _validate_model_feature_order(
            diagnosis,
            feature_spec["feature_order"],
            "diagnosis",
        )
        try:
            anomaly_type = _single_prediction(
                diagnosis,
                output.features,
                "anomaly diagnosis",
            )
            diagnosis_probabilities = _prediction_probabilities(
                diagnosis,
                output.features,
            )
            if (
                diagnosis_probabilities is not None
                and str(anomaly_type) not in diagnosis_probabilities
            ):
                raise InferenceError(
                    "The diagnosis prediction does not match diagnosis model classes_."
                )
            diagnosis_confidence = (
                diagnosis_probabilities.get(str(anomaly_type))
                if diagnosis_probabilities is not None
                else None
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise InferenceError(f"Diagnosis model failed during prediction: {exc}") from exc
        if anomaly_type is None or not str(anomaly_type).strip():
            raise InferenceError("The diagnosis model returned an empty anomaly type.")
        if (
            not isinstance(anomaly_type, str)
            or anomaly_type not in self.config.schema.anomaly_types
            or anomaly_type.strip().lower() in {"normal", "none", "benign"}
        ):
            raise InferenceError(
                f"The diagnosis model returned unsupported anomaly category "
                f"'{anomaly_type}'. Supported categories are: "
                + ", ".join(
                    category
                    for category in self.config.schema.anomaly_types
                    if category.strip().lower() not in {"normal", "none", "benign"}
                )
                + "."
            )
        attributions = self._attribute_models(
            output,
            diagnosis,
            anomaly_type,
            diagnosis_confidence,
        )
        if attributions is None:
            important_kpis = self._attribute_kpis(
                output.features,
                output.anomaly_index,
            )
        else:
            diagnosis_attribution = attributions.get("multiclass_diagnosis")
            binary_attribution = attributions.get("binary_detection")
            selected = (
                diagnosis_attribution
                if isinstance(diagnosis_attribution, dict)
                else binary_attribution
            )
            important_kpis = (
                [
                    item["kpi"]
                    for item in selected.get("top_kpis", [])[
                        : self.config.inference.attribution_top_k
                    ]
                ]
                if isinstance(selected, dict)
                else None
            )
        temporal_localization = self._localize(output, sampling_rate_hz)
        explanation = _build_explanation(
            anomaly_type,
            output.probability,
            important_kpis,
            temporal_localization,
            output.model_version,
            attributions is not None,
        )
        return {
            "status": "anomaly",
            "anomaly_probability": output.probability,
            "model_versions": self._model_versions(),
            "predicted_anomaly_type": str(anomaly_type),
            "diagnosis_confidence": diagnosis_confidence,
            "diagnosis_probabilities": diagnosis_probabilities,
            "important_kpis": important_kpis,
            "attributions": attributions,
            "temporal_localization": temporal_localization,
            "model_version": output.model_version,
            "explanation": explanation,
        }

    def predict_binary(
        self,
        sample: object,
        *,
        kpi_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Return binary detection only, without requiring diagnosis artifacts."""
        return _binary_response(
            self._run_detection(sample, kpi_names=kpi_names)
        )

    def model_info(self) -> dict[str, Any]:
        """Return saved model versions, supported categories, and feature order."""
        metadata = self._load_model_metadata()
        feature_spec = self._load_feature_spec()
        _validate_input_feature_contract(feature_spec)
        _validate_metadata_feature_order(
            metadata,
            feature_spec["feature_order"],
            "detection",
        )
        versions = metadata.get("model_versions")
        if not isinstance(versions, dict):
            versions = {"pipeline": self._load_model_version()}
        else:
            versions = {
                str(name): str(version)
                for name, version in versions.items()
                if isinstance(name, str) and isinstance(version, (str, int, float))
            }
            if not versions:
                versions = {"pipeline": self._load_model_version()}
        return {
            "model_versions": versions,
            "supported_anomaly_categories": [
                category
                for category in self.config.schema.anomaly_types
                if category.strip().lower() not in {"normal", "none", "benign"}
            ],
            "decision_threshold": self._decision_threshold(),
            "dataset_revision": (
                metadata.get("dataset_revision")
                if isinstance(metadata.get("dataset_revision"), str)
                else None
            ),
            "feature_configuration": {
                "kpi_names": list(feature_spec["kpi_names"]),
                "processed_kpi_names": list(feature_spec["processed_kpi_names"]),
                "categorical_kpis": feature_spec["categorical_kpis"],
                "feature_order": list(feature_spec["feature_order"]),
                "features": [
                    feature_spec["feature_by_name"][name]
                    for name in feature_spec["feature_order"]
                ],
            },
        }

    def _model_versions(self) -> dict[str, str]:
        metadata = self._load_model_metadata()
        raw_versions = metadata.get("model_versions")
        versions = (
            {
                str(name): str(version)
                for name, version in raw_versions.items()
                if isinstance(name, str)
                and isinstance(version, (str, int, float))
            }
            if isinstance(raw_versions, dict)
            else {}
        )
        versions.setdefault("detection", self._load_model_version())
        return versions

    def loading_status(self) -> dict[str, Any]:
        """Report process liveness separately from validated prediction readiness."""
        presence = artifact_presence(self.config)
        required_components = (
            "detection",
            "preprocessing",
            "feature_config",
            "metadata",
        )
        missing = [name for name in required_components if not presence[name]]
        readiness_status = "ready"
        readiness_error: str | None = None
        if missing:
            readiness_status = "missing_artifacts"
            readiness_error = "Required prediction artifacts are missing: " + ", ".join(missing)
        else:
            try:
                self._ensure_prediction_ready()
            except ArtifactNotFoundError as exc:
                readiness_status = "missing_artifacts"
                readiness_error = str(exc)
            except (ConfigurationError, InferenceError) as exc:
                readiness_status = "invalid_artifacts"
                readiness_error = str(exc)

        loaded = {
            "detection": "detection" in self._loaded,
            "preprocessing": "preprocessing" in self._loaded,
            "diagnosis": "diagnosis" in self._loaded,
            "attribution": "attribution" in self._loaded,
            "temporal_localization": "temporal localization" in self._loaded,
            "feature_configuration": self._feature_spec is not None,
            "metadata": self._model_metadata is not None,
        }
        return {
            "liveness_status": "alive",
            "prediction_ready": readiness_status == "ready",
            "readiness_status": readiness_status,
            "readiness_error": readiness_error,
            "model_loading_status": (
                "loaded"
                if all(
                    (
                        loaded["detection"],
                        loaded["preprocessing"],
                        loaded["feature_configuration"],
                        loaded["metadata"],
                    )
                )
                else "unavailable"
            ),
            "loaded_components": loaded,
            "artifacts_present": presence,
        }

    def _ensure_prediction_ready(self) -> None:
        detector = self._load_joblib(
            "detection",
            self.config.inference.detection_artifact,
            required=True,
        )
        preprocessor = self._load_joblib(
            "preprocessing",
            self.config.inference.preprocessing_artifact,
            required=True,
        )
        if not callable(getattr(preprocessor, "transform", None)):
            raise InferenceError("The preprocessing artifact does not provide transform().")
        feature_spec = self._load_feature_spec()
        metadata = self._load_model_metadata()
        _validate_input_feature_contract(feature_spec)
        _validate_metadata_feature_order(
            metadata,
            feature_spec["feature_order"],
            "detection",
        )
        _validate_model_feature_order(
            detector,
            feature_spec["feature_order"],
            "detection",
        )
        self._load_model_version()
        self._decision_threshold()

    def _run_detection(
        self,
        sample: object,
        *,
        kpi_names: list[str] | tuple[str, ...] | None = None,
    ) -> _DetectionOutput:
        values = _validate_sample(sample, kpi_names=kpi_names)
        detector = self._load_joblib(
            "detection",
            self.config.inference.detection_artifact,
            required=True,
        )
        preprocessor = self._load_joblib(
            "preprocessing",
            self.config.inference.preprocessing_artifact,
            required=True,
        )
        feature_spec = self._load_feature_spec()
        _validate_input_feature_contract(feature_spec)
        _validate_metadata_feature_order(
            self._load_model_metadata(),
            feature_spec["feature_order"],
            "detection",
        )
        _validate_model_feature_order(
            detector,
            feature_spec["feature_order"],
            "detection",
        )
        model_version = self._load_model_version()
        _validate_sample_categories(values, feature_spec)

        try:
            preprocessing_input = pd.DataFrame(
                values.copy(),
                columns=feature_spec["kpi_names"],
            )
            transformed = preprocessor.transform(preprocessing_input)
        except (AttributeError, TypeError, ValueError) as exc:
            raise InferenceError(f"Preprocessing artifact failed during transform: {exc}") from exc
        try:
            processed = np.asarray(transformed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InferenceError("Preprocessing returned non-numeric KPI values.") from exc
        expected_shape = (TIME_STEPS, len(feature_spec["processed_kpi_names"]))
        if processed.shape != expected_shape:
            raise InferenceError(
                "The preprocessing artifact returned an incompatible shape; "
                f"expected {expected_shape}, received {processed.shape}."
            )
        if not np.isfinite(processed).all():
            raise InferenceError("Preprocessing produced non-finite KPI values.")

        try:
            features = extract_features(processed, feature_spec)
        except ValueError as exc:
            raise InferenceError(f"Feature extraction failed: {exc}") from exc
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise InferenceError("Feature extraction produced non-finite values.")
        try:
            classes, anomaly_index, probability = _predict_detection(detector, features)
            threshold = self._decision_threshold()
            if threshold is None:
                predicted = _single_prediction(detector, features, "binary detection")
                is_anomaly = _same_label(predicted, classes[anomaly_index])
            else:
                is_anomaly = probability >= threshold
        except (AttributeError, TypeError, ValueError) as exc:
            raise InferenceError(f"Detection model failed during prediction: {exc}") from exc
        return _DetectionOutput(
            features=features,
            original_sequence=values.copy(),
            processed=processed,
            anomaly_index=anomaly_index,
            probability=probability,
            is_anomaly=is_anomaly,
            model_version=model_version,
        )

    def _load_joblib(self, name: str, relative_path: str, *, required: bool) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        path = self.config.resolve(relative_path)
        if not path.is_file():
            if required:
                raise ArtifactNotFoundError(
                    f"Required {name} artifact is missing: "
                    f"{display_path(self.config.project_root, path)}."
                )
            return None
        try:
            artifact = joblib.load(path)
        except (
            OSError,
            EOFError,
            ImportError,
            AttributeError,
            ValueError,
            KeyError,
            pickle.UnpicklingError,
        ) as exc:
            raise InferenceError(
                f"Could not load {name} artifact "
                f"'{display_path(self.config.project_root, path)}': {exc}"
            ) from exc
        if artifact is None:
            raise InferenceError(f"The saved {name} artifact is empty.")
        self._loaded[name] = artifact
        return artifact

    def _load_feature_spec(self) -> dict[str, Any]:
        if self._feature_spec is not None:
            return self._feature_spec
        path = self.config.resolve(self.config.inference.feature_config_artifact)
        if not path.is_file():
            raise ArtifactNotFoundError(
                "Required feature configuration is missing: "
                f"{display_path(self.config.project_root, path)}."
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Could not read feature configuration "
                f"'{display_path(self.config.project_root, path)}': {exc}"
            ) from exc
        self._feature_spec = _validate_feature_spec(value)
        return self._feature_spec

    def _load_model_version(self) -> str:
        if self._model_version is not None:
            return self._model_version
        metadata = self._load_model_metadata()
        version = metadata.get("model_version")
        if not isinstance(version, str) or not version.strip():
            raise ConfigurationError(
                "Model metadata must contain a non-empty 'model_version' string."
            )
        self._model_version = version.strip()
        return self._model_version

    def _load_model_metadata(self) -> dict[str, Any]:
        if self._model_metadata is not None:
            return self._model_metadata
        path = self.config.resolve(self.config.inference.metadata_artifact)
        if not path.is_file():
            raise ArtifactNotFoundError(
                "Required model metadata is missing: "
                f"{display_path(self.config.project_root, path)}."
            )
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Could not read model metadata "
                f"'{display_path(self.config.project_root, path)}': {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ConfigurationError("Model metadata must be a JSON object.")
        self._model_metadata = metadata
        return self._model_metadata

    def _decision_threshold(self) -> float | None:
        metadata = self._load_model_metadata()
        if "decision_threshold" not in metadata:
            return None
        threshold = metadata["decision_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or threshold < 0.0
            or threshold > 1.0
        ):
            raise ConfigurationError(
                "Model metadata 'decision_threshold' must be a finite number "
                "between 0 and 1."
            )
        return float(threshold)

    def _attribute_kpis(
        self, features: pd.DataFrame, anomaly_index: int
    ) -> list[str] | None:
        artifact_path = self.config.inference.attribution_artifact
        if artifact_path is None:
            return None
        explainer = self._load_joblib("attribution", artifact_path, required=False)
        if explainer is None:
            return None
        explain = getattr(explainer, "shap_values", None)
        if callable(explain):
            try:
                shap_values = explain(features)
            except (AttributeError, TypeError, ValueError) as exc:
                raise InferenceError(f"KPI attribution failed: {exc}") from exc
        elif callable(explainer):
            try:
                shap_values = explainer(features)
            except (AttributeError, TypeError, ValueError) as exc:
                raise InferenceError(f"KPI attribution failed: {exc}") from exc
        else:
            raise InferenceError("The attribution artifact does not provide shap_values().")
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            if anomaly_index >= len(shap_values):
                raise InferenceError("Attribution output does not match detector classes.")
            shap_values = shap_values[anomaly_index]
        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise InferenceError("Attribution output must contain one sample.")
            values = values[0]
            if values.shape == (features.shape[1], 2):
                values = values[:, anomaly_index]
            elif values.shape == (2, features.shape[1]):
                values = values[anomaly_index]
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1 or values.size != features.shape[1]:
            raise InferenceError(
                "Attribution output must contain one value per configured feature."
            )
        if not np.isfinite(values).all():
            raise InferenceError("Attribution output contains non-finite values.")

        spec = self._feature_spec
        if spec is None:
            raise InferenceError("Feature configuration was not initialized.")
        scores: dict[str, float] = {}
        for feature_name, score in zip(features.columns, np.abs(values), strict=True):
            definition = spec["feature_by_name"][feature_name]
            kpi = definition.get("source_kpi", definition["kpi"])
            scores[kpi] = scores.get(kpi, 0.0) + float(score)
        channel_order = spec["kpi_names"]
        ranked = sorted(
            (kpi for kpi in channel_order if scores.get(kpi, 0.0) > 0),
            key=lambda kpi: (-scores[kpi], channel_order.index(kpi)),
        )
        return ranked[: self.config.inference.attribution_top_k]

    def _attribute_models(
        self,
        output: _DetectionOutput,
        diagnosis: object,
        anomaly_type: str,
        diagnosis_confidence: float | None,
    ) -> dict[str, Any] | None:
        artifact_path = self.config.inference.attribution_artifact
        if artifact_path is None:
            return None
        explainer = self._load_joblib("attribution", artifact_path, required=False)
        if explainer is None:
            return None
        explain = getattr(explainer, "explain_prediction", None)
        if not callable(explain):
            return None
        detector = self._loaded.get("detection")
        classes = getattr(detector, "classes_", None)
        if classes is None or output.anomaly_index >= len(classes):
            raise InferenceError(
                "The saved detector classes are unavailable for attribution."
            )
        try:
            binary = explain(
                output.features,
                model=detector,
                model_kind="binary_detection",
                predicted_class=list(classes)[output.anomaly_index],
                probability=output.probability,
            )
            if diagnosis_confidence is None:
                raise InferenceError(
                    "Diagnosis model probabilities are required for attribution."
                )
            multiclass = explain(
                output.features,
                model=diagnosis,
                model_kind="multiclass_diagnosis",
                predicted_class=anomaly_type,
                probability=diagnosis_confidence,
            )
        except InferenceError:
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise InferenceError(f"KPI attribution failed: {exc}") from exc
        if not isinstance(binary, dict) or not isinstance(multiclass, dict):
            raise InferenceError("Attribution artifact returned an invalid explanation.")
        binary["top_kpis"] = binary.get("top_kpis", [])[
            : self.config.inference.attribution_top_k
        ]
        multiclass["top_kpis"] = multiclass.get("top_kpis", [])[
            : self.config.inference.attribution_top_k
        ]
        return {
            "binary_detection": binary,
            "multiclass_diagnosis": multiclass,
            "aggregation_method": binary.get("aggregation_method"),
            "interpretation": binary.get("interpretation"),
        }

    def _localize(
        self,
        output: _DetectionOutput,
        sampling_rate_hz: float,
    ) -> dict[str, Any] | None:
        artifact_path = self.config.inference.temporal_localization_artifact
        if artifact_path is None:
            return None
        localizer = self._load_joblib(
            "temporal localization",
            artifact_path,
            required=False,
        )
        if localizer is None:
            return None
        if callable(getattr(localizer, "predict_interval", None)):
            try:
                if getattr(localizer, "input_contract", None) == "raw_kpi_sequence_v1":
                    prediction = localizer.predict_interval(
                        output.original_sequence.copy(),
                        sampling_rate_hz=sampling_rate_hz,
                    )
                else:
                    prediction = localizer.predict_interval(output.processed.copy())
            except (AttributeError, TypeError, ValueError) as exc:
                raise InferenceError(f"Temporal localization failed: {exc}") from exc
        else:
            predict = getattr(localizer, "predict", None)
            if not callable(predict):
                raise InferenceError(
                    "The localization artifact does not provide predict_interval() or predict()."
                )
            try:
                prediction = predict(
                    output.processed.reshape(1, output.processed.size)
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise InferenceError(f"Temporal localization failed: {exc}") from exc
        return _parse_temporal_localization(prediction)


def _binary_response(output: _DetectionOutput) -> dict[str, Any]:
    return {
        "status": "anomaly" if output.is_anomaly else "normal",
        "anomaly_probability": output.probability,
        "model_version": output.model_version,
    }


def _validate_sample(
    sample: object,
    *,
    kpi_names: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    if kpi_names is not None and tuple(kpi_names) != KPI_NAMES:
        raise SchemaValidationError(
            ["kpi_names must match the configured 18-channel KPI order exactly."]
        )
    try:
        values = np.asarray(sample, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaValidationError(
            ["sample must contain numeric values and supported protocol strings."]
        ) from exc
    if values.shape != (TIME_STEPS, KPI_CHANNELS):
        raise SchemaValidationError(
            [
                f"sample must have shape ({TIME_STEPS}, {KPI_CHANNELS}); "
                f"received {values.shape}."
            ]
        )
    category_indexes = {KPI_NAMES.index(name) for name in CATEGORICAL_KPIS}
    has_categories = False
    for row_index in range(TIME_STEPS):
        for channel_index in range(KPI_CHANNELS):
            value = values[row_index, channel_index]
            if value is None and channel_index in category_indexes:
                has_categories = True
                continue
            if isinstance(value, str) and channel_index in category_indexes:
                if value not in ALLOWED_PROTOCOL_VALUES:
                    raise SchemaValidationError(
                        [f"sample contains an unsupported protocol category '{value}'."]
                    )
                has_categories = True
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SchemaValidationError(
                    ["sample must contain only numeric KPIs and protocol categories."]
                ) from exc
            if not np.isfinite(numeric_value):
                raise SchemaValidationError(
                    ["sample must not contain NaN or infinite values."]
                )
            values[row_index, channel_index] = numeric_value
    if not has_categories:
        return np.asarray(values, dtype=float)
    return values


def _validate_feature_spec(value: object) -> dict[str, Any]:
    try:
        return validate_feature_configuration(value)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _validate_input_feature_contract(feature_spec: Mapping[str, Any]) -> None:
    """Reject incompatible raw and one-hot KPI contracts before transforming."""
    categorical_kpis = feature_spec["categorical_kpis"]
    if not categorical_kpis:
        return
    if tuple(feature_spec["kpi_names"]) != KPI_NAMES:
        raise ConfigurationError(
            "The feature configuration KPI order does not match the canonical input order."
        )
    expected_domains = {
        kpi: list(PROTOCOL_CATEGORIES) for kpi in CATEGORICAL_KPIS
    }
    if categorical_kpis != expected_domains:
        raise ConfigurationError(
            "The feature configuration categorical domains do not match preprocessing."
        )
    if feature_spec["processed_kpi_names"] != list(ENCODED_FEATURE_NAMES):
        raise ConfigurationError(
            "The feature configuration processed KPI order does not match preprocessing."
        )


def _validate_metadata_feature_order(
    metadata: Mapping[str, Any],
    feature_order: list[str],
    model_name: str,
) -> None:
    saved_order = metadata.get("feature_order")
    if saved_order is not None and (
        not isinstance(saved_order, list) or saved_order != feature_order
    ):
        raise ConfigurationError(
            f"The {model_name} metadata feature order does not match feature_config.json."
        )
    feature_count = metadata.get("feature_count")
    if feature_count is not None and (
        isinstance(feature_count, bool)
        or not isinstance(feature_count, int)
        or feature_count != len(feature_order)
    ):
        raise ConfigurationError(
            f"The {model_name} metadata feature count does not match feature_config.json."
        )

    preprocessing = metadata.get("preprocessing_configuration")
    if not isinstance(preprocessing, Mapping):
        return
    input_contract = preprocessing.get("input")
    if isinstance(input_contract, Mapping):
        input_order = input_contract.get("kpi_order")
        if input_order is not None and input_order != list(KPI_NAMES):
            raise ConfigurationError(
                f"The {model_name} preprocessing metadata has an incompatible KPI order."
            )
        input_shape = input_contract.get("shape")
        if input_shape is not None and input_shape != [TIME_STEPS, KPI_CHANNELS]:
            raise ConfigurationError(
                f"The {model_name} preprocessing metadata has an incompatible input shape."
            )
    output_contract = preprocessing.get("output")
    if isinstance(output_contract, Mapping):
        processed_order = output_contract.get("feature_order")
        if processed_order is not None and (
            not isinstance(processed_order, list)
            or not processed_order
            or len(processed_order) != len(set(processed_order))
        ):
            raise ConfigurationError(
                f"The {model_name} preprocessing metadata has an invalid processed KPI order."
            )
        if processed_order is not None and processed_order != list(ENCODED_FEATURE_NAMES):
            raise ConfigurationError(
                f"The {model_name} preprocessing metadata does not match the "
                "configured processed KPI order."
            )


def _validate_model_feature_order(
    model: object,
    feature_order: list[str],
    model_name: str,
) -> None:
    """Compare fitted estimator feature metadata with the configured order."""
    pending = [model]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        names = getattr(candidate, "feature_names_in_", None)
        if names is not None:
            fitted_order = [str(name) for name in np.asarray(names).reshape(-1)]
            if fitted_order != feature_order:
                raise ConfigurationError(
                    f"The {model_name} model feature order does not match feature_config.json."
                )
            return
        feature_count = getattr(candidate, "n_features_in_", None)
        if feature_count is not None and feature_count != len(feature_order):
            raise ConfigurationError(
                f"The {model_name} model feature count does not match feature_config.json."
            )
        for attribute in ("estimator_", "estimator", "model", "classifier"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)


def _validate_sample_categories(
    values: np.ndarray,
    feature_spec: Mapping[str, Any],
) -> None:
    categorical_kpis = feature_spec["categorical_kpis"]
    if not categorical_kpis:
        if values.dtype == object:
            raise SchemaValidationError(
                ["Categorical input requires a feature configuration with category domains."]
            )
        return
    for kpi in categorical_kpis:
        channel_index = feature_spec["kpi_names"].index(kpi)
        for value in values[:, channel_index]:
            if value is not None and (
                not isinstance(value, str) or value not in ALLOWED_PROTOCOL_VALUES
            ):
                raise SchemaValidationError(
                    [
                        f"Categorical KPI '{kpi}' must contain supported protocol "
                        "strings or null."
                    ]
                )


def _predict_detection(
    model: object, features: pd.DataFrame
) -> tuple[list[object], int, float]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise InferenceError("The detection model must expose its binary classes_.")
    class_values = list(classes)
    if len(class_values) != 2:
        raise InferenceError("The detection model must have exactly two classes.")
    positive = [
        index
        for index, label in enumerate(class_values)
        if _is_anomaly_label(label)
    ]
    if len(positive) != 1:
        raise InferenceError(
            "The detection model classes_ must identify one anomaly class "
            "(for example 1 or 'anomaly')."
        )
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise InferenceError("The detection model does not provide predict_proba().")
    probabilities = predict_proba(features)
    scores = np.asarray(probabilities, dtype=float)
    if (
        scores.shape != (1, 2)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(np.sum(scores[0]), 1.0)
    ):
        raise InferenceError("The detection model returned invalid class probabilities.")
    anomaly_probability = float(scores[0, positive[0]])
    if anomaly_probability < 0 or anomaly_probability > 1:
        raise InferenceError("The detection model probability must be between 0 and 1.")
    return class_values, positive[0], anomaly_probability


def _is_anomaly_label(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 1.0
    return str(value).strip().lower() in _ANOMALY_LABELS


def _single_prediction(model: object, values: object, label: str) -> object:
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise InferenceError(f"The saved {label} artifact does not provide predict().")
    prediction = np.asarray(predict(values), dtype=object)
    if prediction.size != 1:
        raise InferenceError(f"The saved {label} artifact must return one prediction.")
    return prediction.reshape(-1)[0]


def _prediction_probabilities(
    model: object,
    features: pd.DataFrame,
) -> dict[str, float] | None:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None
    values = np.asarray(predict_proba(features), dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != 1
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.isclose(np.sum(values[0]), 1.0)
    ):
        raise InferenceError("The diagnosis model returned invalid probabilities.")
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    class_values = list(classes)
    if len(class_values) != values.shape[1]:
        raise InferenceError("Diagnosis probabilities do not match model classes_.")
    if any(not isinstance(label, str) or not label.strip() for label in class_values):
        raise InferenceError("Diagnosis model classes_ must be non-empty strings.")
    if len(set(class_values)) != len(class_values):
        raise InferenceError("Diagnosis model classes_ must be unique.")
    return {
        label: float(probability)
        for label, probability in zip(class_values, values[0], strict=True)
    }


def _same_label(left: object, right: object) -> bool:
    return left == right or str(left).strip().lower() == str(right).strip().lower()


def _parse_temporal_localization(prediction: object) -> dict[str, Any] | None:
    if prediction is None:
        return None
    details: Mapping[str, Any] = (
        prediction if isinstance(prediction, Mapping) else {}
    )
    if details:
        if "start" not in details or "end" not in details:
            raise InferenceError("Temporal localization must include start and end.")
        values = np.asarray([details["start"], details["end"]])
    else:
        values = np.asarray(prediction)
    if values.ndim > 1:
        if values.shape[0] != 1:
            raise InferenceError("Temporal localization must return one interval.")
        values = values[0]
    values = values.reshape(-1)
    if values.size == TIME_STEPS and np.isin(values, (0, 1, False, True)).all():
        indexes = np.flatnonzero(values.astype(bool))
        if indexes.size == 0:
            return None
        start, end = int(indexes[0]), int(indexes[-1])
    elif values.size == 2:
        try:
            boundaries = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InferenceError("Temporal interval boundaries must be numeric.") from exc
        if (
            not np.isfinite(boundaries).all()
            or not np.equal(boundaries, np.floor(boundaries)).all()
        ):
            raise InferenceError("Temporal interval boundaries must be finite integers.")
        start, end = (int(boundaries[0]), int(boundaries[1]))
    else:
        raise InferenceError(
            f"Temporal localization must return an interval or a {TIME_STEPS}-step mask."
        )
    if start < 0 or end < start or end >= TIME_STEPS:
        raise InferenceError("Temporal localization returned out-of-range boundaries.")
    result: dict[str, Any] = {"start": start, "end": end}
    for key in ("start_seconds", "end_seconds", "duration_seconds", "sampling_rate_hz"):
        if key in details:
            try:
                value = float(details[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise InferenceError(
                    f"Temporal localization field '{key}' must be numeric."
                ) from exc
            if not np.isfinite(value) or value < 0 or (
                key == "sampling_rate_hz" and value == 0
            ):
                raise InferenceError(
                    f"Temporal localization field '{key}' is outside its valid range."
                )
            result[key] = value
    if "localization_score" in details:
        try:
            score = float(details["localization_score"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise InferenceError("Temporal localization score must be numeric.") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise InferenceError("Temporal localization score must be between 0 and 1.")
        result["localization_score"] = score
    if "score_calibrated" in details:
        if not isinstance(details["score_calibrated"], bool):
            raise InferenceError("score_calibrated must be a boolean.")
        result["score_calibrated"] = details["score_calibrated"]
    if "method" in details:
        if not isinstance(details["method"], str) or not details["method"]:
            raise InferenceError("Temporal localization method must be a non-empty string.")
        result["method"] = details["method"]
    if "affected_kpis" in details:
        affected = details["affected_kpis"]
        if affected is not None and (
            not isinstance(affected, list)
            or any(not isinstance(value, str) or not value for value in affected)
        ):
            raise InferenceError("affected_kpis must be a list of non-empty strings or null.")
        result["affected_kpis"] = affected
    return result


def _build_explanation(
    anomaly_type: object,
    probability: float,
    important_kpis: list[str] | None,
    temporal_localization: dict[str, int] | None,
    model_version: str,
    has_attributions: bool = False,
) -> str:
    text = (
        f"Detected {anomaly_type} with anomaly probability {probability:.3f} "
        f"using model version {model_version}."
    )
    if important_kpis:
        text += " Most influential KPIs: " + ", ".join(important_kpis) + "."
    if temporal_localization is not None:
        text += (
            " Localized to time steps "
            f"{temporal_localization['start']}-{temporal_localization['end']}."
        )
    if has_attributions:
        text += " KPI importance reflects model associations, not physical causes."
    return text


def artifact_presence(config: AppConfig) -> dict[str, bool]:
    """Report configured inference artifacts that are present as files."""
    paths: dict[str, str | None] = {
        "detection": config.inference.detection_artifact,
        "preprocessing": config.inference.preprocessing_artifact,
        "feature_config": config.inference.feature_config_artifact,
        "metadata": config.inference.metadata_artifact,
        "diagnosis": config.diagnosis.artifact,
        "localization": config.localization.artifact,
        "temporal_localization": config.inference.temporal_localization_artifact,
        "attribution": config.inference.attribution_artifact,
    }
    return {
        name: config.resolve(path).is_file() if path is not None else False
        for name, path in paths.items()
    }


def run_inference(config: AppConfig, sample: object) -> dict[str, Any]:
    """Convenience entry point; application servers should reuse the pipeline."""
    return InferencePipeline(config).predict(sample)
