from __future__ import annotations

import json
import asyncio
from dataclasses import replace
from pathlib import Path

import joblib
import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from cl5g.common.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    SchemaValidationError,
)
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.pipeline.inference import InferenceError, InferencePipeline
from cl5g.models.attribution import KPIAttributionExplainer
from cl5g.models.diagnosis import StringLabelClassifier
from backend.app import create_app


class IdentityPreprocessor:
    def transform(self, sample: pd.DataFrame) -> pd.DataFrame:
        assert list(sample.columns) == [f"kpi_{index}" for index in range(18)]
        return sample

    def fit(self, sample: pd.DataFrame) -> None:
        raise AssertionError("Inference must not fit preprocessing.")


class BinaryDetector:
    classes_ = np.array([0, 1])

    def __init__(self, anomalous: bool) -> None:
        self.anomalous = anomalous

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["latency_last", "throughput_mean"]
        return np.array([int(self.anomalous)])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["latency_last", "throughput_mean"]
        return np.array([[0.06, 0.94]] if self.anomalous else [[0.94, 0.06]])

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        raise AssertionError("Inference must not fit the detector.")


class DiagnosisModel:
    def __init__(self, category: str = "High Network Congestion (Gradual Buildup)") -> None:
        self.category = category
        self.classes_ = np.array(["normal", category])

    def predict(self, features: object) -> np.ndarray:
        return np.array([self.category])

    def predict_proba(self, features: object) -> np.ndarray:
        return np.array([[0.18, 0.82]])

    def fit(self, features: object, labels: np.ndarray) -> None:
        raise AssertionError("Inference must not fit diagnosis.")


class AttributionExplainer:
    def shap_values(self, features: object) -> np.ndarray:
        return np.array([[[0.0, 0.1], [0.0, 0.9]]])


class TemporalLocalizer:
    def predict(self, sample: np.ndarray) -> np.ndarray:
        return np.array([[40, 85]])


class RawSequenceTemporalLocalizer:
    input_contract = "raw_kpi_sequence_v1"

    def predict_interval(
        self,
        sample: np.ndarray,
        *,
        sampling_rate_hz: float,
    ) -> dict[str, object]:
        assert sample.shape == (128, 18)
        assert sample[0, 0] == 0.0
        assert sampling_rate_hz == 20.0
        return {
            "start": 10,
            "end": 29,
            "start_seconds": 0.5,
            "end_seconds": 1.45,
            "duration_seconds": 1.0,
            "sampling_rate_hz": sampling_rate_hz,
            "localization_score": 0.8,
            "score_calibrated": False,
            "method": "test_supervised_per_timestep",
            "affected_kpis": None,
        }


class OffsetPreprocessor:
    def transform(self, sample: pd.DataFrame) -> pd.DataFrame:
        return sample + 2.0

    def fit(self, sample: pd.DataFrame) -> None:
        raise AssertionError("Inference must not fit preprocessing.")


class FeatureVerifier:
    classes_ = np.array([0, 1])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["latency_last", "throughput_mean"]
        assert features.loc[0, "latency_last"] == 2.0
        assert features.loc[0, "throughput_mean"] == 2.0
        return np.array([0])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.97, 0.03]])

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        raise AssertionError("Inference must not fit the detector.")


def _configured_pipeline(
    tmp_path: Path,
    *,
    anomalous: bool,
    diagnosis_category: str = "High Network Congestion (Gradual Buildup)",
    include_temporal_artifact: bool = True,
    include_attribution_artifact: bool = True,
) -> InferencePipeline:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    base = load_config()
    inference = replace(
        base.inference,
        detection_artifact="artifacts/detection.joblib",
        preprocessing_artifact="artifacts/preprocessing.joblib",
        feature_config_artifact="artifacts/feature_config.json",
        metadata_artifact="artifacts/model_metadata.json",
        temporal_localization_artifact=(
            "artifacts/temporal_localization.joblib"
            if include_temporal_artifact
            else None
        ),
        attribution_artifact=(
            "artifacts/attribution.joblib"
            if anomalous and include_attribution_artifact
            else None
        ),
    )
    diagnosis = replace(base.diagnosis, artifact="artifacts/diagnosis.joblib")
    config: AppConfig = replace(
        base,
        project_root=tmp_path,
        inference=inference,
        diagnosis=diagnosis,
    )

    joblib.dump(BinaryDetector(anomalous), artifact_dir / "detection.joblib")
    joblib.dump(IdentityPreprocessor(), artifact_dir / "preprocessing.joblib")
    (artifact_dir / "feature_config.json").write_text(
        json.dumps(
            {
                "kpi_names": [f"kpi_{index}" for index in range(18)],
                "feature_order": ["latency_last", "throughput_mean"],
                "features": [
                    {"name": "throughput_mean", "kpi": "kpi_0", "statistic": "mean"},
                    {"name": "latency_last", "kpi": "kpi_1", "statistic": "last"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps({"model_version": "test-2026.1"}),
        encoding="utf-8",
    )
    if anomalous:
        joblib.dump(
            DiagnosisModel(diagnosis_category),
            artifact_dir / "diagnosis.joblib",
        )
        if include_attribution_artifact:
            joblib.dump(AttributionExplainer(), artifact_dir / "attribution.joblib")
        if include_temporal_artifact:
            joblib.dump(
                TemporalLocalizer(),
                artifact_dir / "temporal_localization.joblib",
            )
    return InferencePipeline(config)


def _post_sample(
    pipeline: InferencePipeline,
    sample: np.ndarray,
    *,
    sampling_rate_hz: float | None = None,
) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(pipeline.config))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            body: dict[str, object] = {"sample": sample.tolist()}
            if sampling_rate_hz is not None:
                body["sampling_rate_hz"] = sampling_rate_hz
            return await client.post("/analyze", json=body)

    return asyncio.run(request())


def test_normal_sample_returns_binary_result_without_loading_diagnosis(
    tmp_path: Path,
) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)

    response = _post_sample(pipeline, np.zeros((128, 18)))

    assert response.status_code == 200
    result = response.json()
    assert result == {
        "status": "normal",
        "anomaly_probability": 0.06,
        "model_versions": {"detection": "test-2026.1"},
        "predicted_anomaly_type": None,
        "diagnosis_confidence": None,
        "diagnosis_probabilities": None,
        "important_kpis": None,
        "attributions": None,
        "temporal_localization": None,
        "model_version": "test-2026.1",
        "explanation": "No anomaly detected by model version test-2026.1.",
    }


def test_inference_uses_saved_threshold_when_metadata_contains_one(
    tmp_path: Path,
) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=True)
    metadata_path = tmp_path / "artifacts" / "model_metadata.json"
    metadata_path.write_text(
        json.dumps({"model_version": "threshold-test", "decision_threshold": 0.95}),
        encoding="utf-8",
    )

    result = pipeline.predict_binary(np.zeros((128, 18)))

    assert result["status"] == "normal"
    assert result["anomaly_probability"] == 0.94


def test_inference_rejects_invalid_saved_threshold(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)
    metadata_path = tmp_path / "artifacts" / "model_metadata.json"
    metadata_path.write_text(
        json.dumps({"model_version": "invalid-threshold", "decision_threshold": 1.5}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="decision_threshold"):
        pipeline.predict_binary(np.zeros((128, 18)))


def test_anomalous_sample_runs_diagnosis_attribution_and_localization(
    tmp_path: Path,
) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=True)

    response = _post_sample(
        pipeline,
        np.arange(128 * 18, dtype=float).reshape(128, 18),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "anomaly"
    assert result["anomaly_probability"] == 0.94
    assert result["predicted_anomaly_type"] == "High Network Congestion (Gradual Buildup)"
    assert result["diagnosis_confidence"] == 0.82
    assert result["diagnosis_probabilities"] == {
        "normal": 0.18,
        "High Network Congestion (Gradual Buildup)": 0.82,
    }
    assert result["important_kpis"] == ["kpi_0", "kpi_1"]
    assert result["temporal_localization"] == {"start": 40, "end": 85}
    assert result["model_version"] == "test-2026.1"
    assert "High Network Congestion (Gradual Buildup)" in result["explanation"]


def test_named_input_must_use_canonical_kpi_order(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)

    with pytest.raises(SchemaValidationError, match="KPI order"):
        pipeline.predict_binary(
            np.zeros((128, 18)),
            kpi_names=[f"channel_{index}" for index in range(18)],
        )


def test_model_feature_order_mismatch_is_rejected(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)
    training = pd.DataFrame(
        np.zeros((2, 2)),
        columns=["wrong_first", "wrong_second"],
    )
    model = RandomForestClassifier(n_estimators=2, random_state=1).fit(
        training,
        np.array([0, 1]),
    )
    joblib.dump(model, tmp_path / "artifacts" / "detection.joblib")

    with pytest.raises(ConfigurationError, match="feature order"):
        pipeline.predict_binary(np.zeros((128, 18)))


def test_metadata_feature_order_mismatch_is_rejected(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)
    metadata_path = tmp_path / "artifacts" / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "mismatch",
                "feature_order": ["throughput_mean", "latency_last"],
                "feature_count": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="metadata feature order"):
        pipeline.predict_binary(np.zeros((128, 18)))


def test_corrupt_model_artifact_reports_load_failure(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)
    (tmp_path / "artifacts" / "detection.joblib").write_bytes(b"not a joblib artifact")

    with pytest.raises(InferenceError, match="Could not load detection artifact"):
        pipeline.predict_binary(np.zeros((128, 18)))


def test_sample_dimensions_are_validated(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)

    with pytest.raises(SchemaValidationError, match="shape"):
        pipeline.predict(np.zeros((127, 18)))


def test_missing_required_artifact_has_a_meaningful_error(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)
    (tmp_path / "artifacts" / "detection.joblib").unlink()

    with pytest.raises(ArtifactNotFoundError, match="detection artifact is missing"):
        pipeline.predict(np.zeros((128, 18)))


def test_unsupported_model_category_is_rejected(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(
        tmp_path,
        anomalous=True,
        diagnosis_category="not-a-supported-category",
    )

    with pytest.raises(InferenceError, match="unsupported anomaly category"):
        pipeline.predict(np.zeros((128, 18)))


def test_missing_temporal_artifact_returns_no_localization(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(
        tmp_path,
        anomalous=True,
        include_temporal_artifact=False,
    )

    result = pipeline.predict(np.zeros((128, 18)))

    assert result["temporal_localization"] is None


def test_localization_receives_original_sequence_and_maps_api_time(
    tmp_path: Path,
) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=True)
    joblib.dump(
        RawSequenceTemporalLocalizer(),
        tmp_path / "artifacts" / "temporal_localization.joblib",
    )

    response = _post_sample(
        pipeline,
        np.arange(128 * 18, dtype=float).reshape(128, 18),
        sampling_rate_hz=20.0,
    )

    assert response.status_code == 200
    interval = response.json()["temporal_localization"]
    assert interval["start"] == 10
    assert interval["end"] == 29
    assert interval["start_seconds"] == 0.5
    assert interval["end_seconds"] == 1.45
    assert interval["duration_seconds"] == 1.0
    assert interval["sampling_rate_hz"] == 20.0
    assert interval["localization_score"] == 0.8
    assert interval["method"] == "test_supervised_per_timestep"
    assert interval["affected_kpis"] is None


def test_missing_attribution_artifact_returns_no_kpi_ranking(tmp_path: Path) -> None:
    pipeline = _configured_pipeline(
        tmp_path,
        anomalous=True,
        include_attribution_artifact=False,
    )

    result = pipeline.predict(np.zeros((128, 18)))

    assert result["important_kpis"] is None
    assert result["attributions"] is None


def test_tree_shap_attributions_are_returned_as_structured_api_data(
    tmp_path: Path,
) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=True)
    training = pd.DataFrame(
        {
            "latency_last": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            "throughput_mean": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        }
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    joblib.dump(
        RandomForestClassifier(n_estimators=20, random_state=2).fit(
            training,
            labels,
        ),
        tmp_path / "artifacts" / "detection.joblib",
    )
    joblib.dump(
        StringLabelClassifier(
            RandomForestClassifier(n_estimators=20, random_state=3)
        ).fit(
            training,
            np.array(
                [
                    "Antenna Failure",
                    "Antenna Failure",
                    "Antenna Failure",
                    "Jamming",
                    "Jamming",
                    "Jamming",
                ],
                dtype=object,
            ),
        ),
        tmp_path / "artifacts" / "diagnosis.joblib",
    )
    feature_configuration = json.loads(
        (tmp_path / "artifacts" / "feature_config.json").read_text(
            encoding="utf-8"
        )
    )
    joblib.dump(
        KPIAttributionExplainer(feature_configuration),
        tmp_path / "artifacts" / "attribution.joblib",
    )

    response = _post_sample(pipeline, np.full((128, 18), 2.0))

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "anomaly"
    assert result["attributions"]["binary_detection"]["method"] == "TreeSHAP"
    assert result["attributions"]["multiclass_diagnosis"]["method"] == "TreeSHAP"
    assert result["attributions"]["multiclass_diagnosis"]["predicted_class"] == "Jamming"
    assert result["important_kpis"]
    assert "not establish physical causality" in result["attributions"]["interpretation"]


def test_inference_applies_saved_preprocessor_and_preserves_feature_order(
    tmp_path: Path,
) -> None:
    pipeline = _configured_pipeline(tmp_path, anomalous=False)

    joblib.dump(
        OffsetPreprocessor(),
        tmp_path / "artifacts" / "preprocessing.joblib",
    )
    joblib.dump(
        FeatureVerifier(),
        tmp_path / "artifacts" / "detection.joblib",
    )

    result = pipeline.predict_binary(np.zeros((128, 18)))

    assert result["status"] == "normal"
    assert result["anomaly_probability"] == 0.03
