from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from backend.main import create_app
from backend.schemas import SampleRequest
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.data.telecomts import CATEGORICAL_KPIS, KPI_NAMES
from cl5g.data.telecomts_preprocessing import (
    ENCODED_FEATURE_NAMES,
    PROTOCOL_CATEGORIES,
    TelecomTSPreprocessor,
)
from frontend.dashboard_data import parse_kpi_sequence


class IdentityPreprocessor:
    def transform(self, sample: pd.DataFrame) -> pd.DataFrame:
        assert list(sample.columns) == list(KPI_NAMES)
        return pd.DataFrame(
            TelecomTSPreprocessor().transform(sample),
            columns=ENCODED_FEATURE_NAMES,
        )

    def fit(self, sample: pd.DataFrame) -> None:
        raise AssertionError("Requests must not fit preprocessing.")


class BinaryDetector:
    classes_ = np.array([0, 1])

    def __init__(self, anomalous: bool = True) -> None:
        self.anomalous = anomalous

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["latency_max", "throughput_mean"]
        return np.array([int(self.anomalous)])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["latency_max", "throughput_mean"]
        return np.array([[0.08, 0.92]] if self.anomalous else [[0.92, 0.08]])

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        raise AssertionError("Requests must not retrain the detector.")


class DiagnosisModel:
    def __init__(self, category: str = "High Network Congestion (Gradual Buildup)") -> None:
        self.category = category
        self.classes_ = np.array(["normal", category])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([self.category])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.8]])

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        raise AssertionError("Requests must not retrain diagnosis.")


class AttributionExplainer:
    def shap_values(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.1, 0.9]])


class TemporalLocalizer:
    def predict(self, sample: np.ndarray) -> np.ndarray:
        return np.array([[12, 72]])


def _create_test_app(
    tmp_path: Path,
    *,
    include_diagnosis: bool = True,
    diagnosis_category: str = "High Network Congestion (Gradual Buildup)",
    include_temporal: bool = True,
    anomalous: bool = True,
) -> FastAPI:
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
            "artifacts/temporal_localization.joblib" if include_temporal else None
        ),
        attribution_artifact="artifacts/attribution.joblib",
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
    joblib.dump(AttributionExplainer(), artifact_dir / "attribution.joblib")
    if include_temporal:
        joblib.dump(TemporalLocalizer(), artifact_dir / "temporal_localization.joblib")
    if include_diagnosis:
        joblib.dump(
            DiagnosisModel(diagnosis_category),
            artifact_dir / "diagnosis.joblib",
        )
    (artifact_dir / "feature_config.json").write_text(
        json.dumps(
            {
                "kpi_names": list(KPI_NAMES),
                "processed_kpi_names": list(ENCODED_FEATURE_NAMES),
                "numeric_kpis": [
                    name for name in KPI_NAMES if name not in CATEGORICAL_KPIS
                ],
                "categorical_kpis": {
                    name: list(PROTOCOL_CATEGORIES) for name in CATEGORICAL_KPIS
                },
                "feature_order": ["latency_max", "throughput_mean"],
                "features": [
                    {"name": "throughput_mean", "kpi": "RSRP", "statistic": "mean"},
                    {"name": "latency_max", "kpi": "DL_BLER", "statistic": "max"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": "api-test-1",
                "model_versions": {
                    "detection": "binary-1",
                    "diagnosis": "diagnosis-1",
                },
            }
        ),
        encoding="utf-8",
    )
    return create_app(config)


def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    json_body: object | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(
                method,
                path,
                json=json_body,
                headers=headers,
            )

    return asyncio.run(send())


def _sample() -> list[list[object]]:
    sample = np.zeros((128, 18), dtype=object)
    sample[:, KPI_NAMES.index("UL_Protocol")] = "None"
    sample[:, KPI_NAMES.index("DL_Protocol")] = "None"
    return sample.tolist()


def test_health_reports_artifact_presence_and_loaded_state(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path)

    response = _request(app, "GET", "/health")
    prediction = _request(app, "POST", "/predict", json_body={"sample": _sample()})
    loaded_response = _request(app, "GET", "/health")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["liveness_status"] == "alive"
    assert result["prediction_ready"] is True
    assert result["readiness_status"] == "ready"
    assert result["model_loading_status"] == "loaded"
    assert result["loaded_components"]["detection"] is True
    assert result["artifacts_present"]["detection"] is True
    assert result["artifacts_present"]["diagnosis"] is True
    assert prediction.status_code == 200
    assert loaded_response.json()["model_loading_status"] == "loaded"
    assert loaded_response.json()["loaded_components"]["detection"] is True


def test_health_returns_live_but_not_ready_when_required_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    app = _create_test_app(tmp_path)
    (tmp_path / "artifacts" / "detection.joblib").unlink()

    response = _request(app, "GET", "/health")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["liveness_status"] == "alive"
    assert result["prediction_ready"] is False
    assert result["readiness_status"] == "missing_artifacts"
    assert result["model_loading_status"] == "unavailable"
    assert "detection" in result["readiness_error"]


def test_health_reports_invalid_artifact_without_failing_liveness(
    tmp_path: Path,
) -> None:
    app = _create_test_app(tmp_path)
    (tmp_path / "artifacts" / "detection.joblib").write_bytes(b"corrupt")

    response = _request(app, "GET", "/health")

    assert response.status_code == 200
    result = response.json()
    assert result["liveness_status"] == "alive"
    assert result["prediction_ready"] is False
    assert result["readiness_status"] == "invalid_artifacts"
    assert "Could not load detection artifact" in result["readiness_error"]


def test_inference_authentication_is_enforced_when_api_key_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CL5G_API_KEY", "test-api-key-for-inference")
    app = _create_test_app(tmp_path)

    health = _request(app, "GET", "/health")
    unauthorized = _request(app, "POST", "/predict", json_body={"sample": _sample()})
    unauthorized_analysis = _request(
        app,
        "POST",
        "/analyze",
        json_body={"sample": _sample()},
    )
    authorized = _request(
        app,
        "POST",
        "/predict",
        json_body={"sample": _sample()},
        headers={"X-API-Key": "test-api-key-for-inference"},
    )
    unauthorized_model_info = _request(app, "GET", "/model-info")

    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert unauthorized_analysis.status_code == 401
    assert authorized.status_code == 200
    assert unauthorized_model_info.status_code == 401


def test_production_requires_strong_api_key_and_explicit_cors_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.main import _validate_deployment_security
    from cl5g.common.exceptions import ConfigurationError

    monkeypatch.setenv("CL5G_ENV", "production")
    monkeypatch.delenv("CL5G_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="CL5G_API_KEY"):
        create_app(load_config())

    monkeypatch.setenv("CL5G_API_KEY", "a" * 40)
    with pytest.raises(ConfigurationError, match="CL5G_CORS_ORIGINS"):
        create_app(load_config())

    monkeypatch.setenv("CL5G_CORS_ORIGINS", "http://dashboard.example")
    with pytest.raises(ConfigurationError, match="HTTPS origins"):
        create_app(load_config())

    monkeypatch.setenv("CL5G_CORS_ORIGINS", "https://dashboard.example")
    _validate_deployment_security()
    app = create_app(load_config())

    assert app.title == "Cross-Layer 5G Anomaly API"


def test_backend_main_uses_hosting_bind_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.main as backend_main

    captured: dict[str, object] = {}
    monkeypatch.setenv("CL5G_API_HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8765")
    monkeypatch.setattr(
        "uvicorn.run",
        lambda application, **options: captured.update(
            application=application,
            **options,
        ),
    )

    backend_main.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8765


def test_predict_returns_binary_result_without_diagnosis_artifact(
    tmp_path: Path,
) -> None:
    app = _create_test_app(tmp_path, include_diagnosis=False)

    response = _request(app, "POST", "/predict", json_body={"sample": _sample()})

    assert response.status_code == 200
    assert response.json() == {
        "status": "anomaly",
        "anomaly_probability": 0.92,
        "model_version": "api-test-1",
    }


def test_normal_prediction_does_not_require_diagnosis_artifact(tmp_path: Path) -> None:
    app = _create_test_app(
        tmp_path,
        include_diagnosis=False,
        anomalous=False,
    )

    response = _request(app, "POST", "/predict", json_body={"sample": _sample()})

    assert response.status_code == 200
    assert response.json() == {
        "status": "normal",
        "anomaly_probability": 0.08,
        "model_version": "api-test-1",
    }


def test_analyze_returns_complete_structured_output(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path)

    response = _request(app, "POST", "/analyze", json_body={"sample": _sample()})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "anomaly"
    assert result["anomaly_probability"] == 0.92
    assert result["predicted_anomaly_type"] == "High Network Congestion (Gradual Buildup)"
    assert result["diagnosis_confidence"] == 0.8
    assert result["diagnosis_probabilities"] == {
        "normal": 0.2,
        "High Network Congestion (Gradual Buildup)": 0.8,
    }
    assert result["model_versions"] == {
        "detection": "binary-1",
        "diagnosis": "diagnosis-1",
    }
    assert result["important_kpis"] == ["RSRP", "DL_BLER"]
    assert result["temporal_localization"] == {"start": 12, "end": 72}
    assert result["model_version"] == "api-test-1"
    assert result["explanation"]


def test_uploaded_csv_to_complete_analysis_end_to_end(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path)
    sample_frame = pd.DataFrame(
        np.arange(128 * 18, dtype=float).reshape(128, 18),
        columns=KPI_NAMES,
    )
    sample_frame["UL_Protocol"] = "TCP"
    sample_frame["DL_Protocol"] = "UDP"
    sample, channel_names = parse_kpi_sequence(
        "network_sample.csv",
        sample_frame.to_csv(index=False).encode("utf-8"),
    )

    response = _request(
        app,
        "POST",
        "/analyze",
        json_body={"sample": sample.tolist(), "kpi_names": channel_names},
    )

    assert channel_names == list(sample_frame.columns)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "anomaly"
    assert result["predicted_anomaly_type"] == "High Network Congestion (Gradual Buildup)"
    assert result["important_kpis"] == ["RSRP", "DL_BLER"]
    assert result["temporal_localization"] == {"start": 12, "end": 72}


def test_model_info_returns_versions_categories_and_feature_order(
    tmp_path: Path,
) -> None:
    app = _create_test_app(tmp_path)

    response = _request(app, "GET", "/model-info")

    assert response.status_code == 200
    result = response.json()
    assert result["model_versions"] == {
        "detection": "binary-1",
        "diagnosis": "diagnosis-1",
    }
    assert result["supported_anomaly_categories"] == [
        "Jamming",
        "Co-Channel Interference (Mild)",
        "Buffer Overflow (Gradual Buildup)",
        "Co-Channel Interference (Severe)",
        "Antenna Failure",
        "Faulty RF Filters (Temporal)",
        "High Network Congestion (Gradual Buildup)",
        "Doppler Shift (Severe)",
        "Faulty Handover Algorithm (Too Frequent)",
        "Resource Allocation Bugs",
        "High Network Congestion (Sudden Spike)",
    ]
    assert result["feature_configuration"]["feature_order"] == [
        "latency_max",
        "throughput_mean",
    ]
    assert result["decision_threshold"] is None


def test_model_info_includes_saved_threshold_and_dataset_revision(
    tmp_path: Path,
) -> None:
    app = _create_test_app(tmp_path)
    metadata_path = tmp_path / "artifacts" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "decision_threshold": 0.43,
            "dataset_revision": "pinned-test-revision",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    response = _request(app, "GET", "/model-info")

    assert response.status_code == 200
    assert response.json()["decision_threshold"] == 0.43
    assert response.json()["dataset_revision"] == "pinned-test-revision"


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ([[0.0] * 18] * 127, "128 time steps"),
        ([[0.0] * 17] + [[0.0] * 18] * 127, "exactly 18 KPI values"),
    ],
)
def test_predict_rejects_invalid_sequence_shape(
    tmp_path: Path,
    sample: list[list[float]],
    message: str,
) -> None:
    app = _create_test_app(tmp_path)

    response = _request(app, "POST", "/predict", json_body={"sample": sample})

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize(
    "bad_value",
    [None, "invalid-category"],
)
def test_predict_rejects_missing_or_categorical_kpi_values(
    tmp_path: Path,
    bad_value: object,
) -> None:
    app = _create_test_app(tmp_path)
    sample = _sample()
    sample[17][4] = bad_value  # type: ignore[list-item]

    response = _request(app, "POST", "/predict", json_body={"sample": sample})

    assert response.status_code == 422


def test_request_schema_accepts_observed_protocol_categories() -> None:
    sample = _sample()
    sample[0][14] = "TCP"  # type: ignore[list-item]
    sample[0][16] = "None"  # type: ignore[list-item]

    request = SampleRequest(sample=sample)

    assert request.sample[0][14] == "TCP"
    assert request.sample[0][16] == "None"


def test_request_schema_accepts_null_only_for_categorical_channels() -> None:
    sample = _sample()
    sample[0][14] = None  # type: ignore[list-item]
    sample[0][16] = None  # type: ignore[list-item]

    request = SampleRequest(sample=sample)

    assert request.sample[0][14] is None
    assert request.sample[0][16] is None


def test_request_schema_rejects_null_numeric_values() -> None:
    sample = _sample()
    sample[0][0] = None  # type: ignore[list-item]

    with pytest.raises(ValidationError, match="missing or non-finite numeric"):
        SampleRequest(sample=sample)


def test_request_schema_rejects_noncanonical_named_kpi_order() -> None:
    with pytest.raises(ValidationError, match="KPI order"):
        SampleRequest(
            sample=_sample(),
            kpi_names=list(reversed(KPI_NAMES)),
        )


@pytest.mark.parametrize(
    ("channel_index", "value"),
    [(0, "TCP"), (14, "LTE")],
)
def test_request_schema_rejects_unknown_or_misplaced_categories(
    channel_index: int,
    value: str,
) -> None:
    sample = _sample()
    sample[0][channel_index] = value  # type: ignore[list-item]

    with pytest.raises(ValidationError):
        SampleRequest(sample=sample)


def test_analyze_rejects_unsupported_model_anomaly_category(tmp_path: Path) -> None:
    app = _create_test_app(
        tmp_path,
        diagnosis_category="unconfigured-category",
    )

    response = _request(app, "POST", "/analyze", json_body={"sample": _sample()})

    assert response.status_code == 500
    assert "unsupported anomaly category" in response.json()["detail"]


def test_analyze_does_not_invent_missing_temporal_localization(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path, include_temporal=False)

    response = _request(app, "POST", "/analyze", json_body={"sample": _sample()})

    assert response.status_code == 200
    assert response.json()["temporal_localization"] is None


def test_missing_artifact_returns_structured_service_error(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path)
    (tmp_path / "artifacts" / "detection.joblib").unlink()

    response = _request(app, "POST", "/predict", json_body={"sample": _sample()})

    assert response.status_code == 503
    assert "detection artifact is missing" in response.json()["detail"]


def test_cors_preflight_and_openapi_are_available(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path)

    preflight = _request(
        app,
        "OPTIONS",
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    docs = _request(app, "GET", "/openapi.json")

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert docs.status_code == 200
    assert {"/health", "/predict", "/analyze", "/model-info"}.issubset(
        docs.json()["paths"]
    )
