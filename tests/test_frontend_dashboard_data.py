from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import ApiClientError, analyze_sample, predict_sample
from frontend.dashboard_data import (
    evaluation_results,
    load_saved_evaluation_reports,
    parse_kpi_sequence,
)
from cl5g.data.telecomts import KPI_NAMES


def test_parse_csv_sequence_preserves_channel_order() -> None:
    source = _canonical_sequence_frame()

    values, names = parse_kpi_sequence(
        "sequence.csv",
        source.to_csv(index=False).encode("utf-8"),
    )

    assert values.shape == (128, 18)
    assert names == list(KPI_NAMES)
    np.testing.assert_array_equal(values, source.to_numpy())


@pytest.mark.parametrize(
    "columns",
    [
        tuple(reversed(KPI_NAMES)),
        (*KPI_NAMES[:-1], "unexpected_kpi"),
    ],
)
def test_parse_csv_rejects_mislabeled_or_reordered_kpis(
    columns: tuple[str, ...],
) -> None:
    source = pd.DataFrame(
        np.arange(128 * 18, dtype=float).reshape(128, 18),
        columns=columns,
    )

    with pytest.raises(ValueError, match="headers must match"):
        parse_kpi_sequence(
            "sequence.csv",
            source.to_csv(index=False).encode("utf-8"),
        )


def test_parse_csv_preserves_protocol_categories_and_nulls() -> None:
    source = pd.DataFrame(
        np.zeros((128, 18), dtype=float),
        columns=KPI_NAMES,
    )
    source["UL_Protocol"] = ["TCP"] * 127 + [""]
    source["DL_Protocol"] = ["None"] * 127 + ["UDP"]

    values, names = parse_kpi_sequence(
        "sequence.csv",
        source.to_csv(index=False).encode("utf-8"),
    )

    assert names == list(KPI_NAMES)
    assert values[0, KPI_NAMES.index("UL_Protocol")] == "TCP"
    assert values[-1, KPI_NAMES.index("UL_Protocol")] == ""
    assert values[0, KPI_NAMES.index("DL_Protocol")] == "None"


def test_parse_csv_rejects_missing_numeric_and_unknown_protocol_values() -> None:
    missing_numeric = _canonical_sequence_frame()
    missing_numeric["RSRP"] = missing_numeric["RSRP"].astype(object)
    missing_numeric.loc[0, "RSRP"] = ""
    with pytest.raises(ValueError, match="Numeric KPI 'RSRP'"):
        parse_kpi_sequence(
            "sequence.csv",
            missing_numeric.to_csv(index=False).encode("utf-8"),
        )

    invalid_protocol = _canonical_sequence_frame()
    invalid_protocol.loc[0, "UL_Protocol"] = "LTE"
    with pytest.raises(ValueError, match="Categorical KPI 'UL_Protocol'"):
        parse_kpi_sequence(
            "sequence.csv",
            invalid_protocol.to_csv(index=False).encode("utf-8"),
        )


def test_parse_json_sequence_supports_sample_object() -> None:
    source = np.zeros((128, 18), dtype=object)
    source[:, KPI_NAMES.index("UL_Protocol")] = "None"
    source[:, KPI_NAMES.index("DL_Protocol")] = "None"

    values, names = parse_kpi_sequence(
        "sequence.json",
        json.dumps({"sample": source.tolist()}).encode("utf-8"),
    )

    assert values.shape == (128, 18)
    assert names[0] == "Channel 1"
    np.testing.assert_array_equal(values, source)


def _canonical_sequence_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        np.arange(128 * 18, dtype=float).reshape(128, 18),
        columns=KPI_NAMES,
    )
    frame["UL_Protocol"] = "TCP"
    frame["DL_Protocol"] = "UDP"
    return frame


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((127, 18)),
        np.zeros((128, 17)),
        np.full((128, 18), np.nan),
    ],
)
def test_parse_json_sequence_rejects_bad_dimensions_or_values(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        parse_kpi_sequence("sequence.json", json.dumps(values.tolist()).encode("utf-8"))


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("bad.txt", b"not a sequence"),
        ("bad.json", b"{invalid json"),
        ("bad.csv", b"a,b\n1,2\n"),
        (
            "missing.csv",
            pd.DataFrame(
                [[np.nan] * 18 for _ in range(128)],
                columns=[f"kpi_{index}" for index in range(18)],
            ).to_csv(index=False).encode("utf-8"),
        ),
    ],
)
def test_invalid_uploaded_sequence_is_rejected(filename: str, content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_kpi_sequence(filename, content)


def test_evaluation_results_calculates_only_from_uploaded_labels_and_scores() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["normal", "normal", "anomaly", "anomaly"],
            "predicted": ["normal", "anomaly", "anomaly", "anomaly"],
            "anomaly_probability": [0.1, 0.7, 0.8, 0.9],
        }
    )

    result = evaluation_results(
        frame,
        actual_column="actual",
        predicted_column="predicted",
        probability_column="anomaly_probability",
    )

    assert result["rows_evaluated"] == 4
    assert result["labels"] == ["anomaly", "normal"]
    assert result["confusion_matrix"].tolist() == [[2, 0], [1, 1]]
    assert result["classification_report"]["accuracy"] == 0.75
    assert result["precision"] is not None
    assert result["recall"] is not None


def test_evaluation_results_requires_actual_and_predicted_columns() -> None:
    with pytest.raises(ValueError, match="Missing evaluation column"):
        evaluation_results(
            pd.DataFrame({"actual": ["normal"]}),
            actual_column="actual",
            predicted_column="predicted",
            probability_column=None,
        )


def test_evaluation_results_rejects_reusing_truth_as_prediction() -> None:
    with pytest.raises(ValueError, match="must be different"):
        evaluation_results(
            pd.DataFrame({"label": ["normal", "anomaly"]}),
            actual_column="label",
            predicted_column="label",
            probability_column=None,
        )


def test_saved_evaluation_reports_match_served_models_and_revision(
    tmp_path: Path,
) -> None:
    revision = "1b3a88a440f4922edfcd1e230e42c003c0e13bdf"
    report_directory = tmp_path / "data" / "reports" / "telecomts" / revision
    expected_reports = (
        (
            report_directory
            / "binary_detection"
            / "seed-42"
            / "final_test_report.json",
            "detector-v1",
        ),
        (
            report_directory
            / "multiclass_diagnosis"
            / "seed-42"
            / "final_test_report.json",
            "diagnosis-v1",
        ),
    )
    for path, version in expected_reports:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_id": "AliMaatouk/TelecomTS",
                    "dataset_revision": revision,
                    "model_version": version,
                    "metrics": {"accuracy": 0.75},
                }
            ),
            encoding="utf-8",
        )

    loaded, errors = load_saved_evaluation_reports(
        tmp_path,
        dataset_revision=revision,
        model_versions={"detection": "detector-v1", "diagnosis": "diagnosis-v1"},
    )

    assert errors == []
    assert set(loaded) == {"binary_detection", "multiclass_diagnosis"}
    assert loaded["binary_detection"]["metrics"]["accuracy"] == 0.75
    assert Path(loaded["binary_detection"]["report_path"]) == expected_reports[0][0]


def test_saved_evaluation_reports_do_not_load_stale_model_results(
    tmp_path: Path,
) -> None:
    revision = "1b3a88a440f4922edfcd1e230e42c003c0e13bdf"
    path = (
        tmp_path
        / "data"
        / "reports"
        / "telecomts"
        / revision
        / "binary_detection"
        / "seed-42"
        / "final_test_report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "AliMaatouk/TelecomTS",
                "dataset_revision": revision,
                "model_version": "old-detector",
                "metrics": {"accuracy": 0.99},
            }
        ),
        encoding="utf-8",
    )

    loaded, errors = load_saved_evaluation_reports(
        tmp_path,
        dataset_revision=revision,
        model_versions={"detection": "current-detector"},
    )

    assert loaded == {}
    assert errors == []


@pytest.mark.parametrize(
    "client_function",
    [predict_sample, analyze_sample],
)
def test_backend_connection_failure_is_reported(
    client_function: Callable[..., object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fail_request)

    with pytest.raises(ApiClientError, match="Could not reach the backend"):
        client_function("http://127.0.0.1:8000", np.zeros((128, 18)).tolist())


def test_complete_analysis_sends_sampling_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    analyze_sample(
        "http://127.0.0.1:8000",
        np.zeros((128, 18)).tolist(),
        sampling_rate_hz=20.0,
    )

    assert captured["url"] == "http://127.0.0.1:8000/analyze"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["sampling_rate_hz"] == 20.0
    assert "sample" in body


def test_api_client_sends_explicit_kpi_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cl5g.data.telecomts import KPI_NAMES

    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    predict_sample(
        "http://127.0.0.1:8000",
        np.zeros((128, 18)).tolist(),
        kpi_names=list(KPI_NAMES),
    )

    assert captured["body"]["kpi_names"] == list(KPI_NAMES)  # type: ignore[index]


def test_api_client_sends_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        captured["api_key"] = request.get_header("X-api-key") or ""
        return _Response()

    monkeypatch.setenv("CL5G_API_KEY", "api-key-test")
    monkeypatch.setenv("CL5G_API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    predict_sample(
        "https://api.example.test",
        np.zeros((128, 18)).tolist(),
    )

    assert captured["api_key"] == "api-key-test"


def test_api_client_does_not_send_key_to_user_changed_backend_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CL5G_API_KEY", "api-key-test")
    monkeypatch.setenv("CL5G_API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("unconfigured host must not receive the API key"),
    )

    with pytest.raises(ApiClientError, match="configured CL5G_API_BASE_URL"):
        predict_sample(
            "https://attacker.example",
            np.zeros((128, 18)).tolist(),
        )


def test_api_base_url_uses_deployment_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cl5g.pipeline.config import load_config
    from frontend.status import api_base_url

    monkeypatch.setenv("CL5G_API_BASE_URL", "https://api.example.test")

    assert api_base_url(load_config()) == "https://api.example.test"


def test_api_base_url_rejects_urls_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cl5g.common.exceptions import ConfigurationError
    from cl5g.pipeline.config import load_config
    from frontend.status import api_base_url

    monkeypatch.setenv("CL5G_API_BASE_URL", "https://user:password@api.example.test")

    with pytest.raises(ConfigurationError, match="CL5G_API_BASE_URL"):
        api_base_url(load_config())


def test_api_base_url_rejects_public_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cl5g.common.exceptions import ConfigurationError
    from cl5g.pipeline.config import load_config
    from frontend.status import api_base_url

    monkeypatch.setenv("CL5G_API_BASE_URL", "http://api.example.test")

    with pytest.raises(ConfigurationError, match="HTTPS"):
        api_base_url(load_config())


def test_streamlit_pages_render_without_fabricated_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontend.api_client as api_client

    monkeypatch.setattr(
        api_client,
        "fetch_health",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "liveness_status": "alive",
            "prediction_ready": True,
            "readiness_status": "ready",
            "readiness_error": None,
            "model_loading_status": "loaded",
            "loaded_components": {},
            "artifacts_present": {},
        },
    )
    monkeypatch.setattr(
        api_client,
        "fetch_model_info",
        lambda *_args, **_kwargs: {
            "model_versions": {"test": "test-version"},
            "supported_anomaly_categories": ["congestion"],
            "feature_configuration": {
                "kpi_names": [f"kpi_{index}" for index in range(18)],
                "feature_order": ["kpi_0_mean"],
                "features": [
                    {"name": "kpi_0_mean", "kpi": "kpi_0", "statistic": "mean"}
                ],
            },
        },
    )
    dashboard = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"),
        default_timeout=45,
    )
    dashboard.session_state["historical_dataset"] = pd.DataFrame(
        {
            "entity_id": ["gnb-a", "gnb-b", "gnb-a"],
            "anomaly_type": ["normal", "congestion", "interference"],
            **{
                f"kpi_{index}": [float(index), float(index + 1), float(index + 2)]
                for index in range(18)
            },
        }
    )
    dashboard.session_state["historical_dataset_source"] = "Test-only fixture"
    dashboard.run(timeout=45)

    assert not dashboard.exception
    assert len(dashboard.get("plotly_chart")) == 1
    assert any(
        metric.label == "Validated model performance"
        and metric.value == "Not available"
        for metric in dashboard.metric
    )
    pages = (
        "Dataset Exploration",
        "KPI Monitoring",
        "Anomaly Detection",
        "Anomaly Diagnosis",
        "KPI Attribution",
        "Temporal Localization",
        "Model Evaluation",
        "About the Project",
        "Network Overview",
    )
    for page in pages:
        dashboard.radio[0].set_value(page).run(timeout=45)
        assert not dashboard.exception, page
        if page == "Dataset Exploration":
            assert len(dashboard.get("plotly_chart")) >= 3


def test_temporal_localization_page_with_sequence_and_analysis_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontend.api_client as api_client

    monkeypatch.setattr(
        api_client,
        "fetch_health",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "liveness_status": "alive",
            "prediction_ready": True,
            "readiness_status": "ready",
            "readiness_error": None,
            "model_loading_status": "loaded",
            "loaded_components": {},
            "artifacts_present": {},
        },
    )
    dashboard = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"),
        default_timeout=45,
    )
    dashboard.session_state["kpi_sequence"] = np.arange(
        128 * 18,
        dtype=float,
    ).reshape(128, 18)
    dashboard.session_state["sequence_channel_names"] = list(KPI_NAMES)
    dashboard.session_state["latest_analysis"] = {
        "status": "anomaly",
        "temporal_localization": {
            "start": 10,
            "end": 29,
            "start_seconds": 1.0,
            "end_seconds": 2.9,
            "duration_seconds": 2.0,
            "sampling_rate_hz": 10.0,
            "localization_score": 0.8,
            "score_calibrated": False,
            "method": "experimental_test",
            "affected_kpis": None,
        },
    }

    dashboard.run(timeout=45)
    dashboard.radio[0].set_value("Temporal Localization").run(timeout=45)

    assert not dashboard.exception
    assert len(dashboard.selectbox) == 1
    assert dashboard.selectbox[0].options[0] == "RSRP"
    assert len(dashboard.get("plotly_chart")) == 1
    assert any("Model interval: steps 10–29 inclusive" in item.value for item in dashboard.success)


def test_kpi_monitoring_renders_individual_comparison_and_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frontend.api_client as api_client

    monkeypatch.setattr(
        api_client,
        "fetch_health",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "liveness_status": "alive",
            "prediction_ready": True,
            "readiness_status": "ready",
            "readiness_error": None,
            "model_loading_status": "loaded",
            "loaded_components": {},
            "artifacts_present": {},
        },
    )
    monkeypatch.setattr(
        api_client,
        "fetch_model_info",
        lambda *_args, **_kwargs: {
            "model_versions": {"test": "test-version"},
            "supported_anomaly_categories": [],
            "feature_configuration": {"kpi_names": list(KPI_NAMES)},
        },
    )
    dashboard = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"),
        default_timeout=45,
    )
    values = np.empty((128, 18), dtype=object)
    values[:] = np.arange(128, dtype=float)[:, None]
    for channel in (KPI_NAMES.index("UL_Protocol"), KPI_NAMES.index("DL_Protocol")):
        values[:, channel] = np.where(np.arange(128) % 2, "UDP", "TCP")
    dashboard.session_state["kpi_sequence"] = values
    dashboard.session_state["sequence_channel_names"] = list(KPI_NAMES)
    dashboard.session_state["latest_analysis"] = {
        "temporal_localization": {"start": 10, "end": 29}
    }

    dashboard.run(timeout=45)
    dashboard.radio[0].set_value("KPI Monitoring").run(timeout=45)

    assert not dashboard.exception
    assert len(dashboard.get("plotly_chart")) >= 3
    assert any(metric.label == "Mean" for metric in dashboard.metric)
    assert any(
        "not live network telemetry" in item.value.lower()
        for item in dashboard.markdown
    )
