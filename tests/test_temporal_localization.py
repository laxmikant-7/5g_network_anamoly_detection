from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from cl5g.common.exceptions import SchemaValidationError
from cl5g.data.telecomts import KPI_NAMES
from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor
from cl5g.evaluation.temporal_localization import (
    _interval_metrics,
    _load_partition,
    _validate_group_splits,
    _write_metrics_plot,
    _write_sample_plot,
    _write_threshold_plot,
)
from cl5g.models.temporal_localization import (
    TemporalIntervalLocalizer,
    interval_from_scores,
    temporal_context_features,
)


class _WindowGate:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = features[:, 52] > 0.5
        return np.column_stack(
            (np.where(positive, 0.1, 0.9), np.where(positive, 0.9, 0.1))
        )


def _raw_sequence() -> dict[str, list[float] | list[str]]:
    sequence: dict[str, list[float] | list[str]] = {
        name: [0.0] * 128 for name in KPI_NAMES
    }
    sequence["UL_Protocol"] = ["TCP"] * 128
    sequence["DL_Protocol"] = ["UDP"] * 128
    sequence["RSRP"][40:86] = [1.0] * 46
    return sequence


def test_context_features_preserve_time_alignment_and_edge_padding() -> None:
    sequence = np.zeros((128, 2), dtype=float)
    sequence[0, 0] = 3.0
    sequence[-1, 1] = 5.0

    features = temporal_context_features(sequence, context_radius=1)

    assert features.shape == (128, 6)
    assert features[0].tolist() == [3.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    assert features[-1].tolist() == [0.0, 0.0, 0.0, 5.0, 0.0, 5.0]


def test_interval_conversion_uses_inclusive_first_and_last_positive_steps() -> None:
    scores = np.zeros(128)
    scores[8:13] = 0.6

    assert interval_from_scores(scores, 0.5) == (8, 12)
    assert interval_from_scores(scores, 0.7) is None


def test_localizer_accepts_original_sequence_and_maps_time() -> None:
    localizer = TemporalIntervalLocalizer(_WindowGate(), threshold=0.5)

    result = localizer.predict_interval(_raw_sequence(), sampling_rate_hz=20)

    assert result is not None
    assert (result["start"], result["end"]) == (40, 85)
    assert result["start_seconds"] == 2.0
    assert result["end_seconds"] == 4.25
    assert result["duration_seconds"] == 2.3
    assert result["localization_score"] == pytest.approx(0.9)
    assert result["score_calibrated"] is False
    assert result["affected_kpis"] is None


def test_interval_metrics_measure_overlap_event_and_boundary_errors() -> None:
    metrics = _interval_metrics([(2, 4)], [(1, 4)])

    assert metrics["mean_temporal_iou"] == 0.75
    assert metrics["event_precision_iou_0_5"] == 1.0
    assert metrics["event_recall_iou_0_5"] == 1.0
    assert metrics["boundary_start_mae_steps"] == 1.0
    assert metrics["boundary_end_mae_steps"] == 0.0
    assert metrics["timestep_precision"] == 0.75
    assert metrics["timestep_recall"] == 1.0


def test_partition_reader_preserves_interval_and_valid_normal_empty_annotation(
    tmp_path: Path,
) -> None:
    kpis = _raw_sequence()
    normal = {
        "sample_id": "normal-1",
        "kpis": {name: values for name, values in kpis.items()},
    }
    anomaly = {
        "sample_id": "anomaly-1",
        "kpis": {name: values for name, values in kpis.items()},
    }
    normal_metadata = {
        "sample_id": "normal-1",
        "source_file_id": "normal.jsonl",
        "label": "Normal",
        "sampling_rate": 10,
        "anomalies": {
            "exists": False,
            "anomaly_duration": {"start": None, "end": None},
            "affected_kpis": [""],
        },
    }
    anomaly_metadata = {
        "sample_id": "anomaly-1",
        "source_file_id": "anomaly.jsonl",
        "label": "Jamming",
        "sampling_rate": 10,
        "anomalies": {
            "exists": True,
            "anomaly_duration": {"start": 10, "end": 30},
            "affected_kpis": ["RSRP"],
        },
    }
    _write_jsonl_gz(tmp_path / "train_sequences.jsonl.gz", [normal, anomaly])
    _write_jsonl_gz(
        tmp_path / "train_metadata.jsonl.gz",
        [normal_metadata, anomaly_metadata],
    )

    records = _load_partition(tmp_path, "train", TelecomTSPreprocessor())

    assert records[0]["interval"] is None
    assert records[1]["interval"] == (10, 30)
    assert records[1]["affected_kpis"] == ["RSRP"]
    assert records[1]["sampling_rate_hz"] == 10.0


def test_partition_reader_rejects_malformed_interval(tmp_path: Path) -> None:
    sequence = {
        "sample_id": "anomaly-1",
        "kpis": _raw_sequence(),
    }
    metadata = {
        "sample_id": "anomaly-1",
        "source_file_id": "anomaly.jsonl",
        "label": "Jamming",
        "sampling_rate": 10,
        "anomalies": {
            "exists": True,
            "anomaly_duration": {"start": 90, "end": 12},
            "affected_kpis": ["RSRP"],
        },
    }
    _write_jsonl_gz(tmp_path / "validation_sequences.jsonl.gz", [sequence])
    _write_jsonl_gz(tmp_path / "validation_metadata.jsonl.gz", [metadata])

    with pytest.raises(SchemaValidationError, match="inclusive interval"):
        _load_partition(tmp_path, "validation", TelecomTSPreprocessor())


def test_group_split_validation_rejects_source_file_overlap() -> None:
    train = [{"sample_id": "train-1", "source_file_id": "shared.jsonl"}]
    validation = [{"sample_id": "valid-1", "source_file_id": "shared.jsonl"}]

    with pytest.raises(SchemaValidationError, match="Source-file overlap"):
        _validate_group_splits(train, validation)


def test_localization_report_visualizations_are_written(tmp_path: Path) -> None:
    metrics = _interval_metrics([(10, 30)], [(8, 32)])
    record = {
        "sample_id": "test-1",
        "sequence": {"RSRP": [float(value) for value in range(128)]},
        "interval": (10, 30),
        "affected_kpis": ["RSRP"],
    }

    _write_threshold_plot([(0.4, metrics)], metrics, tmp_path)
    _write_metrics_plot(metrics, metrics, metrics, metrics, tmp_path)
    _write_sample_plot([record], [(8, 32)], tmp_path)

    for filename in (
        "validation_thresholds.png",
        "interval_metrics.png",
        "sample_test_localization.png",
    ):
        assert (tmp_path / filename).is_file()
        assert (tmp_path / filename).stat().st_size > 0


def _write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
