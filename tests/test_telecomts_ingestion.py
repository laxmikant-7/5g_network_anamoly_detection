from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from cl5g.data.telecomts import (
    CATEGORICAL_KPIS,
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TAXONOMY_LABELS,
    SourceFile,
    process_source_files,
    validate_record,
)


def _record(
    *,
    anomaly_type: str = "Jamming",
    anomalous: bool = True,
    sample_id: str | None = "sample-1",
) -> dict[str, Any]:
    kpis: dict[str, list[Any]] = {
        name: [float(index) for index in range(128)]
        for name in NUMERIC_KPIS
    }
    for name in CATEGORICAL_KPIS:
        kpis[name] = ["TCP"] * 128
    kpis["UL_Protocol"][0] = ""
    kpis["DL_Protocol"][1] = "None"
    record: dict[str, Any] = {
        "start_time": "2025-01-01 00:00:00.000",
        "end_time": "2025-01-01 00:00:12.700",
        "sampling_rate": 10,
        "KPIs": kpis,
        "anomalies": {
            "exists": anomalous,
            "type": anomaly_type if anomalous else "",
            "anomaly_duration": (
                {"start": 10, "end": 20}
                if anomalous
                else {"start": None, "end": None}
            ),
            "affected_kpis": ["RSRP", "UL_SNR"] if anomalous else [""],
            "troubleshooting_tickets": "retained in raw source only",
        },
        "labels": {
            "zone": "A",
            "application": "File",
            "mobility": "No",
            "congestion": "No",
            "anomaly_present": "Yes" if anomalous else "No",
        },
    }
    if sample_id is not None:
        record["sample_id"] = sample_id
    return record


def _write_source(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if isinstance(record, str):
                stream.write(record + "\n")
            else:
                stream.write(json.dumps(record, allow_nan=True) + "\n")


def test_validate_record_preserves_dimensions_categories_and_annotations() -> None:
    source = _record()

    errors, normalized = validate_record(source)

    assert errors == []
    assert normalized is not None
    assert list(normalized["kpis"]) == list(KPI_NAMES)
    assert all(len(values) == 128 for values in normalized["kpis"].values())
    assert normalized["kpis"]["UL_Protocol"][:2] == ["", "TCP"]
    assert normalized["anomaly_type"] == "Jamming"
    assert normalized["annotations"]["anomaly_duration"] == {"start": 10, "end": 20}
    assert normalized["annotations"]["affected_kpis"] == ["RSRP", "UL_SNR"]


@pytest.mark.parametrize("source_label", TAXONOMY_LABELS[1:])
def test_each_taxonomy_label_maps_without_merging(source_label: str) -> None:
    errors, normalized = validate_record(_record(anomaly_type=source_label))

    assert errors == []
    assert normalized is not None
    assert normalized["source_anomaly_type"] == source_label
    assert normalized["anomaly_type"] == source_label


def test_normal_source_labels_map_to_normal_and_allow_absent_annotations() -> None:
    errors, normalized = validate_record(_record(anomalous=False, anomaly_type=""))

    assert errors == []
    assert normalized is not None
    assert normalized["anomaly_type"] == "Normal"
    assert normalized["binary_label"] is False
    assert normalized["annotations"]["anomaly_duration"] == {
        "start": None,
        "end": None,
    }


def test_validation_reports_unknown_labels_and_missing_values() -> None:
    source = _record(anomaly_type="Unmapped Category")
    source["KPIs"]["RSRP"][9] = float("inf")
    source["KPIs"]["DL_Protocol"][4] = None

    errors, normalized = validate_record(source)

    assert normalized is None
    assert any("not finite" in error for error in errors)
    assert any("missing or not a string" in error for error in errors)
    assert any("Unknown anomalous source label" in error for error in errors)


def test_validation_rejects_wrong_sequence_length_kpi_names_and_protocol_values() -> None:
    source = _record()
    source["KPIs"]["RSRP"] = source["KPIs"]["RSRP"][:127]
    source["KPIs"]["Unexpected_KPI"] = [1.0] * 128
    source["KPIs"]["UL_Protocol"][5] = "SCTP"

    errors, _ = validate_record(source)

    assert any("must contain 128 values" in error for error in errors)
    assert any("Unexpected KPI" in error for error in errors)
    assert any("unsupported categorical value" in error for error in errors)


def test_anomalous_records_require_temporal_and_affected_kpi_annotations() -> None:
    source = _record()
    source["anomalies"]["anomaly_duration"] = {"start": None, "end": None}
    source["anomalies"]["affected_kpis"] = [""]

    errors, normalized = validate_record(source)

    assert any("invalid inclusive anomaly duration" in error for error in errors)
    assert any("missing affected KPI annotations" in error for error in errors)


def test_processing_reports_malformed_and_duplicate_records_and_is_repeatable(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "raw" / "chunked.jsonl.gz"
    source = SourceFile(
        path="anomalous/jammer/File/processed/chunked.jsonl",
        size=0,
        oid="fixture",
    )
    _write_source(
        source_path,
        [
            _record(),
            _record(anomalous=False, anomaly_type="", sample_id="sample-2"),
            _record(sample_id="sample-1"),
            "{broken json",
        ],
    )
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"

    first = process_source_files(
        [(source, source_path)],
        revision=DEFAULT_REVISION,
        processed_directory=processed,
        report_directory=reports,
    )
    first_sequences = (processed / "sequences.jsonl.gz").read_bytes()
    first_metadata = (processed / "metadata.jsonl.gz").read_bytes()
    second = process_source_files(
        [(source, source_path)],
        revision=DEFAULT_REVISION,
        processed_directory=processed,
        report_directory=reports,
    )

    assert first["source_file_count"] == 1
    assert first["source_line_count"] == 4
    assert first["loaded_samples"] == 2
    assert first["normal_samples"] == 1
    assert first["anomalous_samples"] == 1
    assert first["invalid_samples"] == 2
    assert first["samples_with_temporal_annotations"] == 2
    assert first["samples_with_affected_kpi_annotations"] == 2
    assert first["source_anomaly_type_distribution"] == {"": 1, "Jamming": 2}
    assert second == first
    assert (processed / "sequences.jsonl.gz").read_bytes() == first_sequences
    assert (processed / "metadata.jsonl.gz").read_bytes() == first_metadata
    assert second["loaded_samples"] == 2
    assert second["invalid_samples"] == 2
    with gzip.open(reports / "invalid_records.jsonl.gz", "rt", encoding="utf-8") as stream:
        invalid = [json.loads(line) for line in stream]
    assert len(invalid) == 2
    assert any("Duplicate sample ID" in " ".join(entry["errors"]) for entry in invalid)
    assert any("Malformed JSON" in " ".join(entry["errors"]) for entry in invalid)
    with gzip.open(processed / "metadata.jsonl.gz", "rt", encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream]
    assert metadata[0]["source_file_id"] == source.path
    assert metadata[0]["source_line_number"] == 1
    assert metadata[0]["anomalies"]["affected_kpis"] == ["RSRP", "UL_SNR"]


def test_missing_source_sample_id_gets_a_stable_derived_id(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "chunked.jsonl.gz"
    source = SourceFile(
        path="normal/mobile/File/processed/chunked.jsonl",
        size=0,
        oid="fixture",
    )
    _write_source(
        source_path,
        [_record(anomalous=False, anomaly_type="", sample_id=None)],
    )
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"

    process_source_files(
        [(source, source_path)],
        revision=DEFAULT_REVISION,
        processed_directory=processed,
        report_directory=reports,
    )
    with gzip.open(processed / "metadata.jsonl.gz", "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
    with gzip.open(processed / "sequences.jsonl.gz", "rt", encoding="utf-8") as stream:
        sequence = json.loads(next(stream))

    assert metadata["source_sample_id_present"] is False
    assert metadata["sample_id"].startswith("telecomts:")
    assert sequence["sample_id"] == metadata["sample_id"]

