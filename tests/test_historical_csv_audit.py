from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cl5g.data.telecomts import KPI_NAMES, NUMERIC_KPIS
from cl5g.data.historical_csv_audit import CSV_COLUMNS, audit_historical_csv


def _write_fixture(tmp_path: Path, *, bad_label: bool = False) -> tuple[Path, Path]:
    csv_path = tmp_path / "historical.csv"
    metadata_path = tmp_path / "metadata.jsonl.gz"
    start = datetime(2025, 1, 1)
    rows: list[dict[str, Any]] = []
    metadata_rows = []
    samples = (
        ("normal-a", "Normal", False, "", None, []),
        ("jam-a", "Jamming", True, "Jamming", {"start": 10, "end": 20}, ["RSRP"]),
    )
    for sample_id, label, exists, source_type, interval, affected in samples:
        for timestep in range(128):
            timestamp = start + timedelta(seconds=timestep / 10)
            row: dict[str, Any] = {
                "timestamp": timestamp.isoformat(sep=" ", timespec="microseconds"),
                "entity_id": sample_id,
                "layer": "ran_du",
                "anomaly_type": "Normal" if bad_label and label == "Jamming" else label,
            }
            for name in KPI_NAMES:
                row[name] = (
                    "TCP" if name in {"UL_Protocol", "DL_Protocol"} and timestep % 2 == 0
                    else "UDP" if name in {"UL_Protocol", "DL_Protocol"}
                    else float(KPI_NAMES.index(name))
                )
            rows.append(row)
        metadata_rows.append(
            {
                "sample_id": sample_id,
                "start_time": start.isoformat(sep=" ", timespec="milliseconds"),
                "sampling_rate": 10,
                "label": label,
                "binary_label": exists,
                "source_anomaly_type": source_type,
                "source_file_id": f"{label}.jsonl",
                "anomalies": {
                    "exists": exists,
                    "type": source_type,
                    "anomaly_duration": interval,
                    "affected_kpis": affected,
                },
            }
        )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        for item in metadata_rows:
            stream.write(json.dumps(item) + "\n")
    return csv_path, metadata_path


def test_audits_historical_csv_and_writes_required_reports_and_figures(tmp_path: Path) -> None:
    csv_path, metadata_path = _write_fixture(tmp_path)
    original_bytes = csv_path.read_bytes()
    output = tmp_path / "reports"

    report = audit_historical_csv(
        tmp_path,
        csv_path=csv_path,
        metadata_path=metadata_path,
        output_directory=output,
    )

    assert report["overall_validation_passed"] is True
    assert report["row_count"] == 256
    assert report["sample_count"] == 2
    assert report["sample_class_distribution"]["Normal"] == 1
    assert report["sample_class_distribution"]["Jamming"] == 1
    assert report["original_metadata_audit"]["selected_samples_matched"] == 2
    assert report["original_metadata_audit"]["label_mapping_mismatches"] == 0
    assert report["original_metadata_audit"]["selected_temporal_annotation_count"] == 1
    assert report["original_metadata_audit"]["selected_affected_kpi_annotation_count"] == 1
    assert report["timestamp_schedule_mismatch_samples"] == []
    assert report["columns_match_canonical_order"] is True
    assert tuple(report["canonical_columns"][4:]) == KPI_NAMES
    assert set(report["numeric_constant_features"]) == set(NUMERIC_KPIS)
    assert csv_path.read_bytes() == original_bytes
    assert (output / "dataset_quality_report.md").is_file()
    assert (output / "dataset_quality_report.json").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 5


def test_audit_flags_csv_label_that_disagrees_with_original_metadata(tmp_path: Path) -> None:
    csv_path, metadata_path = _write_fixture(tmp_path, bad_label=True)

    report = audit_historical_csv(
        tmp_path,
        csv_path=csv_path,
        metadata_path=metadata_path,
        output_directory=tmp_path / "reports",
    )

    assert report["overall_validation_passed"] is False
    assert report["checks"]["labels_match_ingestion_metadata"] is False
    assert report["original_metadata_audit"]["selected_label_mismatches"]


def test_audit_reports_entities_with_incorrect_sequence_length(tmp_path: Path) -> None:
    csv_path, metadata_path = _write_fixture(tmp_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(row for row in rows if row["entity_id"] != "jam-a" or row["timestamp"] != "2025-01-01 00:00:00.000000")

    report = audit_historical_csv(
        tmp_path,
        csv_path=csv_path,
        metadata_path=metadata_path,
        output_directory=tmp_path / "reports",
    )

    assert report["overall_validation_passed"] is False
    assert report["wrong_length_sample_count"] == 1
    assert report["checks"]["exactly_128_rows_per_entity"] is False
