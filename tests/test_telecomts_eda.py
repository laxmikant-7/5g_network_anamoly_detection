from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from cl5g.data.telecomts import CATEGORICAL_KPIS, DEFAULT_REVISION, NUMERIC_KPIS
from cl5g.data.telecomts_eda import run_eda


def _sequence(sample_id: str, offset: float) -> dict[str, Any]:
    kpis: dict[str, list[Any]] = {
        name: [offset + index / 10 for index in range(128)]
        for name in NUMERIC_KPIS
    }
    for name in CATEGORICAL_KPIS:
        kpis[name] = ["TCP"] * 128
    kpis["UL_Protocol"][0] = ""
    kpis["DL_Protocol"][1] = "None"
    return {"sample_id": sample_id, "kpis": kpis}


def _metadata(sample_id: str, label: str, source: str) -> dict[str, Any]:
    anomalous = label != "Normal"
    return {
        "sample_id": sample_id,
        "source_file_id": source,
        "source_line_number": 1,
        "label": label,
        "binary_label": anomalous,
        "start_time": "2025-01-01 00:00:00.000",
        "end_time": "2025-01-01 00:00:12.700",
        "sampling_rate": 10,
        "source_anomaly_type": label if anomalous else "",
        "anomalies": {
            "exists": anomalous,
            "type": label if anomalous else "",
            "anomaly_duration": (
                {"start": 10, "end": 20}
                if anomalous
                else {"start": None, "end": None}
            ),
            "affected_kpis": ["RSRP"] if anomalous else [],
        },
        "labels": {
            "zone": "A",
            "application": "File",
            "mobility": "No",
            "congestion": "No",
            "anomaly_present": "Yes" if anomalous else "No",
        },
    }


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_eda_generates_reports_and_all_requested_visualizations(tmp_path: Path) -> None:
    revision = DEFAULT_REVISION
    processed = tmp_path / "data" / "processed" / "telecomts" / revision
    reports = tmp_path / "data" / "reports" / "telecomts" / revision
    _write_gzip_jsonl(
        processed / "sequences.jsonl.gz",
        [_sequence("normal-1", 1.0), _sequence("anomaly-1", 2.0)],
    )
    _write_gzip_jsonl(
        processed / "metadata.jsonl.gz",
        [
            _metadata("normal-1", "Normal", "normal/mobile/File/processed/chunked.jsonl"),
            _metadata(
                "anomaly-1",
                "Jamming",
                "anomalous/jammer/File/processed/chunked.jsonl",
            ),
        ],
    )
    (reports / "validation.json").parent.mkdir(parents=True, exist_ok=True)
    (reports / "validation.json").write_text(
        json.dumps(
            {
                "dataset_id": "AliMaatouk/TelecomTS",
                "revision": revision,
                "loaded_samples": 2,
                "source_file_count": 2,
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(reports / "invalid_records.jsonl.gz", "wt", encoding="utf-8"):
        pass

    result = run_eda(tmp_path, revision)

    assert result["statistics"]["valid_samples"] == 2
    assert result["statistics"]["normal_samples"] == 1
    assert result["statistics"]["anomalous_samples"] == 1
    assert result["statistics"]["temporal_annotations"]["samples_with_intervals"] == 1
    assert result["quality"]["invalid_processed_samples"] == 0
    assert result["quality"]["ingestion_invalid_samples"] == 0
    assert result["quality"]["matches_ingestion_sample_count"]
    output = reports / "eda"
    assert (output / "dataset_statistics.json").is_file()
    assert (output / "data_quality.json").is_file()
    assert (output / "preprocessing_config.json").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 10
    second_result = run_eda(tmp_path, revision)
    assert second_result == result


def test_eda_reports_sequence_metadata_mismatch_instead_of_silently_dropping(
    tmp_path: Path,
) -> None:
    revision = DEFAULT_REVISION
    processed = tmp_path / "data" / "processed" / "telecomts" / revision
    reports = tmp_path / "data" / "reports" / "telecomts" / revision
    _write_gzip_jsonl(processed / "sequences.jsonl.gz", [_sequence("normal-1", 1.0)])
    _write_gzip_jsonl(processed / "metadata.jsonl.gz", [])
    (reports / "validation.json").parent.mkdir(parents=True, exist_ok=True)
    (reports / "validation.json").write_text(
        json.dumps(
            {
                "dataset_id": "AliMaatouk/TelecomTS",
                "revision": revision,
                "loaded_samples": 1,
                "source_file_count": 1,
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(reports / "invalid_records.jsonl.gz", "wt", encoding="utf-8"):
        pass

    result = run_eda(tmp_path, revision)

    assert result["quality"]["invalid_processed_samples"] == 1
    assert result["quality"]["sequence_metadata_row_count_mismatches"] == 1
    assert result["quality"]["valid_samples"] == 0
    assert result["quality"]["sample_issues"][0]["errors"] == [
        "sequence_metadata_row_count_mismatch"
    ]
