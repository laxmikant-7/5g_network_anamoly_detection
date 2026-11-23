from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from cl5g.data.telecomts import (
    ALLOWED_PROTOCOL_VALUES,
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TAXONOMY_LABELS,
)
from generate_historical_csv import (
    CSV_COLUMNS,
    HistoricalCsvError,
    generate_historical_csv,
)


def _record(sample_id: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a small deterministic unit-test fixture, not TelecomTS data."""
    sequence = {
        "sample_id": sample_id,
        "kpis": {
            name: (
                [float(index) for index in range(128)]
                if name in NUMERIC_KPIS
                else ["TCP", "UDP", "", "None"] * 32
            )
            for name in KPI_NAMES
        },
    }
    metadata = {
        "sample_id": sample_id,
        "start_time": "2025-01-01 00:00:00.000",
        "sampling_rate": 10,
        "label": label,
    }
    return sequence, metadata


def _write_jsonl_gz(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                (row if isinstance(row, str) else json.dumps(row)) + "\n"
            )


def _write_inputs(
    directory: Path,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    corrupt_sequence_line: bool = False,
) -> tuple[Path, Path]:
    sequences_path = directory / "sequences.jsonl.gz"
    metadata_path = directory / "metadata.jsonl.gz"
    sequences: list[object] = [sequence for sequence, _ in records]
    if corrupt_sequence_line:
        sequences.insert(0, "{corrupt json")
        metadata = [item for _, item in records]
        metadata.insert(0, records[0][1])
    else:
        metadata = [item for _, item in records]
    _write_jsonl_gz(sequences_path, sequences)
    _write_jsonl_gz(metadata_path, metadata)
    return sequences_path, metadata_path


def _generate(
    tmp_path: Path,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    samples_per_class: int = 100,
    max_output_bytes: int = 190 * 1024 * 1024,
) -> tuple[dict[str, Any], Path]:
    sequence_path, metadata_path = _write_inputs(tmp_path / "input", records)
    output = tmp_path / "output.csv"
    report_path = tmp_path / "report.json"
    report = generate_historical_csv(
        sequence_path,
        metadata_path,
        output,
        report_path,
        samples_per_class=samples_per_class,
        seed=42,
        max_output_bytes=max_output_bytes,
    )
    return report, output


def test_reads_gzip_jsonl_and_matches_sample_ids(tmp_path: Path) -> None:
    records = [
        _record("normal-1", "Normal"),
        _record("jam-1", "Jamming"),
    ]
    report, output = _generate(tmp_path, records, samples_per_class=2)

    assert report["dataset_revision"] == DEFAULT_REVISION
    assert report["total_selected_samples"] == 2
    assert report["invalid_samples_skipped"] == 0
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["entity_id"] for row in rows} == {"normal-1", "jam-1"}


def test_mismatched_metadata_id_is_reported_and_skipped(tmp_path: Path) -> None:
    sequence, metadata = _record("sample-a", "Jamming")
    metadata["sample_id"] = "sample-b"
    report, _ = _generate(
        tmp_path,
        [(sequence, metadata), _record("normal-1", "Normal")],
    )

    assert report["invalid_samples_skipped"] == 1
    assert "sample_id values do not match" in report["invalid_sample_examples"][0]["errors"][0]
    assert report["total_selected_samples"] == 1


def test_invalid_timestep_length_is_reported_not_used(tmp_path: Path) -> None:
    sequence, metadata = _record("short-kpi", "Normal")
    sequence["kpis"]["RSRP"] = sequence["kpis"]["RSRP"][:-1]
    report, _ = _generate(
        tmp_path,
        [(sequence, metadata), _record("normal-1", "Normal")],
    )

    assert report["invalid_samples_skipped"] == 1
    assert any("exactly 128 values" in error for error in report["invalid_sample_examples"][0]["errors"])
    assert report["total_selected_samples"] == 1


def test_categorical_values_are_preserved_without_numeric_encoding(tmp_path: Path) -> None:
    sequence, metadata = _record("protocol-sample", "Jamming")
    report, output = _generate(tmp_path, [(sequence, metadata)])

    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["UL_Protocol"] == "TCP"
    assert rows[1]["UL_Protocol"] == "UDP"
    assert rows[2]["UL_Protocol"] == ""
    assert rows[3]["UL_Protocol"] == "None"
    assert set(row["UL_Protocol"] for row in rows) <= ALLOWED_PROTOCOL_VALUES
    assert report["missing_values"] == 0


def test_each_sample_writes_exactly_128_rows_and_keeps_label(tmp_path: Path) -> None:
    report, output = _generate(
        tmp_path,
        [_record("normal-1", "Normal"), _record("jam-1", "Jamming")],
    )

    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 128
    assert report["total_output_rows"] == 256
    assert {row["anomaly_type"] for row in rows if row["entity_id"] == "jam-1"} == {
        "Jamming"
    }
    assert all(row["layer"] == "ran_du" for row in rows)


def test_csv_columns_follow_dashboard_canonical_order(tmp_path: Path) -> None:
    _, output = _generate(tmp_path, [_record("normal-1", "Normal")])

    with output.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        assert tuple(next(reader)) == CSV_COLUMNS
    assert CSV_COLUMNS[4:] == KPI_NAMES


def test_output_size_and_report_match_written_file(tmp_path: Path) -> None:
    report, output = _generate(tmp_path, [_record("normal-1", "Normal")])

    assert output.stat().st_size == report["output_size_bytes"]
    assert output.stat().st_size <= report["maximum_output_size_bytes"]


def test_corrupt_jsonl_record_is_counted_with_actionable_example(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    sequence_path, metadata_path = _write_inputs(
        input_dir,
        [_record("normal-1", "Normal")],
        corrupt_sequence_line=True,
    )
    output = tmp_path / "output.csv"
    report = generate_historical_csv(
        sequence_path,
        metadata_path,
        output,
        tmp_path / "report.json",
    )

    assert report["invalid_samples_skipped"] == 1
    assert any(
        "Malformed sequence JSON" in error
        for error in report["invalid_sample_examples"][0]["errors"]
    )
    assert report["total_selected_samples"] == 1


def test_corrupted_incomplete_class_sample_is_not_written(tmp_path: Path) -> None:
    sequence, metadata = _record("bad-protocol", "Jamming")
    sequence["kpis"]["DL_Protocol"][9] = "SCTP"
    report, output = _generate(
        tmp_path,
        [(sequence, metadata), _record("normal-1", "Normal")],
    )

    assert report["invalid_samples_skipped"] == 1
    assert report["total_output_rows"] == 128
    with output.open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 128


def test_missing_processed_files_explain_offline_ingestion_step(tmp_path: Path) -> None:
    with pytest.raises(HistoricalCsvError, match="--offline"):
        generate_historical_csv(
            tmp_path / "missing-sequences.jsonl.gz",
            tmp_path / "missing-metadata.jsonl.gz",
            tmp_path / "out.csv",
            tmp_path / "report.json",
        )


def test_output_limit_reduces_class_sample_count(tmp_path: Path) -> None:
    records = [
        _record(f"{label}-{index}", label)
        for label in TAXONOMY_LABELS
        for index in range(2)
    ]
    input_directory = tmp_path / "inputs"
    sequences_path, metadata_path = _write_inputs(input_directory, records)

    one_per_class = generate_historical_csv(
        sequences_path,
        metadata_path,
        tmp_path / "one.csv",
        tmp_path / "one.json",
        samples_per_class=1,
    )
    two_per_class = generate_historical_csv(
        sequences_path,
        metadata_path,
        tmp_path / "two.csv",
        tmp_path / "two.json",
        samples_per_class=2,
    )
    assert one_per_class["output_size_bytes"] < two_per_class["output_size_bytes"]
    limit = (one_per_class["output_size_bytes"] + two_per_class["output_size_bytes"]) // 2

    reduced = generate_historical_csv(
        sequences_path,
        metadata_path,
        tmp_path / "limited.csv",
        tmp_path / "limited.json",
        samples_per_class=2,
        max_output_bytes=limit,
    )

    assert reduced["final_samples_per_class_limit"] == 1
    assert reduced["total_selected_samples"] == len(TAXONOMY_LABELS)
    assert reduced["output_size_bytes"] <= limit
