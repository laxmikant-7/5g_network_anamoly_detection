from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from cl5g.common.exceptions import SchemaValidationError
from cl5g.data.split import split_source_grouped


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            mtime=0,
        ) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
                )


def _dataset(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    sequences: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    labels = ("Normal", "Jamming", "Co-Channel Interference (Mild)")
    for group_index in range(12):
        source_id = f"source-{group_index:02d}"
        label = labels[group_index % len(labels)]
        for sample_index in range(2 + group_index % 3):
            sample_id = f"{source_id}-sample-{sample_index}"
            sequences.append(
                {
                    "sample_id": sample_id,
                    "kpis": {
                        "RSRP": [float(group_index), float(sample_index)],
                    },
                }
            )
            metadata.append(
                {
                    "sample_id": sample_id,
                    "source_file_id": source_id,
                    "label": label,
                    "binary_label": label != "Normal",
                }
            )
    sequences_path = tmp_path / "sequences.jsonl.gz"
    metadata_path = tmp_path / "metadata.jsonl.gz"
    _write_jsonl_gz(sequences_path, sequences)
    _write_jsonl_gz(metadata_path, metadata)
    return sequences_path, metadata_path, sequences, metadata


def _read_partition(directory: Path, name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    for suffix in ("sequences", "metadata"):
        path = directory / f"{name}_{suffix}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            results.append([json.loads(line) for line in stream])
    return results[0], results[1]


def test_source_grouped_split_has_no_source_file_overlap(tmp_path: Path) -> None:
    sequences_path, metadata_path, _, _ = _dataset(tmp_path)

    report = split_source_grouped(
        sequences_path,
        metadata_path,
        tmp_path / "split",
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )

    assert report["split_method"] == (
        "StratifiedGroupKFold test candidates and seeded GroupShuffleSplit "
        "validation candidates on source_file_id"
    )
    assert report["source_overlap"] == {
        "train_and_validation": [],
        "train_and_test": [],
        "validation_and_test": [],
    }
    assert "Antenna Failure" in report["missing_anomaly_categories"]["test"]
    assert report["class_support"]["Antenna Failure"]["test"]["samples"] == 0
    source_sets = [
        set(report["partition_summary"][name]["source_files"])
        for name in ("train", "validation", "test")
    ]
    assert not (source_sets[0] & source_sets[1])
    assert not (source_sets[0] & source_sets[2])
    assert not (source_sets[1] & source_sets[2])


def test_source_grouped_split_is_reproducible(tmp_path: Path) -> None:
    sequences_path, metadata_path, _, _ = _dataset(tmp_path)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = split_source_grouped(
        sequences_path,
        metadata_path,
        first,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=23,
    )
    second_report = split_source_grouped(
        sequences_path,
        metadata_path,
        second,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=23,
    )

    assert first_report == second_report
    for name in ("train", "validation", "test"):
        assert _read_partition(first, name) == _read_partition(second, name)
        for suffix in ("sequences", "metadata"):
            first_file = first / f"{name}_{suffix}.jsonl.gz"
            second_file = second / f"{name}_{suffix}.jsonl.gz"
            assert first_file.read_bytes() == second_file.read_bytes()


def test_source_grouped_split_accounts_for_records_without_changing_content(
    tmp_path: Path,
) -> None:
    sequences_path, metadata_path, original_sequences, original_metadata = _dataset(tmp_path)
    output = tmp_path / "split"

    report = split_source_grouped(
        sequences_path,
        metadata_path,
        output,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )

    collected_sequences: dict[str, dict[str, Any]] = {}
    collected_metadata: dict[str, dict[str, Any]] = {}
    for partition in ("train", "validation", "test"):
        sequences, metadata = _read_partition(output, partition)
        assert len(sequences) == len(metadata)
        for sequence, row in zip(sequences, metadata, strict=True):
            sample_id = sequence["sample_id"]
            assert sample_id == row["sample_id"]
            assert sample_id not in collected_sequences
            collected_sequences[sample_id] = sequence
            collected_metadata[sample_id] = row

    original_by_id = {row["sample_id"]: row for row in original_sequences}
    original_metadata_by_id = {row["sample_id"]: row for row in original_metadata}
    assert set(collected_sequences) == set(original_by_id)
    assert set(collected_metadata) == set(original_metadata_by_id)
    for sample_id, original in original_by_id.items():
        assert collected_sequences[sample_id] == original
        assert collected_metadata[sample_id]["label"] == original_metadata_by_id[sample_id]["label"]
        assert collected_metadata[sample_id] == original_metadata_by_id[sample_id]
    assert sum(
        entry["sample_count"]
        for entry in report["partition_summary"].values()
    ) == len(original_sequences)

    persisted_report = json.loads((output / "split_metadata.json").read_text(encoding="utf-8"))
    assert persisted_report == report


def test_misaligned_inputs_are_rejected_without_silent_drops(tmp_path: Path) -> None:
    sequences_path, metadata_path, sequences, metadata = _dataset(tmp_path)
    metadata[0]["sample_id"] = "different-id"
    _write_jsonl_gz(metadata_path, metadata)

    with pytest.raises(SchemaValidationError, match="Sample IDs are missing or misaligned"):
        split_source_grouped(sequences_path, metadata_path, tmp_path / "split")


def test_prior_test_use_is_marked_non_pristine(tmp_path: Path) -> None:
    sequences_path, metadata_path, _, _ = _dataset(tmp_path)

    report = split_source_grouped(
        sequences_path,
        metadata_path,
        tmp_path / "historical",
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=42,
        historical_test_data_used=True,
    )

    assert report["historical_test_data"]["previously_used"] is True
    assert report["historical_test_data"]["test_set_pristine"] is False
