"""Reproducible, source-file-aware TelecomTS partitions."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from cl5g.common.exceptions import SchemaValidationError
from cl5g.data.schema import SchemaConfig
from cl5g.data.telecomts import TAXONOMY_LABELS
from cl5g.pipeline.config import AppConfig, SplitConfig, load_config

_PARTITIONS = ("train", "validation", "test")
_SPLIT_FILE_NAMES = {
    "train": ("train_sequences.jsonl.gz", "train_metadata.jsonl.gz"),
    "validation": ("validation_sequences.jsonl.gz", "validation_metadata.jsonl.gz"),
    "test": ("test_sequences.jsonl.gz", "test_metadata.jsonl.gz"),
}
_NORMAL_LABEL = "Normal"


def split_source_grouped(
    sequences_path: Path,
    metadata_path: Path,
    output_directory: Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
    historical_test_data_used: bool = False,
) -> dict[str, Any]:
    """Create reproducible train/validation/test files without source overlap.

    Nested ``StratifiedGroupKFold`` folds provide candidate partitions. The
    selected candidate prioritizes feasible class support, then sample
    fractions and per-class proportions. Source-file groups are never split.
    """
    _validate_split_options(validation_fraction, test_fraction, seed)
    if not isinstance(sequences_path, Path) or not isinstance(metadata_path, Path):
        raise TypeError("sequences_path and metadata_path must be pathlib.Path values.")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path.")
    if not isinstance(historical_test_data_used, bool):
        raise TypeError("historical_test_data_used must be a bool.")
    if not sequences_path.is_file():
        raise FileNotFoundError(f"Processed sequences are missing: {sequences_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processed metadata are missing: {metadata_path}")

    sample_ids, source_ids, labels = _read_aligned_metadata(
        sequences_path,
        metadata_path,
    )
    train_indices, validation_indices, test_indices = _select_partitions(
        labels,
        source_ids,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    assignments: dict[str, str] = {}
    for partition, indices in zip(
        _PARTITIONS,
        (train_indices, validation_indices, test_indices),
        strict=True,
    ):
        for index in indices:
            source_id = source_ids[int(index)]
            existing = assignments.setdefault(source_id, partition)
            if existing != partition:
                raise RuntimeError(
                    f"Source file '{source_id}' was assigned to multiple partitions."
                )

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        counts = _write_partitions(
            sequences_path,
            metadata_path,
            output_directory,
            assignments,
            sample_ids,
            source_ids,
            labels,
        )
        report = _build_report(
            labels,
            source_ids,
            assignments,
            counts,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            historical_test_data_used=historical_test_data_used,
        )
        report_path = output_directory / "split_metadata.json"
        temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary_paths.append(temporary_report)
        _write_json(temporary_report, report)
        os.replace(temporary_report, report_path)
        return report
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise


def split_time_ordered(
    frame: pd.DataFrame,
    schema: SchemaConfig,
    split: SplitConfig,
) -> None:
    """Retain the reserved chronological splitter contract.

    TelecomTS source-aware partitions are implemented by
    :func:`split_source_grouped`; this legacy table splitter remains reserved.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(schema, SchemaConfig):
        raise TypeError("schema must be a SchemaConfig.")
    if not isinstance(split, SplitConfig):
        raise TypeError("split must be a SplitConfig.")
    if schema.timestamp_column not in frame.columns:
        raise SchemaValidationError(
            [f"Missing required column '{schema.timestamp_column}'."]
        )
    raise NotImplementedError(
        f"Time-ordered splitting ({split.strategy}) is not implemented."
    )


def _validate_split_options(
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")
        if not math.isfinite(float(value)) or value <= 0 or value >= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must sum to less than 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")


def _read_aligned_metadata(
    sequences_path: Path,
    metadata_path: Path,
) -> tuple[list[str], list[str], list[str]]:
    sample_ids: list[str] = []
    source_ids: list[str] = []
    labels: list[str] = []
    seen_sample_ids: set[str] = set()
    try:
        with (
            gzip.open(sequences_path, "rt", encoding="utf-8") as sequence_stream,
            gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
        ):
            for line_number, pair in enumerate(
                zip_longest(sequence_stream, metadata_stream),
                start=1,
            ):
                sequence_line, metadata_line = pair
                if sequence_line is None or metadata_line is None:
                    raise SchemaValidationError(
                        [f"Sequence and metadata files have different lengths at row {line_number}."]
                    )
                try:
                    sequence = json.loads(sequence_line)
                    metadata = json.loads(metadata_line)
                except json.JSONDecodeError as exc:
                    raise SchemaValidationError(
                        [f"Malformed JSON at aligned row {line_number}: {exc.msg}."]
                    ) from exc
                if not isinstance(sequence, dict) or not isinstance(metadata, dict):
                    raise SchemaValidationError(
                        [f"Aligned row {line_number} must contain JSON objects."]
                    )
                sample_id = sequence.get("sample_id")
                metadata_sample_id = metadata.get("sample_id")
                source_id = metadata.get("source_file_id")
                label = metadata.get("label")
                if (
                    not isinstance(sample_id, str)
                    or not sample_id.strip()
                    or metadata_sample_id != sample_id
                ):
                    raise SchemaValidationError(
                        [f"Sample IDs are missing or misaligned at row {line_number}."]
                    )
                if sample_id in seen_sample_ids:
                    raise SchemaValidationError(
                        [f"Duplicate sample ID '{sample_id}' at row {line_number}."]
                    )
                if not isinstance(sequence.get("kpis"), dict):
                    raise SchemaValidationError(
                        [f"Sequence at row {line_number} is missing its KPI mapping."]
                    )
                if not isinstance(source_id, str) or not source_id.strip():
                    raise SchemaValidationError(
                        [f"Missing source_file_id at row {line_number}."]
                    )
                if not isinstance(label, str) or not label.strip():
                    raise SchemaValidationError(
                        [f"Missing anomaly label at row {line_number}."]
                    )
                seen_sample_ids.add(sample_id)
                sample_ids.append(sample_id)
                source_ids.append(source_id)
                labels.append(label)
    except (OSError, EOFError, UnicodeError, gzip.BadGzipFile) as exc:
        raise SchemaValidationError([f"Could not read aligned TelecomTS files: {exc}"]) from exc

    if not sample_ids:
        raise SchemaValidationError(["The processed TelecomTS files contain no samples."])
    if len(set(source_ids)) < 3:
        raise SchemaValidationError(
            ["At least three source files are required for train/validation/test partitions."]
        )
    return sample_ids, source_ids, labels


def _select_partitions(
    labels: list[str],
    source_ids: list[str],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_fractions = np.asarray(
        [1.0 - validation_fraction - test_fraction, validation_fraction, test_fraction],
        dtype=np.float64,
    )
    group_count = len(set(source_ids))
    test_folds = min(group_count - 1, max(2, round(1.0 / test_fraction)))
    validation_share_of_remainder = validation_fraction / (1.0 - test_fraction)
    y = np.asarray(labels, dtype=str)
    groups = np.asarray(source_ids, dtype=str)
    unique_labels = sorted(set(labels))
    unique_groups = sorted(set(source_ids))
    label_indexes = {label: index for index, label in enumerate(unique_labels)}
    group_indexes = {source_id: index for index, source_id in enumerate(unique_groups)}
    group_label_counts = np.zeros((len(unique_groups), len(unique_labels)), dtype=np.int64)
    for label, source_id in zip(labels, source_ids, strict=True):
        group_label_counts[group_indexes[source_id], label_indexes[label]] += 1
    features = np.zeros((len(labels), 1), dtype=np.uint8)
    outer = StratifiedGroupKFold(
        n_splits=test_folds,
        shuffle=True,
        random_state=seed,
    )
    best: tuple[tuple[int, int, int, float, float], tuple[list[int], ...]] | None = None
    for outer_fold, (remainder_indices, test_indices) in enumerate(
        outer.split(features, y, groups)
    ):
        test_group_ids = sorted(set(groups[test_indices]))
        remainder_group_ids = sorted(set(groups[remainder_indices]))
        if len(remainder_group_ids) < 2:
            continue
        validation_group_count = min(
            len(remainder_group_ids) - 1,
            max(1, round(len(remainder_group_ids) * validation_share_of_remainder)),
        )
        inner = GroupShuffleSplit(
            n_splits=max(1024, min(2048, len(remainder_group_ids) * 64)),
            test_size=validation_group_count,
            random_state=seed + outer_fold + 1,
        )
        group_features = np.zeros((len(remainder_group_ids), 1), dtype=np.uint8)
        for train_relative, validation_relative in inner.split(
            group_features,
            groups=np.asarray(remainder_group_ids, dtype=str),
        ):
            train_group_ids = [remainder_group_ids[index] for index in train_relative]
            validation_group_ids = [
                remainder_group_ids[index] for index in validation_relative
            ]
            partition_group_indexes = tuple(
                [
                    group_indexes[source_id]
                    for source_id in source_group_ids
                ]
                for source_group_ids in (
                    train_group_ids,
                    validation_group_ids,
                    test_group_ids,
                )
            )
            score = _partition_score(
                partition_group_indexes,
                group_label_counts,
                unique_labels,
                target_fractions,
            )
            if best is None or score < best[0]:
                best = (score, partition_group_indexes)
    if best is None:
        raise SchemaValidationError(
            ["Unable to construct three non-empty source-grouped partitions."]
        )
    group_partition = {
        unique_groups[group_index]: partition
        for partition, partition_group_indexes in zip(
            _PARTITIONS,
            best[1],
            strict=True,
        )
        for group_index in partition_group_indexes
    }
    row_assignments = np.asarray(
        [group_partition[source_id] for source_id in source_ids],
        dtype=str,
    )
    return (
        np.flatnonzero(row_assignments == "train"),
        np.flatnonzero(row_assignments == "validation"),
        np.flatnonzero(row_assignments == "test"),
    )


def _partition_score(
    partition_group_indexes: tuple[list[int], ...],
    group_label_counts: np.ndarray,
    labels: list[str],
    target_fractions: np.ndarray,
) -> tuple[int, int, int, float, float]:
    total_class_counts = group_label_counts.sum(axis=0)
    total_samples = int(total_class_counts.sum())
    partition_class_counts = [
        group_label_counts[indices].sum(axis=0)
        for indices in partition_group_indexes
    ]
    sample_fractions = np.asarray(
        [int(counts.sum()) / total_samples for counts in partition_class_counts],
        dtype=np.float64,
    )
    size_error = float(
        np.sum(((sample_fractions - target_fractions) / target_fractions) ** 2)
    )
    missing_by_partition: list[int] = []
    class_source_counts = np.count_nonzero(group_label_counts, axis=0)
    for class_counts in partition_class_counts:
        missing_by_partition.append(
            sum(
                class_counts[index] == 0
                for index in range(len(labels))
                if class_source_counts[index] >= len(_PARTITIONS)
            )
        )
    ordered_missing = (
        missing_by_partition[2],
        missing_by_partition[1],
        missing_by_partition[0],
    )

    class_proportion_error = 0.0
    for class_index in range(len(labels)):
        class_fractions = [
            counts[class_index] / total_class_counts[class_index]
            for counts in partition_class_counts
        ]
        class_proportion_error += sum(
            abs(actual - target)
            for actual, target in zip(
                class_fractions,
                target_fractions,
                strict=True,
            )
        )
    class_proportion_error /= max(1, len(labels))
    return (
        *ordered_missing,
        round(size_error, 12),
        round(class_proportion_error, 12),
    )


def _write_partitions(
    sequences_path: Path,
    metadata_path: Path,
    output_directory: Path,
    assignments: dict[str, str],
    expected_sample_ids: list[str],
    expected_source_ids: list[str],
    expected_labels: list[str],
) -> dict[str, int]:
    temporary_files: dict[str, tuple[Path, Path]] = {}
    streams: dict[str, tuple[gzip.GzipFile, gzip.GzipFile, Any, Any]] = {}
    counts = {partition: 0 for partition in _PARTITIONS}
    for partition in _PARTITIONS:
        sequence_name, metadata_name = _SPLIT_FILE_NAMES[partition]
        sequence_tmp = output_directory / (sequence_name + ".tmp")
        metadata_tmp = output_directory / (metadata_name + ".tmp")
        temporary_files[partition] = (sequence_tmp, metadata_tmp)
    try:
        for partition, (sequence_tmp, metadata_tmp) in temporary_files.items():
            sequence_raw = sequence_tmp.open("wb")
            metadata_raw = metadata_tmp.open("wb")
            sequence_gzip = gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=sequence_raw,
                mtime=0,
            )
            metadata_gzip = gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=metadata_raw,
                mtime=0,
            )
            streams[partition] = (
                sequence_gzip,
                metadata_gzip,
                sequence_raw,
                metadata_raw,
            )

        seen_ids: set[str] = set()
        with (
            gzip.open(sequences_path, "rt", encoding="utf-8") as sequence_stream,
            gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
        ):
            for index, pair in enumerate(
                zip_longest(sequence_stream, metadata_stream),
            ):
                sequence_line, metadata_line = pair
                if sequence_line is None or metadata_line is None:
                    raise SchemaValidationError(
                        ["Aligned source files changed while partitions were being written."]
                    )
                sequence = json.loads(sequence_line)
                metadata = json.loads(metadata_line)
                sample_id = sequence["sample_id"]
                if (
                    index >= len(expected_sample_ids)
                    or sample_id != expected_sample_ids[index]
                    or metadata.get("sample_id") != expected_sample_ids[index]
                    or metadata.get("source_file_id") != expected_source_ids[index]
                    or metadata.get("label") != expected_labels[index]
                ):
                    raise SchemaValidationError(
                        [
                            "Source sample IDs, groups, or labels changed while "
                            f"writing row {index + 1}."
                        ]
                    )
                partition = assignments[metadata["source_file_id"]]
                sequence_gzip, metadata_gzip, _, _ = streams[partition]
                sequence_gzip.write(_json_line(sequence))
                metadata_gzip.write(_json_line(metadata))
                counts[partition] += 1
                seen_ids.add(sample_id)
        if len(seen_ids) != len(expected_sample_ids):
            raise SchemaValidationError(
                ["Not all sample IDs were written to exactly one partition."]
            )
        for stream_set in streams.values():
            for stream in stream_set:
                stream.close()
        for partition, paths in temporary_files.items():
            for temporary_path, final_name in zip(
                paths,
                _SPLIT_FILE_NAMES[partition],
                strict=True,
            ):
                os.replace(temporary_path, output_directory / final_name)
        return counts
    except Exception:
        for stream_set in streams.values():
            for stream in stream_set:
                if not stream.closed:
                    stream.close()
        for paths in temporary_files.values():
            for path in paths:
                path.unlink(missing_ok=True)
        raise


def _build_report(
    labels: list[str],
    source_ids: list[str],
    assignments: dict[str, str],
    counts: dict[str, int],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    historical_test_data_used: bool,
) -> dict[str, Any]:
    all_labels = sorted(set(TAXONOMY_LABELS) | set(labels))
    labels_by_partition = {
        partition: {
            label: 0
            for label in all_labels
        }
        for partition in _PARTITIONS
    }
    sources_by_partition = {partition: set() for partition in _PARTITIONS}
    class_sources: dict[str, dict[str, set[str]]] = {
        label: {partition: set() for partition in _PARTITIONS}
        for label in all_labels
    }
    for label, source_id in zip(labels, source_ids, strict=True):
        partition = assignments[source_id]
        labels_by_partition[partition][label] += 1
        sources_by_partition[partition].add(source_id)
        class_sources[label][partition].add(source_id)

    expected_samples = len(labels)
    partition_summary: dict[str, Any] = {}
    for partition in _PARTITIONS:
        distribution = labels_by_partition[partition]
        partition_summary[partition] = {
            "sample_count": counts[partition],
            "sample_fraction": counts[partition] / expected_samples,
            "normal_samples": distribution.get(_NORMAL_LABEL, 0),
            "anomalous_samples": sum(
                amount
                for label, amount in distribution.items()
                if label != _NORMAL_LABEL
            ),
            "label_distribution": distribution,
            "source_file_count": len(sources_by_partition[partition]),
            "source_files": sorted(sources_by_partition[partition]),
        }

    overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(_PARTITIONS):
        for right in _PARTITIONS[left_index + 1 :]:
            overlap[f"{left}_and_{right}"] = sorted(
                sources_by_partition[left] & sources_by_partition[right]
            )
    class_support = {
        label: {
            partition: {
                "samples": labels_by_partition[partition][label],
                "source_files": len(class_sources[label][partition]),
            }
            for partition in _PARTITIONS
        }
        for label in sorted(class_sources)
    }
    anomaly_labels = sorted(label for label in class_support if label != _NORMAL_LABEL)
    missing_test_categories = [
        label
        for label in anomaly_labels
        if class_support[label]["test"]["samples"] == 0
    ]
    missing_validation_categories = [
        label
        for label in anomaly_labels
        if class_support[label]["validation"]["samples"] == 0
    ]

    return {
        "split_method": (
            "StratifiedGroupKFold test candidates and seeded GroupShuffleSplit "
            "validation candidates on source_file_id"
        ),
        "seed": seed,
        "requested_fractions": {
            "train": 1.0 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "total_samples": expected_samples,
        "total_source_files": len(set(source_ids)),
        "partition_summary": partition_summary,
        "source_overlap": overlap,
        "class_support": class_support,
        "missing_anomaly_categories": {
            "validation": missing_validation_categories,
            "test": missing_test_categories,
        },
        "historical_test_data": {
            "previously_used": historical_test_data_used,
            "test_set_pristine": not historical_test_data_used,
            "assessment": (
                "The caller reported prior test-data use; this partition is "
                "historical and must not be presented as an untouched final test."
                if historical_test_data_used
                else (
                    "No prior model or evaluation artifacts were found in this "
                    "workspace before generating this partition. Declare any "
                    "external prior test use with --historical-test-data-used."
                )
            ),
        },
        "usage_policy": (
            "Use validation for feature/model selection and threshold tuning. "
            "Do not use the test partition for model-selection decisions."
        ),
    }


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Create TelecomTS train/validation/test partitions by source file."
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--revision", default="1b3a88a440f4922edfcd1e230e42c003c0e13bdf")
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--historical-test-data-used",
        action="store_true",
        help="Mark the generated test partition as previously used, not pristine.",
    )
    args = parser.parse_args()
    config_path = (
        args.project_root / "config.yaml"
        if args.project_root is not None
        else None
    )
    config: AppConfig = load_config(config_path)
    if config.split.strategy != "source_file_grouped":
        parser.error(
            "config.yaml split.strategy must be 'source_file_grouped' for this command."
        )
    revision = args.revision
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        parser.error("--revision must be a 40-character lowercase commit SHA.")
    processed_directory = config.resolve(config.paths.processed) / "telecomts" / revision
    seed = args.seed if args.seed is not None else config.seed
    output_directory = processed_directory / "splits" / f"seed-{seed}"
    report = split_source_grouped(
        processed_directory / "sequences.jsonl.gz",
        processed_directory / "metadata.jsonl.gz",
        output_directory,
        validation_fraction=(
            args.validation_fraction
            if args.validation_fraction is not None
            else config.split.validation_fraction
        ),
        test_fraction=(
            args.test_fraction
            if args.test_fraction is not None
            else config.split.test_fraction
        ),
        seed=seed,
        historical_test_data_used=args.historical_test_data_used,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "samples": {
                    name: summary["sample_count"]
                    for name, summary in report["partition_summary"].items()
                },
                "missing_test_categories": report["missing_anomaly_categories"]["test"],
            },
            indent=2,
        )
    )
    return 0


def main() -> None:
    """Console entry point."""
    raise SystemExit(_run_cli())


if __name__ == "__main__":
    main()
