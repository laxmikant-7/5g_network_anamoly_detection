"""Streaming quality analysis and exploratory plots for ingested TelecomTS."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cl5g.data.telecomts import (
    CATEGORICAL_KPIS,
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TAXONOMY_LABELS,
    TIME_STEPS,
)
from cl5g.data.telecomts_preprocessing import (
    PROTOCOL_CATEGORIES,
    TelecomTSPreprocessor,
    write_preprocessing_configuration,
)

_LOGGER = logging.getLogger("data.telecomts_eda")
_SENTINEL = object()
_SELECTED_KPIS = ("RSRP", "DL_BLER", "UL_SNR", "PRB_Utilization_DL")
_DISTRIBUTION_SAMPLE_STRIDE = 16
_CORRELATION_BATCH_SAMPLES = 128


def run_eda(project_root: Path, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    """Analyze the processed sequences and write reports and ten figures."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a 40-character lowercase hexadecimal commit.")
    project_root = project_root.resolve()
    processed_directory = project_root / "data" / "processed" / "telecomts" / revision
    report_directory = project_root / "data" / "reports" / "telecomts" / revision
    sequences_path = processed_directory / "sequences.jsonl.gz"
    metadata_path = processed_directory / "metadata.jsonl.gz"
    ingestion_report_path = report_directory / "validation.json"
    invalid_records_path = report_directory / "invalid_records.jsonl.gz"
    for path in (
        sequences_path,
        metadata_path,
        ingestion_report_path,
        invalid_records_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required TelecomTS ingestion output is missing: {path}")

    ingestion_report = json.loads(ingestion_report_path.read_text(encoding="utf-8"))
    if not isinstance(ingestion_report, dict):
        raise ValueError("The ingestion validation report must be a JSON object.")
    if (
        ingestion_report.get("dataset_id") != "AliMaatouk/TelecomTS"
        or ingestion_report.get("revision") != revision
    ):
        raise ValueError(
            "The ingestion report dataset/revision does not match the requested EDA revision."
        )
    figures_directory = report_directory / "eda" / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    preprocessor = TelecomTSPreprocessor()
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_label_counts: dict[str, Counter[str]] = {}
    protocol_counts: dict[str, Counter[str]] = {
        name: Counter() for name in CATEGORICAL_KPIS
    }
    missing_counts: Counter[str] = Counter()
    nonfinite_counts: Counter[str] = Counter()
    quality_error_counts: Counter[str] = Counter()
    quality_issues: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    numeric_sums = np.zeros(len(NUMERIC_KPIS), dtype=np.float64)
    numeric_squares = np.zeros(len(NUMERIC_KPIS), dtype=np.float64)
    numeric_minima = np.full(len(NUMERIC_KPIS), np.inf)
    numeric_maxima = np.full(len(NUMERIC_KPIS), -np.inf)
    correlation_sum = np.zeros(len(NUMERIC_KPIS), dtype=np.float64)
    correlation_cross_sum = np.zeros(
        (len(NUMERIC_KPIS), len(NUMERIC_KPIS)), dtype=np.float64
    )
    correlation_count = 0
    correlation_blocks: list[np.ndarray] = []
    distribution_samples: list[np.ndarray] = []
    per_sample_medians: dict[str, list[np.ndarray]] = {
        "Normal": [],
        "Anomalous": [],
    }
    intervals: list[tuple[int, int, int]] = []
    samples_with_affected_kpi_annotations = 0
    examples: dict[str, dict[str, Any]] = {}
    row_count = 0
    valid_count = 0
    mismatch_count = 0
    def add_issue(row: int, sample_id: Any, errors: list[str]) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        for error in errors:
            category = error.split(":", 1)[0]
            quality_error_counts[category] += 1
        if len(quality_issues) < 100:
            quality_issues.append(
                {"row": row, "sample_id": sample_id, "errors": errors}
            )
        _LOGGER.error("Invalid processed sample at row %s: %s", row, "; ".join(errors))

    with (
        gzip.open(sequences_path, "rt", encoding="utf-8") as sequences_stream,
        gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
    ):
        for row_number, pair in enumerate(
            zip_longest(sequences_stream, metadata_stream, fillvalue=_SENTINEL),
            start=1,
        ):
            sequence_line, metadata_line = pair
            row_count += 1
            if sequence_line is _SENTINEL or metadata_line is _SENTINEL:
                add_issue(row_number, None, ["sequence_metadata_row_count_mismatch"])
                continue
            try:
                sequence = json.loads(sequence_line)
                metadata = json.loads(metadata_line)
            except json.JSONDecodeError as exc:
                add_issue(row_number, None, [f"malformed_json: {exc.msg}"])
                continue
            sample_id = sequence.get("sample_id") if isinstance(sequence, dict) else None
            errors: list[str] = []
            if not isinstance(sequence, dict):
                errors.append("invalid_sequence_record: expected an object")
            if not isinstance(metadata, dict):
                errors.append("invalid_metadata_record: expected an object")
            if errors:
                add_issue(row_number, sample_id, errors)
                continue
            if not isinstance(sample_id, str) or not sample_id:
                errors.append("invalid_sample_id: expected a non-empty string")
            if metadata.get("sample_id") != sample_id:
                errors.append("sample_id_mismatch: sequence and metadata IDs differ")
            if sample_id in seen_sample_ids:
                errors.append("duplicate_sample_id: ID already appeared")
            else:
                seen_sample_ids.add(sample_id)
            if not isinstance(metadata.get("source_file_id"), str):
                errors.append("invalid_source_file_id: expected a string")
            kpis = sequence.get("kpis")
            if not isinstance(kpis, Mapping):
                errors.append("invalid_kpis: expected an object")
            else:
                missing_names = set(KPI_NAMES) - set(kpis)
                unexpected_names = set(kpis) - set(KPI_NAMES)
                if missing_names:
                    errors.append(
                        "missing_kpi_names: " + ", ".join(sorted(missing_names))
                    )
                if unexpected_names:
                    errors.append(
                        "unexpected_kpi_names: "
                        + ", ".join(sorted(str(name) for name in unexpected_names))
                    )
                for name in NUMERIC_KPIS:
                    values = kpis.get(name)
                    if not isinstance(values, list) or len(values) != TIME_STEPS:
                        errors.append(f"invalid_sequence_dimensions: {name}")
                        continue
                    for value in values:
                        if value is None:
                            missing_counts[name] += 1
                        elif isinstance(value, bool) or not isinstance(
                            value, (int, float)
                        ):
                            errors.append(f"invalid_numerical_type: {name}")
                            break
                        elif not math.isfinite(float(value)):
                            nonfinite_counts[name] += 1
                            errors.append(f"non_finite_numerical_value: {name}")
                            break
                for name in CATEGORICAL_KPIS:
                    values = kpis.get(name)
                    if not isinstance(values, list) or len(values) != TIME_STEPS:
                        errors.append(f"invalid_sequence_dimensions: {name}")
                        continue
                    for value in values:
                        if value is None:
                            missing_counts[name] += 1
                        elif not isinstance(value, str) or value not in {
                            "",
                            "None",
                            "TCP",
                            "UDP",
                        }:
                            errors.append(f"unsupported_categorical_value: {name}")
                            break
            label = metadata.get("label")
            if label not in TAXONOMY_LABELS:
                errors.append("invalid_label: unsupported or missing label")
            source_label = metadata.get("source_anomaly_type")
            expected_source_label = "" if label == "Normal" else label
            if source_label != expected_source_label:
                errors.append("invalid_label_mapping: source and normalized labels differ")
            anomalies = metadata.get("anomalies")
            if not isinstance(anomalies, Mapping):
                errors.append("invalid_annotation: anomalies must be an object")
            else:
                is_anomaly = label != "Normal"
                if anomalies.get("exists") is not is_anomaly:
                    errors.append("invalid_annotation: label/presence disagreement")
                if metadata.get("binary_label") is not is_anomaly:
                    errors.append("invalid_label: binary label/presence disagreement")
                if is_anomaly:
                    duration = anomalies.get("anomaly_duration")
                    if not isinstance(duration, Mapping):
                        errors.append("missing_temporal_annotation: interval is absent")
                    else:
                        start, end = duration.get("start"), duration.get("end")
                        if (
                            isinstance(start, bool)
                            or isinstance(end, bool)
                            or not isinstance(start, int)
                            or not isinstance(end, int)
                            or start < 0
                            or end < start
                            or end >= TIME_STEPS
                        ):
                            errors.append("invalid_temporal_annotation: interval bounds")
                    affected = anomalies.get("affected_kpis")
                    if not isinstance(affected, list) or not affected:
                        errors.append("missing_affected_kpi_annotation: list is empty")
                    elif any(
                        not isinstance(name, str) or name not in KPI_NAMES
                        for name in affected
                    ):
                        errors.append("invalid_affected_kpi_annotation: unknown KPI")
                else:
                    duration = anomalies.get("anomaly_duration")
                    affected = anomalies.get("affected_kpis")
                    if (
                        isinstance(duration, Mapping)
                        and (
                            duration.get("start") is not None
                            or duration.get("end") is not None
                        )
                    ) or (isinstance(affected, list) and any(affected)):
                        errors.append("unexpected_normal_annotation: anomaly details present")
            labels = metadata.get("labels")
            expected_presence = "No" if label == "Normal" else "Yes"
            if (
                not isinstance(labels, Mapping)
                or labels.get("anomaly_present") != expected_presence
            ):
                errors.append("invalid_context_label: anomaly_present disagrees with label")
            if errors:
                add_issue(row_number, sample_id, errors)
                continue

            try:
                encoded = preprocessor.transform_sample(sequence)
            except (TypeError, ValueError) as exc:
                add_issue(row_number, sample_id, [f"preprocessing_validation: {exc}"])
                continue

            numeric = encoded[:, : len(NUMERIC_KPIS)]
            label_counts[label] += 1
            source_id = metadata["source_file_id"]
            source_counts[source_id] += 1
            source_label_counts.setdefault(source_id, Counter())[label] += 1
            for name in CATEGORICAL_KPIS:
                protocol_counts[name].update(kpis[name])
            numeric_sums += numeric.sum(axis=0)
            numeric_squares += np.square(numeric).sum(axis=0)
            numeric_minima = np.minimum(numeric_minima, numeric.min(axis=0))
            numeric_maxima = np.maximum(numeric_maxima, numeric.max(axis=0))
            correlation_blocks.append(numeric)
            if len(correlation_blocks) >= _CORRELATION_BATCH_SAMPLES:
                block = np.concatenate(correlation_blocks, axis=0)
                correlation_sum += block.sum(axis=0)
                correlation_cross_sum += block.T @ block
                correlation_count += block.shape[0]
                correlation_blocks.clear()
            distribution_samples.append(numeric[::_DISTRIBUTION_SAMPLE_STRIDE])
            binary_class = "Normal" if label == "Normal" else "Anomalous"
            per_sample_medians[binary_class].append(np.median(numeric, axis=0))

            example: dict[str, Any] = {
                "label": label,
                "kpis": {name: list(kpis[name]) for name in _SELECTED_KPIS},
                "interval": None,
            }
            if binary_class == "Normal" and "Normal" not in examples:
                examples["Normal"] = example
            if binary_class == "Anomalous":
                samples_with_affected_kpi_annotations += 1
                duration = anomalies.get("anomaly_duration")
                if isinstance(duration, Mapping):
                    start, end = duration.get("start"), duration.get("end")
                    if isinstance(start, int) and isinstance(end, int):
                        intervals.append((start, end, end - start + 1))
                        example["interval"] = (start, end)
                        example["selection_score"] = abs(end - start + 1 - 64)
                        current_example = examples.get("Anomalous")
                        if (
                            current_example is None
                            or example["selection_score"]
                            < current_example["selection_score"]
                        ):
                            examples["Anomalous"] = example
            valid_count += 1

    if correlation_blocks:
        block = np.concatenate(correlation_blocks, axis=0)
        correlation_sum += block.sum(axis=0)
        correlation_cross_sum += block.T @ block
        correlation_count += block.shape[0]

    invalid_report_count, ingestion_error_counts = _read_ingestion_errors(
        invalid_records_path
    )
    for category, count in ingestion_error_counts.items():
        quality_error_counts[f"ingestion:{category}"] += count

    numeric_statistics: dict[str, dict[str, float]] = {}
    sample_for_distributions = (
        np.concatenate(distribution_samples, axis=0)
        if distribution_samples
        else np.empty((0, len(NUMERIC_KPIS)))
    )
    for index, name in enumerate(NUMERIC_KPIS):
        mean = float(numeric_sums[index] / correlation_count) if correlation_count else 0.0
        variance = (
            max(0.0, float(numeric_squares[index] / correlation_count - mean * mean))
            if correlation_count
            else 0.0
        )
        sample_values = sample_for_distributions[:, index]
        quantiles = (
            np.quantile(sample_values, [0.25, 0.5, 0.75])
            if sample_values.size
            else np.zeros(3)
        )
        numeric_statistics[name] = {
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "minimum": float(numeric_minima[index]) if valid_count else 0.0,
            "q1_sampled": float(quantiles[0]),
            "median_sampled": float(quantiles[1]),
            "q3_sampled": float(quantiles[2]),
            "maximum": float(numeric_maxima[index]) if valid_count else 0.0,
        }

    if correlation_count:
        covariance = (
            correlation_cross_sum / correlation_count
            - np.outer(correlation_sum / correlation_count, correlation_sum / correlation_count)
        )
        standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard_deviations, standard_deviations)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
    else:
        correlation = np.zeros((len(NUMERIC_KPIS), len(NUMERIC_KPIS)))

    label_distribution = {
        label: int(label_counts.get(label, 0)) for label in TAXONOMY_LABELS
    }
    interval_array = np.asarray(intervals, dtype=np.int64).reshape((-1, 3))
    statistics = {
        "dataset_id": ingestion_report.get("dataset_id"),
        "dataset_revision": revision,
        "processed_samples_seen": row_count,
        "valid_samples": valid_count,
        "normal_samples": int(label_counts.get("Normal", 0)),
        "anomalous_samples": int(
            sum(count for label, count in label_counts.items() if label != "Normal")
        ),
        "source_file_count": len(source_counts),
        "label_distribution": label_distribution,
        "numeric_kpi_statistics": numeric_statistics,
        "numeric_kpi_correlation": {
            name: {
                other_name: float(correlation[i, j])
                for j, other_name in enumerate(NUMERIC_KPIS)
            }
            for i, name in enumerate(NUMERIC_KPIS)
        },
        "categorical_distributions": {
            name: {category: int(protocol_counts[name].get(category, 0)) for category in ("", "None", "TCP", "UDP")}
            for name in CATEGORICAL_KPIS
        },
        "source_file_distribution": {
            source: {
                "total": int(source_counts[source]),
                "normal": int(source_label_counts[source].get("Normal", 0)),
                "anomalous": int(
                    sum(
                        count
                        for label, count in source_label_counts[source].items()
                        if label != "Normal"
                    )
                ),
            }
            for source in sorted(source_counts)
        },
        "temporal_annotations": {
            "samples_with_intervals": int(len(intervals)),
            "samples_with_affected_kpi_annotations": int(
                samples_with_affected_kpi_annotations
            ),
            "start_mean": float(interval_array[:, 0].mean()) if intervals else None,
            "end_mean": float(interval_array[:, 1].mean()) if intervals else None,
            "duration_mean": float(interval_array[:, 2].mean()) if intervals else None,
            "duration_median": float(np.median(interval_array[:, 2])) if intervals else None,
        },
        "distribution_sampling": {
            "method": "take every 16th timestep from each valid sample",
            "sampled_timestep_count": int(sample_for_distributions.shape[0]),
        },
        "figure_files": [
            "01_label_distribution.png",
            "02_anomaly_categories.png",
            "03_numeric_kpi_distributions.png",
            "04_numeric_kpi_correlations.png",
            "05_selected_time_series.png",
            "06_normal_anomaly_kpi_comparison.png",
            "07_source_file_distribution.png",
            "08_temporal_anomaly_intervals.png",
            "09_missing_value_analysis.png",
            "10_categorical_distributions.png",
        ],
    }
    quality = {
        "dataset_revision": revision,
        "processed_samples_seen": row_count,
        "valid_samples": valid_count,
        "invalid_processed_samples": mismatch_count,
        "duplicate_sample_ids": int(quality_error_counts.get("duplicate_sample_id", 0)),
        "sequence_metadata_row_count_mismatches": int(
            quality_error_counts.get("sequence_metadata_row_count_mismatch", 0)
        ),
        "ingestion_invalid_samples": invalid_report_count,
        "source_file_count": len(source_counts),
        "matches_ingestion_sample_count": (
            valid_count == ingestion_report.get("loaded_samples")
        ),
        "matches_ingestion_source_file_count": (
            len(source_counts) == ingestion_report.get("source_file_count")
        ),
        "sequence_shape": [TIME_STEPS, len(KPI_NAMES)],
        "expected_kpi_names": list(KPI_NAMES),
        "missing_values_by_kpi": {
            name: int(missing_counts.get(name, 0)) for name in KPI_NAMES
        },
        "non_finite_values_by_numeric_kpi": {
            name: int(nonfinite_counts.get(name, 0)) for name in NUMERIC_KPIS
        },
        "validation_error_counts": dict(sorted(quality_error_counts.items())),
        "issues_truncated": max(0, mismatch_count - len(quality_issues)),
        "sample_issues": quality_issues,
    }

    _write_json(report_directory / "eda" / "dataset_statistics.json", statistics)
    _write_json(report_directory / "eda" / "data_quality.json", quality)
    write_preprocessing_configuration(
        report_directory / "eda" / "preprocessing_config.json",
        revision=revision,
        sample_count=valid_count,
    )
    _create_figures(
        figures_directory,
        label_distribution=label_distribution,
        numeric_sample=sample_for_distributions,
        correlation=correlation,
        examples=examples,
        per_sample_medians=per_sample_medians,
        source_label_counts=source_label_counts,
        intervals=interval_array,
        missing_counts=missing_counts,
        protocol_counts=protocol_counts,
    )
    _LOGGER.info(
        "TelecomTS EDA complete: %s valid, %s invalid processed samples; reports in %s",
        valid_count,
        mismatch_count,
        report_directory / "eda",
    )
    return {"statistics": statistics, "quality": quality}


def _read_ingestion_errors(path: Path) -> tuple[int, Counter[str]]:
    count = 0
    errors: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors["malformed_error_report_row"] += 1
                continue
            if not isinstance(row, dict):
                errors["malformed_error_report_row"] += 1
                continue
            for message in row.get("errors", []):
                errors[str(message).split(":", 1)[0]] += 1
    return count, errors


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _create_figures(
    directory: Path,
    *,
    label_distribution: Mapping[str, int],
    numeric_sample: np.ndarray,
    correlation: np.ndarray,
    examples: Mapping[str, Mapping[str, Any]],
    per_sample_medians: Mapping[str, list[np.ndarray]],
    source_label_counts: Mapping[str, Counter[str]],
    intervals: np.ndarray,
    missing_counts: Mapping[str, int],
    protocol_counts: Mapping[str, Counter[str]],
) -> None:
    labels = list(label_distribution)
    values = [label_distribution[label] for label in labels]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(["Normal", "Anomalous"], [label_distribution["Normal"], sum(values[1:])])
    axis.set(title="Normal versus anomalous samples", ylabel="Samples")
    _save_figure(fig, directory / "01_label_distribution.png")

    anomaly_labels = [label for label in labels if label != "Normal"]
    anomaly_values = [label_distribution[label] for label in anomaly_labels]
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.barh(anomaly_labels[::-1], anomaly_values[::-1])
    axis.set(title="Anomaly category distribution", xlabel="Samples")
    fig.tight_layout()
    _save_figure(fig, directory / "02_anomaly_categories.png")

    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    for index, (axis, name) in enumerate(zip(axes.flat, NUMERIC_KPIS)):
        if numeric_sample.size:
            axis.hist(numeric_sample[:, index], bins=40, color="#4472C4", alpha=0.85)
        axis.set_title(name)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    fig.suptitle("Numerical KPI distributions (deterministic timestep sample)")
    fig.tight_layout()
    _save_figure(fig, directory / "03_numeric_kpi_distributions.png")

    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(NUMERIC_KPIS)), NUMERIC_KPIS, rotation=90)
    axis.set_yticks(range(len(NUMERIC_KPIS)), NUMERIC_KPIS)
    axis.set_title("Numerical KPI Pearson correlation")
    fig.colorbar(image, ax=axis, label="Correlation")
    fig.tight_layout()
    _save_figure(fig, directory / "04_numeric_kpi_correlations.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, name in zip(axes.flat, _SELECTED_KPIS):
        for kind, color in (("Normal", "#4472C4"), ("Anomalous", "#C0504D")):
            example = examples.get(kind)
            if example:
                axis.plot(
                    range(TIME_STEPS),
                    example["kpis"][name],
                    label=f"{kind}: {example['label']}",
                    color=color,
                    linewidth=1.1,
                )
                if kind == "Anomalous" and example["interval"] is not None:
                    axis.axvspan(*example["interval"], color=color, alpha=0.15)
        axis.set_title(name)
        axis.set_xlabel("Time step")
        if axis.lines:
            axis.legend(fontsize="small")
    fig.suptitle("Representative normal/anomalous sequences")
    fig.tight_layout()
    _save_figure(fig, directory / "05_selected_time_series.png")

    medians = {
        kind: (
            np.asarray(rows, dtype=np.float64)
            if rows
            else np.empty((0, len(NUMERIC_KPIS)))
        )
        for kind, rows in per_sample_medians.items()
    }
    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    for index, (axis, name) in enumerate(zip(axes.flat, NUMERIC_KPIS)):
        groups = [
            medians[kind][:, index]
            for kind in ("Normal", "Anomalous")
            if medians[kind].shape[0]
        ]
        if groups:
            axis.boxplot(
                groups,
                tick_labels=[
                    kind
                    for kind in ("Normal", "Anomalous")
                    if medians[kind].shape[0]
                ],
                showfliers=False,
            )
        axis.set_title(name)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    fig.suptitle("Per-sample KPI medians: normal versus anomaly")
    fig.tight_layout()
    _save_figure(fig, directory / "06_normal_anomaly_kpi_comparison.png")

    sources = sorted(source_label_counts)
    source_normal = [source_label_counts[source].get("Normal", 0) for source in sources]
    source_anomalous = [
        sum(count for label, count in source_label_counts[source].items() if label != "Normal")
        for source in sources
    ]
    fig, axis = plt.subplots(figsize=(12, max(8, len(sources) * 0.28)))
    y = np.arange(len(sources))
    axis.barh(y, source_normal, label="Normal")
    axis.barh(y, source_anomalous, left=source_normal, label="Anomalous")
    axis.set_yticks(
        y,
        [source.replace("/processed/chunked.jsonl", "") for source in sources],
        fontsize=6,
    )
    axis.set(title="Sample distribution by source file", xlabel="Samples")
    axis.legend()
    fig.tight_layout()
    _save_figure(fig, directory / "07_source_file_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if intervals.size:
        axes[0].hist(intervals[:, 0], bins=np.arange(0, TIME_STEPS + 1, 8))
        axes[1].hist(intervals[:, 1], bins=np.arange(0, TIME_STEPS + 1, 8))
        axes[2].hist(intervals[:, 2], bins=20)
    for axis, title in zip(axes, ("Start timestep", "End timestep", "Duration (steps)")):
        axis.set(title=title, ylabel="Samples")
    fig.suptitle("Temporal anomaly interval distribution (inclusive bounds)")
    fig.tight_layout()
    _save_figure(fig, directory / "08_temporal_anomaly_intervals.png")

    fig, axis = plt.subplots(figsize=(12, 5))
    missing_values = [missing_counts.get(name, 0) for name in KPI_NAMES]
    axis.bar(KPI_NAMES, missing_values)
    axis.set(title="Missing values in processed sequences", ylabel="Missing entries")
    axis.tick_params(axis="x", rotation=75)
    if not any(missing_values):
        axis.set_ylim(0, 1)
        axis.text(
            0.5,
            0.5,
            "No missing KPI values detected",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    fig.tight_layout()
    _save_figure(fig, directory / "09_missing_value_analysis.png")

    fig, axes = plt.subplots(1, len(CATEGORICAL_KPIS), figsize=(12, 5))
    for axis, name in zip(axes, CATEGORICAL_KPIS):
        counts = protocol_counts[name]
        display_names = [
            "(empty string)" if category == "" else category
            for category in PROTOCOL_CATEGORIES[:-1]
        ]
        axis.bar(
            display_names,
            [counts.get(category, 0) for category in PROTOCOL_CATEGORIES[:-1]],
        )
        axis.set(title=name, ylabel="Values")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Protocol categorical KPI distributions")
    fig.tight_layout()
    _save_figure(fig, directory / "10_categorical_distributions.png")


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TelecomTS EDA reports from a completed ingestion."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_eda(args.project_root, args.revision)


if __name__ == "__main__":
    main()
