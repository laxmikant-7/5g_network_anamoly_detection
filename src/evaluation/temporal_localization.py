"""Train and evaluate group-aware TelecomTS temporal interval localization."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cl5g.common.exceptions import ProjectError, SchemaValidationError
from cl5g.data.telecomts import (
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TAXONOMY_LABELS,
    TIME_STEPS,
)
from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor
from cl5g.models.temporal_localization import (
    TemporalIntervalLocalizer,
    interval_from_scores,
    temporal_context_features,
)
from cl5g.pipeline.config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)
_PARTITIONS = ("train", "validation", "test")
_THRESHOLDS = tuple(round(float(value), 2) for value in np.linspace(0.10, 0.90, 41))
_CONTEXT_RADIUS = 2
_RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 80,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": 0.8,
    "class_weight": "balanced_subsample",
}


def run_temporal_localization(
    config: AppConfig,
    *,
    revision: str = DEFAULT_REVISION,
    split_directory: Path | None = None,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    """Fit on training groups, tune on validation groups, evaluate test once."""
    split_root = split_directory or (
        config.resolve(config.paths.processed)
        / "telecomts"
        / revision
        / "splits"
        / f"seed-{config.seed}"
    )
    reports = report_directory or (
        config.project_root
        / "data"
        / "reports"
        / "telecomts"
        / revision
        / "temporal_localization"
        / f"seed-{config.seed}"
    )
    state_path = reports / "test_evaluation_state.json"
    if state_path.exists():
        raise ProjectError(
            "Temporal localization test evaluation has already started for this "
            "revision and seed; refusing to reuse the final test partition."
        )

    preprocessor = TelecomTSPreprocessor()
    train_records = _load_partition(split_root, "train", preprocessor)
    validation_records = _load_partition(split_root, "validation", preprocessor)
    _validate_group_splits(train_records, validation_records)
    training_anomalies = [row for row in train_records if row["label"] != "Normal"]
    validation_anomalies = [
        row for row in validation_records if row["label"] != "Normal"
    ]
    if not training_anomalies or not validation_anomalies:
        raise SchemaValidationError(
            ["Training and validation partitions must contain annotated anomalies."]
        )

    train_features, train_targets = _training_matrix(
        training_anomalies,
        preprocessor,
        context_radius=_CONTEXT_RADIUS,
    )
    estimator = RandomForestClassifier(
        **_RANDOM_FOREST_PARAMETERS,
        random_state=config.seed,
        n_jobs=-1,
    )
    estimator.fit(train_features, train_targets)
    validation_scores = _record_scores(
        estimator,
        validation_anomalies,
        preprocessor,
        context_radius=_CONTEXT_RADIUS,
    )
    validation_targets = [row["interval"] for row in validation_anomalies]
    threshold_metrics = [
        (
            threshold,
            _interval_metrics(
                validation_targets,
                [
                    interval_from_scores(scores, threshold)
                    for scores in validation_scores
                ],
            ),
        )
        for threshold in _THRESHOLDS
    ]
    selected_threshold, validation_metrics = max(
        threshold_metrics,
        key=lambda item: (
            item[1]["mean_temporal_iou"],
            item[1]["timestep_f1"],
            item[1]["event_recall_iou_0_5"],
            -item[0],
        ),
    )
    validation_baseline = _interval_metrics(
        validation_targets,
        [(0, TIME_STEPS - 1)] * len(validation_anomalies),
    )
    localizer = TemporalIntervalLocalizer(
        estimator,
        threshold=selected_threshold,
        context_radius=_CONTEXT_RADIUS,
    )
    artifact_path = config.resolve(
        config.inference.temporal_localization_artifact
        or "artifacts/temporal_localization.joblib"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(localizer, artifact_path)

    reports.mkdir(parents=True, exist_ok=True)
    write_json(
        reports / "model_configuration.json",
        {
            "dataset_revision": revision,
            "seed": config.seed,
            "model": "RandomForestClassifier",
            "model_kind": "experimental supervised per-timestep interval classifier",
            "random_forest_parameters": _RANDOM_FOREST_PARAMETERS,
            "context_radius": _CONTEXT_RADIUS,
            "context_window_steps": 2 * _CONTEXT_RADIUS + 1,
            "processed_channels": 26,
            "feature_order": "centered time context of stateless TelecomTS encoded KPI channels",
            "selected_threshold": selected_threshold,
            "threshold_selection": "maximum mean validation temporal IoU; timestep F1 then recall tie-break",
            "sampling_rate_hz": "provided per analysis request; defaults to TelecomTS 10 Hz",
            "affected_kpi_prediction": "not emitted; attribution labels are not treated as model output",
        },
    )
    _write_threshold_plot(threshold_metrics, validation_baseline, reports)
    write_json(
        state_path,
        {
            "status": "started",
            "dataset_revision": revision,
            "seed": config.seed,
            "test_partition_loaded": False,
            "test_partition_historical": True,
            "reason": (
                "The source-file test partition has already been used for binary "
                "detection and diagnosis evaluation in this workspace."
            ),
        },
    )

    test_records = _load_partition(split_root, "test", preprocessor)
    _validate_group_splits(train_records, validation_records, test_records)
    test_anomalies = [row for row in test_records if row["label"] != "Normal"]
    test_scores = _record_scores(
        estimator,
        test_anomalies,
        preprocessor,
        context_radius=_CONTEXT_RADIUS,
    )
    test_predictions = [
        interval_from_scores(scores, selected_threshold) for scores in test_scores
    ]
    test_metrics = _interval_metrics(
        [row["interval"] for row in test_anomalies],
        test_predictions,
    )
    test_baseline = _interval_metrics(
        [row["interval"] for row in test_anomalies],
        [(0, TIME_STEPS - 1)] * len(test_anomalies),
    )
    source_overlap = _source_overlap(
        train_records,
        validation_records,
        test_records,
    )
    annotation_audit = _annotation_audit(
        (*train_records, *validation_records, *test_records)
    )
    interval_lengths = annotation_audit["interval_length_steps"]
    limitations = [
        (
            f"{interval_lengths['full_sequence_count']} of "
            f"{annotation_audit['anomalous_samples_with_valid_intervals']} "
            "anomalous annotations cover the full 128-step sequence and "
            "provide no within-sequence boundary information."
        ),
        "The annotation boundaries are source-provided labels and have not been independently verified against raw event timing.",
        "Affected KPI labels are available for training records but this interval model does not predict affected KPIs; inference returns null rather than infer them from attribution scores.",
        "Localization scores are uncalibrated timestep probabilities, not confidence intervals.",
        "The test partition is group-disjoint but historical: it was already used for detection and diagnosis evaluation.",
    ]
    if test_metrics["mean_temporal_iou"] <= test_baseline["mean_temporal_iou"]:
        limitations.append(
            "On the historical group-held-out test partition, supervised mean "
            "temporal IoU did not exceed the full-window baseline; treat the "
            "supervised output as experimental."
        )
    report = {
        "dataset_revision": revision,
        "seed": config.seed,
        "sampling_rate_hz": annotation_audit["sampling_rates_hz"],
        "seconds_per_timestep": (
            1.0 / annotation_audit["sampling_rates_hz"][0]
            if len(annotation_audit["sampling_rates_hz"]) == 1
            else None
        ),
        "annotation_audit": annotation_audit,
        "partition_summary": {
            name: _partition_summary(records)
            for name, records in (
                ("train", train_records),
                ("validation", validation_records),
                ("test", test_records),
            )
        },
        "source_file_overlap": source_overlap,
        "training": {
            "annotated_anomaly_samples": len(training_anomalies),
            "timestep_count": int(len(train_targets)),
            "positive_timestep_fraction": float(np.mean(train_targets)),
            "source_file_count": len({row["source_file_id"] for row in training_anomalies}),
        },
        "model_selection": {
            "validation_anomaly_samples": len(validation_anomalies),
            "selected_threshold": selected_threshold,
            "selected_model_metrics": validation_metrics,
            "full_window_baseline_metrics": validation_baseline,
            "threshold_selection_data": "validation only",
        },
        "test_evaluation": {
            "anomaly_samples": len(test_anomalies),
            "selected_model_metrics": test_metrics,
            "full_window_baseline_metrics": test_baseline,
            "historical_test_partition": True,
            "test_partition_used_for_selection": False,
        },
        "limitations": limitations,
        "artifacts": {
            "localizer": str(artifact_path),
            "report_directory": str(reports),
        },
    }
    write_json(reports / "temporal_localization_report.json", report)
    _write_metrics_plot(validation_metrics, validation_baseline, test_metrics, test_baseline, reports)
    _write_sample_plot(test_anomalies, test_predictions, reports)
    write_json(
        state_path,
        {
            "status": "completed",
            "dataset_revision": revision,
            "seed": config.seed,
            "test_partition_loaded": True,
            "test_partition_historical": True,
            "selected_threshold": selected_threshold,
            "test_anomaly_samples": len(test_anomalies),
        },
    )
    LOGGER.info(
        "Temporal localization complete: validation IoU %.4f, test IoU %.4f "
        "(full-window baselines %.4f / %.4f).",
        validation_metrics["mean_temporal_iou"],
        test_metrics["mean_temporal_iou"],
        validation_baseline["mean_temporal_iou"],
        test_baseline["mean_temporal_iou"],
    )
    return report


def _load_partition(
    directory: Path,
    name: str,
    preprocessor: TelecomTSPreprocessor,
) -> list[dict[str, Any]]:
    sequence_path = directory / f"{name}_sequences.jsonl.gz"
    metadata_path = directory / f"{name}_metadata.jsonl.gz"
    if not sequence_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Required {name} partition files are missing.")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with (
            gzip.open(sequence_path, "rt", encoding="utf-8") as sequence_stream,
            gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
        ):
            for line_number, (sequence_line, metadata_line) in enumerate(
                zip_longest(sequence_stream, metadata_stream),
                start=1,
            ):
                if sequence_line is None or metadata_line is None:
                    raise SchemaValidationError(
                        [f"{name} sequence and metadata counts differ at row {line_number}."]
                    )
                try:
                    sequence_row = json.loads(sequence_line)
                    metadata = json.loads(metadata_line)
                except json.JSONDecodeError as exc:
                    raise SchemaValidationError(
                        [f"Malformed {name} JSON at row {line_number}: {exc}."]
                    ) from exc
                sample_id = sequence_row.get("sample_id")
                if (
                    not isinstance(sample_id, str)
                    or not sample_id
                    or metadata.get("sample_id") != sample_id
                    or sample_id in seen
                ):
                    raise SchemaValidationError(
                        [f"Invalid or duplicate sample ID in {name} row {line_number}."]
                    )
                label = metadata.get("label")
                if label not in TAXONOMY_LABELS:
                    raise SchemaValidationError(
                        [f"Invalid anomaly label in {name} row {line_number}."]
                    )
                source_id = metadata.get("source_file_id")
                kpis = sequence_row.get("kpis")
                if not isinstance(source_id, str) or not source_id:
                    raise SchemaValidationError(
                        [f"Missing source-file ID in {name} row {line_number}."]
                    )
                if not isinstance(kpis, dict) or set(kpis) != set(KPI_NAMES):
                    raise SchemaValidationError(
                        [f"Invalid KPI channels in {name} row {line_number}."]
                    )
                try:
                    preprocessor.transform_sample(kpis)
                except (TypeError, ValueError) as exc:
                    raise SchemaValidationError(
                        [f"Invalid KPI sequence in {name} row {line_number}: {exc}."]
                    ) from exc
                anomalies = metadata.get("anomalies")
                if not isinstance(anomalies, dict) or anomalies.get("exists") != (
                    label != "Normal"
                ):
                    raise SchemaValidationError(
                        [f"Anomaly label/annotation disagreement in {name} row {line_number}."]
                    )
                duration = anomalies.get("anomaly_duration")
                affected = anomalies.get("affected_kpis")
                interval: tuple[int, int] | None = None
                if label == "Normal":
                    if not (
                        isinstance(duration, dict)
                        and duration.get("start") is None
                        and duration.get("end") is None
                        and affected == [""]
                    ):
                        raise SchemaValidationError(
                            [f"Normal record has non-empty annotations in {name} row {line_number}."]
                        )
                else:
                    if not isinstance(duration, dict):
                        raise SchemaValidationError(
                            [f"Missing interval annotation in {name} row {line_number}."]
                        )
                    start, end = duration.get("start"), duration.get("end")
                    if (
                        isinstance(start, bool)
                        or not isinstance(start, int)
                        or isinstance(end, bool)
                        or not isinstance(end, int)
                        or start < 0
                        or end < start
                        or end >= TIME_STEPS
                    ):
                        raise SchemaValidationError(
                            [f"Invalid inclusive interval in {name} row {line_number}."]
                        )
                    if (
                        not isinstance(affected, list)
                        or not affected
                        or any(name not in NUMERIC_KPIS for name in affected)
                    ):
                        raise SchemaValidationError(
                            [f"Invalid affected KPI annotation in {name} row {line_number}."]
                        )
                    interval = (start, end)
                sample_rate = metadata.get("sampling_rate")
                if (
                    isinstance(sample_rate, bool)
                    or not isinstance(sample_rate, (int, float))
                    or not np.isfinite(sample_rate)
                    or sample_rate <= 0
                ):
                    raise SchemaValidationError(
                        [f"Invalid sampling rate in {name} row {line_number}."]
                    )
                seen.add(sample_id)
                records.append(
                    {
                        "sample_id": sample_id,
                        "source_file_id": source_id,
                        "label": label,
                        "sequence": kpis,
                        "interval": interval,
                        "affected_kpis": affected,
                        "sampling_rate_hz": float(sample_rate),
                    }
                )
    except (OSError, EOFError, UnicodeError, gzip.BadGzipFile) as exc:
        raise SchemaValidationError([f"Could not read the {name} partition: {exc}."]) from exc
    if not records:
        raise SchemaValidationError([f"The {name} partition is empty."])
    return records


def _training_matrix(
    records: list[dict[str, Any]],
    preprocessor: TelecomTSPreprocessor,
    *,
    context_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for record in records:
        encoded = preprocessor.transform_sample(record["sequence"])
        feature_rows.append(
            temporal_context_features(
                encoded,
                context_radius=context_radius,
            ).astype(np.float32)
        )
        target = np.zeros(TIME_STEPS, dtype=np.uint8)
        start, end = record["interval"]
        target[start : end + 1] = 1
        target_rows.append(target)
    return np.concatenate(feature_rows), np.concatenate(target_rows)


def _record_scores(
    estimator: object,
    records: list[dict[str, Any]],
    preprocessor: TelecomTSPreprocessor,
    *,
    context_radius: int,
) -> list[np.ndarray]:
    classes = getattr(estimator, "classes_", None)
    predict_proba = getattr(estimator, "predict_proba", None)
    if classes is None or not callable(predict_proba):
        raise ProjectError("The fitted timestep model cannot return class probabilities.")
    positive = [index for index, value in enumerate(classes) if value == 1]
    if len(positive) != 1:
        raise ProjectError("The fitted timestep model has no unique positive class.")
    scores: list[np.ndarray] = []
    for record in records:
        encoded = preprocessor.transform_sample(record["sequence"])
        features = temporal_context_features(
            encoded,
            context_radius=context_radius,
        ).astype(np.float32)
        probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        score = probabilities[:, positive[0]]
        if score.shape != (TIME_STEPS,) or not np.isfinite(score).all():
            raise ProjectError("The timestep model returned invalid validation scores.")
        scores.append(score)
    return scores


def _interval_metrics(
    targets: list[tuple[int, int]],
    predictions: list[tuple[int, int] | None],
) -> dict[str, float | int]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Targets and predictions must be non-empty and aligned.")
    ious: list[float] = []
    boundary_start_errors: list[int] = []
    boundary_end_errors: list[int] = []
    true_positive = false_positive = false_negative = 0
    event_matches = predicted_events = 0
    for (true_start, true_end), prediction in zip(targets, predictions, strict=True):
        true_mask = np.zeros(TIME_STEPS, dtype=bool)
        true_mask[true_start : true_end + 1] = True
        predicted_mask = np.zeros(TIME_STEPS, dtype=bool)
        if prediction is not None:
            pred_start, pred_end = prediction
            if pred_start < 0 or pred_end < pred_start or pred_end >= TIME_STEPS:
                raise ValueError("Predicted interval is outside the 128-step sequence.")
            predicted_mask[pred_start : pred_end + 1] = True
            predicted_events += 1
            boundary_start_errors.append(abs(pred_start - true_start))
            boundary_end_errors.append(abs(pred_end - true_end))
        else:
            boundary_start_errors.append(TIME_STEPS)
            boundary_end_errors.append(TIME_STEPS)
        intersection = int(np.logical_and(true_mask, predicted_mask).sum())
        union = int(np.logical_or(true_mask, predicted_mask).sum())
        iou = intersection / union if union else 0.0
        ious.append(iou)
        event_matches += int(iou >= 0.5)
        true_positive += intersection
        false_positive += int(np.logical_and(predicted_mask, ~true_mask).sum())
        false_negative += int(np.logical_and(~predicted_mask, true_mask).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": len(targets),
        "mean_temporal_iou": float(np.mean(ious)),
        "median_temporal_iou": float(np.median(ious)),
        "event_precision_iou_0_5": (
            float(event_matches / predicted_events) if predicted_events else 0.0
        ),
        "event_recall_iou_0_5": float(event_matches / len(targets)),
        "boundary_start_mae_steps": float(np.mean(boundary_start_errors)),
        "boundary_end_mae_steps": float(np.mean(boundary_end_errors)),
        "boundary_mae_steps": float(
            (np.mean(boundary_start_errors) + np.mean(boundary_end_errors)) / 2
        ),
        "timestep_precision": float(precision),
        "timestep_recall": float(recall),
        "timestep_f1": float(f1),
        "predicted_interval_count": predicted_events,
        "missing_interval_count": len(targets) - predicted_events,
    }


def _validate_group_splits(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]] | None = None,
) -> None:
    partitions = {"train": train, "validation": validation}
    if test is not None:
        partitions["test"] = test
    group_sets = {
        name: {row["source_file_id"] for row in records}
        for name, records in partitions.items()
    }
    names = list(group_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = group_sets[first] & group_sets[second]
            if overlap:
                raise SchemaValidationError(
                    [f"Source-file overlap between {first} and {second}: {sorted(overlap)}."]
                )
    identifiers = [
        row["sample_id"]
        for records in partitions.values()
        for row in records
    ]
    if len(identifiers) != len(set(identifiers)):
        raise SchemaValidationError(["Sample IDs overlap across localization partitions."])


def _source_overlap(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, list[str]]:
    groups = {
        "train": {row["source_file_id"] for row in train},
        "validation": {row["source_file_id"] for row in validation},
        "test": {row["source_file_id"] for row in test},
    }
    return {
        f"{left}_and_{right}": sorted(groups[left] & groups[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }


def _partition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in records)
    return {
        "sample_count": len(records),
        "source_file_count": len({row["source_file_id"] for row in records}),
        "anomaly_sample_count": sum(count for label, count in labels.items() if label != "Normal"),
        "normal_sample_count": labels.get("Normal", 0),
        "label_distribution": dict(sorted(labels.items())),
    }


def _annotation_audit(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    anomalies = [row for row in records if row["label"] != "Normal"]
    normal = [row for row in records if row["label"] == "Normal"]
    lengths = [row["interval"][1] - row["interval"][0] + 1 for row in anomalies]
    affected = Counter(
        kpi
        for row in anomalies
        for kpi in row["affected_kpis"]
    )
    return {
        "sample_specific_anomaly_intervals": True,
        "normal_samples_with_valid_empty_interval_annotations": len(normal),
        "anomalous_samples_with_valid_intervals": len(anomalies),
        "missing_or_invalid_intervals": 0,
        "anomalous_samples_with_affected_kpi_annotations": sum(
            bool(row["affected_kpis"]) for row in anomalies
        ),
        "unique_affected_kpis": len(affected),
        "affected_kpi_sample_counts": dict(sorted(affected.items())),
        "sampling_rates_hz": sorted({row["sampling_rate_hz"] for row in records}),
        "interval_length_steps": {
            "minimum": min(lengths),
            "median": float(np.median(lengths)),
            "maximum": max(lengths),
            "full_sequence_count": sum(length == TIME_STEPS for length in lengths),
            "partial_interval_count": sum(length < TIME_STEPS for length in lengths),
        },
    }


def _write_threshold_plot(
    metrics: list[tuple[float, dict[str, float | int]]],
    baseline: dict[str, float | int],
    directory: Path,
) -> None:
    thresholds = [item[0] for item in metrics]
    ious = [item[1]["mean_temporal_iou"] for item in metrics]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, ious, marker="o", label="Supervised timestep model")
    axis.axhline(
        baseline["mean_temporal_iou"],
        linestyle="--",
        color="#777777",
        label="Full-window baseline",
    )
    axis.set(
        xlabel="Timestep probability threshold",
        ylabel="Mean temporal IoU",
        title="Validation-only localization threshold selection",
        ylim=(0.0, 1.0),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "validation_thresholds.png", dpi=160)
    plt.close(figure)
    with (directory / "validation_thresholds.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        stream.write("threshold,mean_temporal_iou,timestep_f1,event_recall_iou_0_5\n")
        for threshold, values in metrics:
            stream.write(
                f"{threshold:.2f},{values['mean_temporal_iou']:.8f},"
                f"{values['timestep_f1']:.8f},{values['event_recall_iou_0_5']:.8f}\n"
            )


def _write_metrics_plot(
    validation: dict[str, float | int],
    validation_baseline: dict[str, float | int],
    test: dict[str, float | int],
    test_baseline: dict[str, float | int],
    directory: Path,
) -> None:
    labels = ("Validation", "Test")
    learned = [
        validation["mean_temporal_iou"],
        test["mean_temporal_iou"],
    ]
    baseline = [
        validation_baseline["mean_temporal_iou"],
        test_baseline["mean_temporal_iou"],
    ]
    locations = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(locations - 0.18, learned, width=0.36, label="Supervised timestep model")
    axis.bar(locations + 0.18, baseline, width=0.36, label="Full-window baseline")
    axis.set(
        xticks=locations,
        xticklabels=labels,
        ylabel="Mean temporal IoU",
        title="Localization interval overlap",
        ylim=(0.0, 1.0),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "interval_metrics.png", dpi=160)
    plt.close(figure)


def _write_sample_plot(
    records: list[dict[str, Any]],
    predictions: list[tuple[int, int] | None],
    directory: Path,
) -> None:
    index = next(
        (
            position
            for position, row in enumerate(records)
            if predictions[position] is not None
            and row["interval"] != (0, TIME_STEPS - 1)
        ),
        0,
    )
    record = records[index]
    kpi = record["affected_kpis"][0]
    values = np.asarray(record["sequence"][kpi], dtype=np.float64)
    true_start, true_end = record["interval"]
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(np.arange(TIME_STEPS), values, color="#345995", label=kpi)
    axis.axvspan(true_start, true_end, color="#3a9d5d", alpha=0.2, label="Annotated interval")
    prediction = predictions[index]
    if prediction is not None:
        axis.axvspan(
            prediction[0],
            prediction[1],
            color="#d1495b",
            alpha=0.18,
            label="Predicted interval",
        )
    axis.set(
        xlabel="Time step (10 Hz; 0.1 s per step)",
        ylabel=kpi,
        title=f"Test sample interval comparison · {record['sample_id']}",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "sample_test_localization.png", dpi=160)
    plt.close(figure)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.yaml.")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split-directory", type=Path)
    parser.add_argument("--report-directory", type=Path)
    args = parser.parse_args()
    try:
        report = run_temporal_localization(
            load_config(args.config),
            revision=args.revision,
            split_directory=args.split_directory,
            report_directory=args.report_directory,
        )
    except (ProjectError, OSError, ValueError) as exc:
        parser.exit(1, f"Temporal localization failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
