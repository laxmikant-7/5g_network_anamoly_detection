"""Fit, select, and evaluate a binary TelecomTS anomaly detector."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from cl5g.common.exceptions import ProjectError, SchemaValidationError
from cl5g.data.telecomts import (
    DEFAULT_REVISION,
    KPI_NAMES,
    LABEL_MAPPING,
    TAXONOMY_LABELS,
)
from cl5g.data.telecomts_preprocessing import (
    TelecomTSPreprocessor,
    preprocessing_configuration,
)
from cl5g.evaluation.binary import (
    evaluate_binary,
    write_json,
    write_test_figures,
    write_threshold_comparison,
)
from cl5g.features.builder import build_features
from cl5g.features.extraction import build_feature_configuration
from cl5g.models.detection import (
    DetectionCandidate,
    build_detection_candidates,
    threshold_comparison,
)
from cl5g.pipeline.config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)
FEATURE_SET = "base_plus_temporal"
_PARTITION_FILES = {
    name: (f"{name}_sequences.jsonl.gz", f"{name}_metadata.jsonl.gz")
    for name in ("train", "validation", "test")
}


@dataclass(frozen=True)
class _Partition:
    features: pd.DataFrame
    binary_labels: np.ndarray
    labels: tuple[str, ...]
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


def run_training(
    config: AppConfig,
    *,
    revision: str = DEFAULT_REVISION,
    split_directory: Path | None = None,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    """Train candidates, select on validation, refit, then evaluate test once."""
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
        / "binary_detection"
        / f"seed-{config.seed}"
    )
    test_state_path = reports / "test_evaluation_state.json"
    if test_state_path.exists():
        raise ProjectError(
            "Final test evaluation has already been started for this run "
            f"({test_state_path}). Do not reuse the final test partition."
        )

    split_report = _load_split_report(split_root / "split_metadata.json", config.seed)
    if not all(
        not groups
        for groups in split_report.get("source_overlap", {}).values()
    ):
        raise SchemaValidationError(
            ["Split metadata reports source-file overlap between partitions."]
        )
    pristine = split_report.get("historical_test_data", {}).get("test_set_pristine")

    feature_configuration = build_feature_configuration(FEATURE_SET)
    preprocessor = TelecomTSPreprocessor()
    train = _load_partition(split_root, "train", feature_configuration, preprocessor)
    validation = _load_partition(
        split_root,
        "validation",
        feature_configuration,
        preprocessor,
    )
    _validate_partition_separation(train, validation)
    if set(np.unique(train.binary_labels)) != {0, 1}:
        raise SchemaValidationError(
            ["Training partition must contain normal and anomalous samples."]
        )
    if set(np.unique(validation.binary_labels)) != {0, 1}:
        raise SchemaValidationError(
            ["Validation partition must contain normal and anomalous samples."]
        )

    candidate_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in build_detection_candidates(train.binary_labels, seed=config.seed):
        LOGGER.info("Fitting validation candidate %s", candidate.name)
        candidate.estimator.fit(train.features, train.binary_labels)
        validation_probability = _positive_probability(
            candidate.estimator,
            validation.features,
        )
        selected_threshold, threshold_rows = threshold_comparison(
            validation.binary_labels,
            validation_probability,
        )
        validation_metrics = evaluate_binary(
            validation.binary_labels,
            validation_probability,
            selected_threshold,
        )
        result = {
            "name": candidate.name,
            "configuration": candidate.configuration,
            "threshold": selected_threshold,
            "validation_metrics": validation_metrics,
            "_selection_key": (
                -validation_metrics["f2"],
                -validation_metrics["recall"],
                -validation_metrics["average_precision"],
                -validation_metrics["precision"],
                validation_metrics["false_positive_rate"],
                candidate.name,
            ),
        }
        candidate_results.append(result)
        if selected is None or result["_selection_key"] < selected["_selection_key"]:
            selected = {
                **result,
                "candidate": candidate,
                "threshold_rows": threshold_rows,
            }

    if selected is None:
        raise ProjectError("No binary detection candidates were available.")

    selected_candidate: DetectionCandidate = selected["candidate"]
    threshold = float(selected["threshold"])
    final_estimator = clone(selected_candidate.estimator)
    final_estimator.fit(train.features, train.binary_labels)

    artifact_path = config.resolve(config.inference.detection_artifact)
    preprocessing_path = config.resolve(config.inference.preprocessing_artifact)
    feature_path = config.resolve(config.inference.feature_config_artifact)
    metadata_path = config.resolve(config.inference.metadata_artifact)
    for path in (artifact_path, preprocessing_path, feature_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    model_version = f"telecomts-binary-{revision[:8]}-seed-{config.seed}"
    model_configuration = {
        "model_version": model_version,
        "selected_candidate": selected_candidate.name,
        "estimator": selected_candidate.configuration,
        "selection_metric": "validation F2",
        "tie_break_order": [
            "validation recall (higher)",
            "validation average precision (higher)",
            "validation precision (higher)",
            "validation false-positive rate (lower)",
            "candidate name (ascending)",
        ],
        "decision_threshold": threshold,
        "threshold_selection": {
            "dataset": "validation",
            "objective": "maximum F2",
            "decision_rule": "anomaly_probability >= decision_threshold",
            "tie_break_order": [
                "recall (higher)",
                "precision (higher)",
                "false-positive rate (lower)",
                "threshold (higher)",
            ],
        },
        "feature_set": FEATURE_SET,
        "feature_count": len(feature_configuration["feature_order"]),
        "feature_order": feature_configuration["feature_order"],
        "seed": config.seed,
        "training_samples": int(len(train.binary_labels)),
        "validation_samples": int(len(validation.binary_labels)),
        "resampling": {
            "used": False,
            "reason": (
                "SMOTE was not used because interpolating engineered protocol "
                "frequency and time-series summary features may create "
                "implausible KPI samples."
            ),
        },
    }

    joblib.dump(final_estimator, artifact_path)
    joblib.dump(preprocessor, preprocessing_path)
    _write_json_atomic(feature_path, feature_configuration)
    metadata = {
        "model_version": model_version,
        "model_versions": {"detection": model_version},
        "decision_threshold": threshold,
        "positive_class": 1,
        "negative_class": 0,
        "feature_set": FEATURE_SET,
        "feature_count": len(feature_configuration["feature_order"]),
        "feature_order": feature_configuration["feature_order"],
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": revision,
        "training_seed": config.seed,
        "artifact_format": "joblib",
        "model_configuration": "model_configuration.json",
        "preprocessing_configuration": preprocessing_configuration(
            revision=revision,
            sample_count=len(train.binary_labels),
        ),
    }
    _write_json_atomic(metadata_path, metadata)

    reports.mkdir(parents=True, exist_ok=True)
    clean_candidate_results = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in candidate_results
    ]
    write_json(
        reports / "validation_model_comparison.json",
        {
            "selection_dataset": "validation",
            "selection_metric": "F2",
            "selected_candidate": selected_candidate.name,
            "selected_threshold": threshold,
            "candidates": clean_candidate_results,
        },
    )
    write_json(reports / "model_configuration.json", model_configuration)
    write_threshold_comparison(
        selected["threshold_rows"],
        threshold,
        reports,
    )

    train_validation_ids = set(train.sample_ids) | set(validation.sample_ids)
    _write_json_atomic(
        test_state_path,
        {
            "state": "started",
            "test_evaluation_count": 1,
            "model_version": model_version,
            "test_partition_loaded": False,
        },
    )
    test = _load_partition(split_root, "test", feature_configuration, preprocessor)
    _write_json_atomic(
        test_state_path,
        {
            "state": "started",
            "test_evaluation_count": 1,
            "model_version": model_version,
            "test_partition_loaded": True,
        },
    )
    if train_validation_ids.intersection(test.sample_ids):
        raise SchemaValidationError(
            ["Final test contains sample IDs already present in train or validation."]
        )
    test_probability = _positive_probability(final_estimator, test.features)
    test_metrics = evaluate_binary(test.binary_labels, test_probability, threshold)
    test_predictions = (test_probability >= threshold).astype(np.int8)
    class_support = _class_support(test.labels, test_predictions)
    test_report = {
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": revision,
        "model_version": model_version,
        "selected_candidate": selected_candidate.name,
        "decision_threshold": threshold,
        "threshold_selected_using": "validation only",
        "final_test_evaluation_count": 1,
        "test_set_pristine": pristine,
        "test_set_history": split_report.get("historical_test_data", {}),
        "sample_count": len(test.sample_ids),
        "source_file_count": len(set(test.source_ids)),
        "source_files": sorted(set(test.source_ids)),
        "normal_samples": int(np.count_nonzero(test.binary_labels == 0)),
        "anomalous_samples": int(np.count_nonzero(test.binary_labels == 1)),
        "missing_anomaly_categories": class_support["missing_anomaly_categories"],
        "metrics": test_metrics,
        "class_wise_detection": class_support["class_wise_detection"],
        "false_positive_samples": int(np.count_nonzero(
            (test.binary_labels == 0) & (test_predictions == 1)
        )),
        "false_negative_samples": int(np.count_nonzero(
            (test.binary_labels == 1) & (test_predictions == 0)
        )),
    }
    write_json(reports / "final_test_report.json", test_report)
    write_json(
        reports / "final_test_classification_report.json",
        test_metrics["classification_report"],
    )
    write_json(
        reports / "final_test_error_analysis.json",
        _error_analysis(test, test_probability, test_predictions),
    )
    write_test_figures(
        test.binary_labels,
        test_probability,
        threshold,
        test_metrics,
        reports,
    )
    _write_json_atomic(
        test_state_path,
        {
            "state": "completed",
            "test_evaluation_count": 1,
            "model_version": model_version,
            "test_partition_loaded": True,
        },
    )
    LOGGER.info(
        "Training complete: selected %s at threshold %.6f; test F2=%.4f, recall=%.4f",
        selected_candidate.name,
        threshold,
        test_metrics["f2"],
        test_metrics["recall"],
    )
    return {
        "model_version": model_version,
        "selected_candidate": selected_candidate.name,
        "decision_threshold": threshold,
        "test_metrics": test_metrics,
        "report_directory": str(reports),
        "test_set_pristine": pristine,
    }


def _load_partition(
    directory: Path,
    name: str,
    feature_configuration: dict[str, Any],
    preprocessor: TelecomTSPreprocessor,
    *,
    batch_size: int = 128,
) -> _Partition:
    """Stream aligned sequence/metadata JSONL files into shared features."""
    if name not in _PARTITION_FILES:
        raise ValueError(f"Unknown partition '{name}'.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    sequence_path, metadata_path = (
        directory / filename for filename in _PARTITION_FILES[name]
    )
    for path in (sequence_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} partition file is missing: {path}")

    feature_batches: list[pd.DataFrame] = []
    binary_labels: list[int] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    source_ids: list[str] = []
    pending_sequences: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def flush_batch() -> None:
        if not pending_sequences:
            return
        features = build_features(
            pd.DataFrame({"kpis": [item["kpis"] for item in pending_sequences]}),
            feature_configuration,
            preprocessor,
        )
        feature_batches.append(features)
        pending_sequences.clear()

    try:
        with (
            gzip.open(sequence_path, "rt", encoding="utf-8") as sequence_stream,
            gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
        ):
            for line_number, pair in enumerate(
                zip_longest(sequence_stream, metadata_stream),
                start=1,
            ):
                sequence_line, metadata_line = pair
                if sequence_line is None or metadata_line is None:
                    raise SchemaValidationError(
                        [f"{name} sequence and metadata counts differ at row {line_number}."]
                    )
                try:
                    sequence = json.loads(sequence_line)
                    metadata = json.loads(metadata_line)
                except json.JSONDecodeError as exc:
                    raise SchemaValidationError(
                        [f"Malformed JSON in {name} partition at row {line_number}: {exc}."]
                    ) from exc
                if not isinstance(sequence, dict) or not isinstance(metadata, dict):
                    raise SchemaValidationError(
                        [f"{name} partition row {line_number} must contain JSON objects."]
                    )
                sample_id = sequence.get("sample_id")
                label = metadata.get("label")
                source_label = metadata.get("source_anomaly_type")
                source_id = metadata.get("source_file_id")
                kpis = sequence.get("kpis")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise SchemaValidationError(
                        [f"Missing sample_id in {name} partition at row {line_number}."]
                    )
                if metadata.get("sample_id") != sample_id:
                    raise SchemaValidationError(
                        [f"Misaligned sample_id in {name} partition at row {line_number}."]
                    )
                if sample_id in seen_ids:
                    raise SchemaValidationError(
                        [f"Duplicate sample_id '{sample_id}' in {name} partition."]
                    )
                if not isinstance(label, str) or label not in TAXONOMY_LABELS:
                    raise SchemaValidationError(
                        [f"Invalid anomaly label in {name} partition at row {line_number}."]
                    )
                if (
                    not isinstance(source_label, str)
                    or LABEL_MAPPING.get(source_label.strip().casefold()) != label
                ):
                    raise SchemaValidationError(
                        [
                            f"Source anomaly label does not map to normalized label "
                            f"in {name} partition at row {line_number}."
                        ]
                    )
                if not isinstance(source_id, str) or not source_id.strip():
                    raise SchemaValidationError(
                        [f"Missing source_file_id in {name} partition at row {line_number}."]
                    )
                if not isinstance(kpis, dict):
                    raise SchemaValidationError(
                        [f"Missing KPI mapping in {name} partition at row {line_number}."]
                    )
                if set(kpis) != set(KPI_NAMES):
                    missing_kpis = sorted(set(KPI_NAMES) - set(kpis))
                    extra_kpis = sorted(set(kpis) - set(KPI_NAMES))
                    details = []
                    if missing_kpis:
                        details.append("missing: " + ", ".join(missing_kpis))
                    if extra_kpis:
                        details.append("unexpected: " + ", ".join(extra_kpis))
                    raise SchemaValidationError(
                        [
                            f"Invalid KPI channels in {name} partition at row "
                            f"{line_number} ({'; '.join(details)})."
                        ]
                    )
                seen_ids.add(sample_id)
                sample_ids.append(sample_id)
                labels.append(label)
                source_ids.append(source_id)
                binary_labels.append(int(label != "Normal"))
                pending_sequences.append({"kpis": kpis})
                if len(pending_sequences) >= batch_size:
                    flush_batch()
        flush_batch()
    except (OSError, EOFError, UnicodeError, gzip.BadGzipFile) as exc:
        raise SchemaValidationError(
            [f"Could not read the {name} partition: {exc}."]
        ) from exc

    if not sample_ids or not feature_batches:
        raise SchemaValidationError([f"The {name} partition contains no records."])
    features = pd.concat(feature_batches, ignore_index=True)
    if list(features.columns) != feature_configuration["feature_order"]:
        raise SchemaValidationError(
            ["Shared feature builder returned an unexpected feature order."]
        )
    values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise SchemaValidationError(
            [f"Non-finite engineered feature found in {name} partition."]
        )
    return _Partition(
        features=features,
        binary_labels=np.asarray(binary_labels, dtype=np.int8),
        labels=tuple(labels),
        sample_ids=tuple(sample_ids),
        source_ids=tuple(source_ids),
    )


def _load_split_report(path: Path, seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Split metadata is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError([f"Could not read split metadata: {exc}."]) from exc
    if not isinstance(report, dict) or report.get("seed") != seed:
        raise SchemaValidationError(["Split metadata is invalid or uses a different seed."])
    if not isinstance(report.get("source_overlap"), dict):
        raise SchemaValidationError(["Split metadata is missing source overlap checks."])
    if not isinstance(report.get("historical_test_data"), dict):
        raise SchemaValidationError(["Split metadata is missing test history status."])
    if not isinstance(
        report["historical_test_data"].get("test_set_pristine"),
        bool,
    ):
        raise SchemaValidationError(
            ["Split metadata must explicitly state whether the test set is pristine."]
        )
    expected_overlap_checks = {
        "train_and_validation",
        "train_and_test",
        "validation_and_test",
    }
    if (
        set(report["source_overlap"]) != expected_overlap_checks
        or any(not isinstance(groups, list) for groups in report["source_overlap"].values())
    ):
        raise SchemaValidationError(
            ["Split metadata has incomplete source-file overlap checks."]
        )
    return report


def _validate_partition_separation(train: _Partition, validation: _Partition) -> None:
    if set(train.sample_ids).intersection(validation.sample_ids):
        raise SchemaValidationError(["Train and validation contain duplicate sample IDs."])
    overlap = set(train.source_ids).intersection(validation.source_ids)
    if overlap:
        raise SchemaValidationError(
            ["Train and validation share source files: " + ", ".join(sorted(overlap))]
        )


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    predict_proba = getattr(estimator, "predict_proba", None)
    classes = getattr(estimator, "classes_", None)
    if not callable(predict_proba) or classes is None:
        raise ProjectError("Binary detector must expose predict_proba() and classes_.")
    class_values = list(classes)
    positive = [
        index
        for index, value in enumerate(class_values)
        if str(value).strip().lower() in {"1", "true", "anomaly", "anomalous"}
    ]
    if len(class_values) != 2 or len(positive) != 1:
        raise ProjectError("Binary detector classes must include exactly one positive class.")
    probabilities = np.asarray(predict_proba(features), dtype=np.float64)
    if (
        probabilities.shape != (len(features), 2)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ProjectError("Binary detector returned invalid class probabilities.")
    return probabilities[:, positive[0]]


def _class_support(
    labels: tuple[str, ...],
    predictions: np.ndarray,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    detected: dict[str, int] = {}
    for label, prediction in zip(labels, predictions, strict=True):
        counts[label] = counts.get(label, 0) + 1
        if label != "Normal" and prediction == 1:
            detected[label] = detected.get(label, 0) + 1
    anomaly_labels = [label for label in TAXONOMY_LABELS if label != "Normal"]
    class_rows = {
        label: {
            "support": counts.get(label, 0),
            "detected": detected.get(label, 0),
            "recall": (
                detected.get(label, 0) / counts[label]
                if counts.get(label, 0)
                else None
            ),
        }
        for label in anomaly_labels
    }
    normal_count = counts.get("Normal", 0)
    normal_false_positives = int(np.count_nonzero(
        np.asarray([label == "Normal" for label in labels]) & (predictions == 1)
    ))
    class_rows["Normal"] = {
        "support": normal_count,
        "false_positives": normal_false_positives,
        "false_positive_rate": (
            normal_false_positives / normal_count if normal_count else None
        ),
    }
    return {
        "class_wise_detection": class_rows,
        "missing_anomaly_categories": [
            label for label in anomaly_labels if counts.get(label, 0) == 0
        ],
    }


def _error_analysis(
    partition: _Partition,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for sample_id, source_id, label, probability, prediction in zip(
        partition.sample_ids,
        partition.source_ids,
        partition.labels,
        probabilities,
        predictions,
        strict=True,
    ):
        record = {
            "sample_id": sample_id,
            "source_file_id": source_id,
            "label": label,
            "anomaly_probability": float(probability),
        }
        if label == "Normal" and prediction == 1:
            false_positives.append(record)
        elif label != "Normal" and prediction == 0:
            false_negatives.append(record)
    return {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.yaml.")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--split-directory",
        type=Path,
        help="Override the source-grouped split directory.",
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        help="Override the report output directory.",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        result = run_training(
            config,
            revision=args.revision,
            split_directory=args.split_directory,
            report_directory=args.report_directory,
        )
    except ProjectError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
