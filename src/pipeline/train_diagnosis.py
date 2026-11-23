"""Fit, validate, and evaluate multiclass TelecomTS anomaly diagnosis."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone

from cl5g.common.exceptions import ProjectError, SchemaValidationError
from cl5g.data.telecomts import (
    DEFAULT_REVISION,
    KPI_NAMES,
)
from cl5g.evaluation.binary import write_json
from cl5g.evaluation.multiclass import (
    ANOMALY_CLASSES,
    compare_validation_test,
    evaluate_multiclass,
)
from cl5g.models.diagnosis import (
    DiagnosisCandidate,
    balanced_sample_weights,
    build_diagnosis_candidates,
)
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.pipeline.inference import InferencePipeline
from cl5g.pipeline.train import (
    FEATURE_SET,
    _Partition,
    _load_partition,
    _load_split_report,
    _validate_partition_separation,
)
from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor
from cl5g.features.extraction import build_feature_configuration

LOGGER = logging.getLogger(__name__)
_MINIMUM_TRAINING_SUPPORT_WARNING = 25


def run_diagnosis_training(
    config: AppConfig,
    *,
    revision: str = DEFAULT_REVISION,
    split_directory: Path | None = None,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    """Select a classifier on validation anomalies and evaluate test anomalies."""
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
        / "multiclass_diagnosis"
        / f"seed-{config.seed}"
    )
    test_state_path = reports / "test_evaluation_state.json"
    if test_state_path.exists():
        raise ProjectError(
            "Diagnosis test evaluation has already been started for this run "
            f"({test_state_path}); do not repeat it."
        )

    split_report = _load_split_report(split_root / "split_metadata.json", config.seed)
    if any(split_report["source_overlap"].values()):
        raise SchemaValidationError(
            ["Split metadata reports source-file overlap between partitions."]
        )

    feature_configuration = build_feature_configuration(FEATURE_SET)
    preprocessor = TelecomTSPreprocessor()
    partitions = {
        name: _load_partition(
            split_root,
            name,
            feature_configuration,
            preprocessor,
        )
        for name in ("train", "validation")
    }
    train_all = partitions["train"]
    validation_all = partitions["validation"]
    _validate_partition_separation(train_all, validation_all)
    train = _anomalous_only(train_all, "train")
    validation = _anomalous_only(validation_all, "validation")
    _validate_taxonomy_support(train.labels, "training")
    _validate_taxonomy_support(validation.labels, "validation")

    train_counts = _class_counts(train.labels)
    insufficient_training_support = {
        label: count
        for label, count in train_counts.items()
        if count < _MINIMUM_TRAINING_SUPPORT_WARNING
    }
    candidate_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in build_diagnosis_candidates(
        np.asarray(train.labels, dtype=object),
        seed=config.seed,
    ):
        LOGGER.info("Fitting diagnosis candidate %s", candidate.name)
        fit_arguments: dict[str, Any] = {}
        if candidate.balanced_sample_weight:
            fit_arguments["sample_weight"] = balanced_sample_weights(
                np.asarray(train.labels, dtype=object)
            )
        candidate.estimator.fit(
            train.features,
            np.asarray(train.labels, dtype=object),
            **fit_arguments,
        )
        validation_predictions = candidate.estimator.predict(validation.features)
        validation_metrics = evaluate_multiclass(
            np.asarray(validation.labels, dtype=object),
            validation_predictions,
        )
        result = {
            "name": candidate.name,
            "configuration": candidate.configuration,
            "validation_metrics": validation_metrics,
            "_selection_key": (
                -validation_metrics["macro_f1"],
                -validation_metrics["macro_recall"],
                -validation_metrics["balanced_accuracy"],
                -validation_metrics["weighted_f1"],
                candidate.name,
            ),
        }
        candidate_results.append(result)
        if selected is None or result["_selection_key"] < selected["_selection_key"]:
            selected = {
                **result,
                "candidate": candidate,
            }
    if selected is None:
        raise ProjectError("No multiclass diagnosis candidates were available.")

    selected_candidate: DiagnosisCandidate = selected["candidate"]
    final_model = clone(selected_candidate.estimator)
    fit_arguments = {}
    if selected_candidate.balanced_sample_weight:
        fit_arguments["sample_weight"] = balanced_sample_weights(
            np.asarray(train.labels, dtype=object)
        )
    final_model.fit(
        train.features,
        np.asarray(train.labels, dtype=object),
        **fit_arguments,
    )
    class_mapping = {
        str(index): str(label)
        for index, label in enumerate(final_model.classes_)
    }
    model_version = f"telecomts-diagnosis-{revision[:8]}-seed-{config.seed}"
    model_configuration = {
        "model_version": model_version,
        "selected_candidate": selected_candidate.name,
        "estimator": selected_candidate.configuration,
        "selection_metric": "validation macro F1 over classes with validation support",
        "tie_break_order": [
            "validation macro recall (higher)",
            "validation balanced accuracy (higher)",
            "validation weighted F1 (higher)",
            "candidate name (ascending)",
        ],
        "class_weighting": (
            "balanced class_weight for tree candidates or inverse-frequency "
            "sample weights for XGBoost; no records were resampled"
        ),
        "feature_set": FEATURE_SET,
        "feature_count": len(feature_configuration["feature_order"]),
        "feature_order": feature_configuration["feature_order"],
        "seed": config.seed,
        "training_anomalous_samples": len(train.labels),
        "training_class_support": train_counts,
        "minimum_support_warning_threshold": _MINIMUM_TRAINING_SUPPORT_WARNING,
        "classes_missing_from_training": [
            label for label in ANOMALY_CLASSES if train_counts.get(label, 0) == 0
        ],
        "insufficient_training_support": insufficient_training_support,
        "validation_anomalous_samples": len(validation.labels),
    }

    artifact_paths = {
        "model": config.resolve(config.diagnosis.artifact),
        "class_mapping": config.resolve("artifacts/diagnosis_class_mapping.json"),
        "configuration": config.resolve("artifacts/diagnosis_configuration.json"),
        "metadata": config.resolve("artifacts/diagnosis_metadata.json"),
    }
    for path in artifact_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, artifact_paths["model"])
    _write_json_atomic(artifact_paths["class_mapping"], class_mapping)
    _write_json_atomic(artifact_paths["configuration"], model_configuration)

    metadata = {
        "model_version": model_version,
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": revision,
        "training_seed": config.seed,
        "artifact_format": "joblib",
        "class_mapping": "diagnosis_class_mapping.json",
        "feature_configuration": config.inference.feature_config_artifact,
        "feature_set": FEATURE_SET,
        "feature_count": len(feature_configuration["feature_order"]),
        "feature_order": feature_configuration["feature_order"],
        "target_classes": list(ANOMALY_CLASSES),
        "training_sample_count": len(train.labels),
        "training_class_support": train_counts,
        "validation_sample_count": len(validation.labels),
        "selected_candidate": selected_candidate.name,
        "selection_metric": "validation macro F1 over supported classes",
        "test_partition_history": split_report["historical_test_data"],
    }
    _write_json_atomic(artifact_paths["metadata"], metadata)
    _register_diagnosis_version(
        config.resolve(config.inference.metadata_artifact),
        model_version,
    )

    reports.mkdir(parents=True, exist_ok=True)
    write_json(
        reports / "validation_model_comparison.json",
        {
            "selection_partition": "validation",
            "selection_metric": "macro F1 over classes with validation support",
            "selected_candidate": selected_candidate.name,
            "candidate_results": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in candidate_results
            ],
        },
    )

    smoke_examples = _run_real_inference_smoke(
        config,
        split_root / "train_sequences.jsonl.gz",
        split_root / "train_metadata.jsonl.gz",
    )
    write_json(reports / "end_to_end_smoke.json", smoke_examples)

    test_state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        test_state_path,
        {
            "state": "started",
            "test_evaluation_count": 1,
            "model_version": model_version,
            "test_partition_loaded": False,
            "test_partition_historical": True,
            "prior_binary_test_evaluation": True,
        },
    )
    test_all = _load_partition(
        split_root,
        "test",
        feature_configuration,
        preprocessor,
    )
    if (
        set(train_all.sample_ids).intersection(test_all.sample_ids)
        or set(validation_all.sample_ids).intersection(test_all.sample_ids)
    ):
        raise SchemaValidationError(
            ["The final test partition overlaps training or validation sample IDs."]
        )
    _validate_partition_separation(train_all, test_all)
    _validate_partition_separation(validation_all, test_all)
    test = _anomalous_only(test_all, "test")
    test_predictions = final_model.predict(test.features)
    validation_predictions = final_model.predict(validation.features)
    validation_metrics = evaluate_multiclass(
        np.asarray(validation.labels, dtype=object),
        validation_predictions,
    )
    test_metrics = evaluate_multiclass(
        np.asarray(test.labels, dtype=object),
        test_predictions,
    )
    test_counts = _class_counts(test.labels)
    validation_counts = _class_counts(validation.labels)
    test_report = {
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": revision,
        "model_version": model_version,
        "selected_candidate": selected_candidate.name,
        "evaluation_partition": "test, anomaly samples only",
        "test_partition_pristine": False,
        "test_partition_history": {
            "previously_used_for_binary_detection_evaluation": True,
            "description": (
                "The test partition was evaluated for binary detection before "
                "this diagnosis task. Diagnosis candidate and model selection "
                "used validation only, but the diagnosis test result is a "
                "historical holdout, not an untouched test estimate."
            ),
        },
        "sample_count": len(test.labels),
        "source_file_count": len(set(test.source_ids)),
        "source_files": sorted(set(test.source_ids)),
        "class_support": test_counts,
        "metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "validation_class_support": validation_counts,
        "train_class_support": train_counts,
        "missing_from_training": [
            label for label in ANOMALY_CLASSES if train_counts.get(label, 0) == 0
        ],
        "missing_from_validation": validation_metrics["missing_classes"],
        "missing_from_test": test_metrics["missing_classes"],
        "insufficient_training_support": insufficient_training_support,
        "poorly_classified_minority_categories": _poor_minority_categories(
            test_metrics,
            train_counts,
        ),
        "frequent_confusions": test_metrics["most_frequent_confusions"],
        "validation_test_comparison": compare_validation_test(
            validation_metrics,
            test_metrics,
        ),
    }
    write_json(reports / "final_test_report.json", test_report)
    write_json(reports / "validation_report.json", validation_metrics)
    write_json(
        reports / "class_support_report.json",
        {
            label: {
                "train": train_counts.get(label, 0),
                "validation": validation_counts.get(label, 0),
                "test": test_counts.get(label, 0),
            }
            for label in ANOMALY_CLASSES
        },
    )
    write_json(
        reports / "model_configuration.json",
        model_configuration,
    )

    _write_json_atomic(
        test_state_path,
        {
            "state": "completed",
            "test_evaluation_count": 1,
            "model_version": model_version,
            "test_partition_loaded": True,
            "test_partition_historical": True,
            "prior_binary_test_evaluation": True,
        },
    )
    LOGGER.info(
        "Diagnosis complete: selected %s; validation macro-F1=%.4f, "
        "historical test macro-F1=%.4f",
        selected_candidate.name,
        validation_metrics["macro_f1"],
        test_metrics["macro_f1"],
    )
    return {
        "model_version": model_version,
        "selected_candidate": selected_candidate.name,
        "train_class_support": train_counts,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_partition_pristine": False,
        "report_directory": str(reports),
        "end_to_end_smoke": smoke_examples,
    }


def _anomalous_only(partition: _Partition, name: str) -> _Partition:
    indexes = np.asarray(
        [index for index, label in enumerate(partition.labels) if label != "Normal"],
        dtype=np.int64,
    )
    if indexes.size == 0:
        raise SchemaValidationError([f"{name} partition contains no anomaly samples."])
    return _Partition(
        features=partition.features.iloc[indexes].reset_index(drop=True),
        binary_labels=partition.binary_labels[indexes],
        labels=tuple(partition.labels[int(index)] for index in indexes),
        sample_ids=tuple(partition.sample_ids[int(index)] for index in indexes),
        source_ids=tuple(partition.source_ids[int(index)] for index in indexes),
    )


def _validate_taxonomy_support(labels: tuple[str, ...], partition: str) -> None:
    unsupported = sorted(set(labels) - set(ANOMALY_CLASSES))
    if unsupported:
        raise SchemaValidationError(
            [f"{partition} contains unsupported diagnosis labels: {unsupported}."]
        )


def _class_counts(labels: tuple[str, ...]) -> dict[str, int]:
    return {
        label: int(sum(observed == label for observed in labels))
        for label in ANOMALY_CLASSES
    }


def _poor_minority_categories(
    metrics: dict[str, Any],
    train_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for label, result in metrics["per_class"].items():
        support = result["support"]
        if support and (
            result["recall"] < 0.5
            or train_counts.get(label, 0) < _MINIMUM_TRAINING_SUPPORT_WARNING
        ):
            rows.append(
                {
                    "class": label,
                    "training_support": train_counts.get(label, 0),
                    "test_support": support,
                    "test_recall": result["recall"],
                    "reason": (
                        "low test recall"
                        if result["recall"] < 0.5
                        else "low training support"
                    ),
                }
            )
    rows.sort(key=lambda row: (row["test_recall"], row["training_support"], row["class"]))
    return rows


def _run_real_inference_smoke(
    config: AppConfig,
    sequence_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Exercise the saved binary-to-diagnosis path using actual training records."""
    normal_pipeline = InferencePipeline(config)
    diagnosis_pipeline = InferencePipeline(config)
    normal_result: dict[str, Any] | None = None
    anomalous_result: dict[str, Any] | None = None
    anomaly_truth: str | None = None
    with (
        gzip.open(sequence_path, "rt", encoding="utf-8") as sequence_stream,
        gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
    ):
        for sequence_line, metadata_line in zip(sequence_stream, metadata_stream):
            sequence = json.loads(sequence_line)
            metadata = json.loads(metadata_line)
            sample_id = sequence["sample_id"]
            label = metadata["label"]
            sample = np.empty((128, len(KPI_NAMES)), dtype=object)
            for channel, kpi in enumerate(KPI_NAMES):
                sample[:, channel] = sequence["kpis"][kpi]
            if label == "Normal" and normal_result is None:
                result = normal_pipeline.predict(sample)
                if result["status"] != "normal":
                    continue
                if normal_pipeline.loading_status()["loaded_components"]["diagnosis"]:
                    raise ProjectError(
                        "The inference pipeline loaded diagnosis for a normal prediction."
                    )
                normal_result = {
                    "sample_id": sample_id,
                    "true_label": label,
                    "result": result,
                    "diagnosis_loaded": False,
                }
            elif label != "Normal" and anomalous_result is None:
                result = diagnosis_pipeline.predict(sample)
                if result["status"] != "anomaly":
                    continue
                if result["predicted_anomaly_type"] not in ANOMALY_CLASSES:
                    raise ProjectError(
                        "End-to-end inference returned an unsupported diagnosis category."
                    )
                anomalous_result = {
                    "sample_id": sample_id,
                    "true_label": label,
                    "result": result,
                }
                anomaly_truth = label
            if normal_result is not None and anomalous_result is not None:
                break
    if normal_result is None or anomalous_result is None:
        raise ProjectError(
            "Could not find real training examples for normal and detected-anomaly "
            "end-to-end inference."
        )
    return {
        "data_source": "actual TelecomTS training partition",
        "evaluation_metric": False,
        "normal_sequence": normal_result,
        "anomalous_sequence": anomalous_result,
        "anomalous_sequence_true_category": anomaly_truth,
        "diagnosis_only_after_binary_anomaly": True,
    }


def _register_diagnosis_version(metadata_path: Path, model_version: str) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(
            [f"Could not update shared model metadata: {exc}."]
        ) from exc
    if not isinstance(metadata, dict):
        raise SchemaValidationError(["Shared model metadata must be a JSON object."])
    versions = metadata.get("model_versions")
    if not isinstance(versions, dict):
        versions = {}
    versions["diagnosis"] = model_version
    metadata["model_versions"] = versions
    _write_json_atomic(metadata_path, metadata)


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
    parser.add_argument("--split-directory", type=Path)
    parser.add_argument("--report-directory", type=Path)
    args = parser.parse_args()
    try:
        result = run_diagnosis_training(
            load_config(args.config),
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
