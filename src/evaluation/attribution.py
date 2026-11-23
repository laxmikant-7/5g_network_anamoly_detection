"""Validation-only SHAP, permutation, and KPI-level explainability reports."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split

from cl5g.common.exceptions import ProjectError, SchemaValidationError
from cl5g.data.telecomts import DEFAULT_REVISION
from cl5g.evaluation.binary import write_json
from cl5g.evaluation.multiclass import ANOMALY_CLASSES
from cl5g.models.attribution import (
    AGGREGATION_METHOD,
    EXPLANATION_CAVEAT,
    KPIAttributionExplainer,
)
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.pipeline.train import (
    FEATURE_SET,
    _load_partition,
)
from cl5g.pipeline.train_diagnosis import _anomalous_only
from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor
from cl5g.features.extraction import build_feature_configuration

LOGGER = logging.getLogger(__name__)
_MAX_EXPLANATION_SAMPLES = 256
_PERMUTATION_REPEATS = 3


def run_explainability(
    config: AppConfig,
    *,
    revision: str = DEFAULT_REVISION,
    split_directory: Path | None = None,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    """Generate global reports from validation data and save the inference adapter."""
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
        / "explainability"
        / f"seed-{config.seed}"
    )
    detection_path = config.resolve(config.inference.detection_artifact)
    diagnosis_path = config.resolve(config.diagnosis.artifact)
    feature_path = config.resolve(config.inference.feature_config_artifact)
    for path in (detection_path, diagnosis_path, feature_path):
        if not path.is_file():
            raise ProjectError(f"Required model/feature artifact is missing: {path}")
    try:
        detector = joblib.load(detection_path)
        diagnosis = joblib.load(diagnosis_path)
        feature_configuration = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, EOFError, ImportError, ValueError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not load explainability inputs: {exc}") from exc

    partition = _load_partition(
        split_root,
        "validation",
        feature_configuration,
        TelecomTSPreprocessor(),
    )
    anomalous = _anomalous_only(partition, "validation")
    feature_names = tuple(feature_configuration["feature_order"])
    if (
        tuple(partition.features.columns) != feature_names
        or tuple(anomalous.features.columns) != feature_names
    ):
        raise SchemaValidationError(
            ["Validation features do not match the saved feature configuration."]
        )

    explainer = KPIAttributionExplainer(feature_configuration)
    artifact_path = config.resolve(config.inference.attribution_artifact) if (
        config.inference.attribution_artifact is not None
    ) else config.resolve("artifacts/attribution.joblib")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, artifact_path)

    reports.mkdir(parents=True, exist_ok=True)
    binary_labels = partition.binary_labels
    binary_positive = _positive_class(detector)
    binary_probabilities = _positive_probabilities(detector, partition.features, binary_positive)
    metadata = _load_model_metadata(config)
    threshold = metadata.get("decision_threshold", 0.5)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise SchemaValidationError(
            ["Detection metadata has an invalid decision_threshold."]
        )
    binary_predictions = binary_probabilities >= float(threshold)
    binary_candidates = np.flatnonzero((binary_labels == 1) & binary_predictions)
    if binary_candidates.size == 0:
        raise ProjectError(
            "No validation anomaly was detected by the saved binary model; "
            "cannot create a representative local explanation."
        )
    sample_index = int(binary_candidates[0])
    sample_id = partition.sample_ids[sample_index]
    diagnosis_partition_index = next(
        (
            index
            for index, value in enumerate(anomalous.sample_ids)
            if value == sample_id
        ),
        None,
    )
    if diagnosis_partition_index is None:
        raise SchemaValidationError(
            ["Binary-positive validation sample is missing from anomaly features."]
        )
    diagnosis_index = diagnosis_partition_index
    diagnosis_prediction = diagnosis.predict(
        anomalous.features.iloc[[diagnosis_index]]
    )[0]
    sample_features = partition.features.iloc[[sample_index]]
    sample_explanations = {
        "sample_id": sample_id,
        "true_label": partition.labels[sample_index],
        "binary_detection": explainer.explain_prediction(
            sample_features,
            model=detector,
            model_kind="binary_detection",
            predicted_class=binary_positive,
            probability=float(binary_probabilities[sample_index]),
        ),
        "multiclass_diagnosis": explainer.explain_prediction(
            sample_features,
            model=diagnosis,
            model_kind="multiclass_diagnosis",
            predicted_class=diagnosis_prediction,
            probability=_class_probability(
                diagnosis,
                anomalous.features.iloc[[diagnosis_index]],
                diagnosis_prediction,
            ),
        ),
        "evaluation_metric": False,
        "interpretation": EXPLANATION_CAVEAT,
    }

    binary_indexes = _stratified_indexes(
        binary_labels,
        max_samples=_MAX_EXPLANATION_SAMPLES,
        seed=config.seed,
    )
    binary_features = partition.features.iloc[binary_indexes]
    binary_targets = binary_labels[binary_indexes]
    binary_shap = explainer.shap_values(
        binary_features,
        model=detector,
        predicted_class=binary_positive,
    )
    binary_permutation = _permutation_scores(
        detector,
        binary_features,
        binary_targets,
        scoring="average_precision",
        seed=config.seed,
    )
    binary_report = _write_model_reports(
        output_directory=reports,
        model_name="binary_detection",
        feature_names=feature_names,
        feature_to_kpi=explainer.feature_to_kpi,
        kpi_order=explainer.kpi_order,
        features=binary_features,
        shap_values=binary_shap,
        permutation=binary_permutation,
        sample_features=sample_features,
        sample_shap=sample_explanations["binary_detection"],
        class_labels=[str(value) for value in getattr(detector, "classes_", [])],
        plot_kind="binary",
    )

    diagnosis_indexes = _stratified_indexes(
        np.asarray(anomalous.labels, dtype=object),
        max_samples=_MAX_EXPLANATION_SAMPLES,
        seed=config.seed,
    )
    diagnosis_features = anomalous.features.iloc[diagnosis_indexes]
    diagnosis_targets = np.asarray(anomalous.labels, dtype=object)[diagnosis_indexes]
    diagnosis_classes = list(getattr(diagnosis, "classes_", []))
    per_class_shap = [
        explainer.shap_values(
            diagnosis_features,
            model=diagnosis,
            predicted_class=class_label,
        )
        for class_label in diagnosis_classes
    ]
    diagnosis_shap = np.stack(per_class_shap, axis=1)
    supported_targets = [
        label for label in ANOMALY_CLASSES if np.any(diagnosis_targets == label)
    ]
    macro_f1_scorer = make_scorer(
        f1_score,
        average="macro",
        labels=supported_targets,
        zero_division=0,
    )
    diagnosis_permutation = _permutation_scores(
        diagnosis,
        diagnosis_features,
        diagnosis_targets,
        scoring=macro_f1_scorer,
        seed=config.seed,
    )
    diagnosis_report = _write_model_reports(
        output_directory=reports,
        model_name="multiclass_diagnosis",
        feature_names=feature_names,
        feature_to_kpi=explainer.feature_to_kpi,
        kpi_order=explainer.kpi_order,
        features=diagnosis_features,
        shap_values=diagnosis_shap,
        permutation=diagnosis_permutation,
        sample_features=sample_features,
        sample_shap=sample_explanations["multiclass_diagnosis"],
        class_labels=[str(value) for value in diagnosis_classes],
        plot_kind="multiclass",
    )

    sample_path = reports / "sample_prediction_explanation.json"
    write_json(sample_path, sample_explanations)
    binary_shap_values, _ = explainer.shap_contributions(
        sample_features,
        model=detector,
        predicted_class=binary_positive,
    )
    _write_waterfall(
        sample_explanations["binary_detection"],
        sample_features.iloc[0].to_numpy(dtype=np.float64),
        list(feature_names),
        binary_shap_values,
        reports / "sample_binary_explanation.png",
    )
    diagnosis_shap_values, _ = explainer.shap_contributions(
        sample_features,
        model=diagnosis,
        predicted_class=diagnosis_prediction,
    )
    _write_waterfall(
        sample_explanations["multiclass_diagnosis"],
        sample_features.iloc[0].to_numpy(dtype=np.float64),
        list(feature_names),
        diagnosis_shap_values,
        reports / "sample_diagnosis_explanation.png",
    )
    report = {
        "dataset_revision": revision,
        "seed": config.seed,
        "explanation_data_partition": "validation only",
        "test_partition_used": False,
        "aggregation_method": AGGREGATION_METHOD,
        "interpretation": EXPLANATION_CAVEAT,
        "permutation_importance": {
            "repeats": _PERMUTATION_REPEATS,
            "max_samples": _MAX_EXPLANATION_SAMPLES,
            "binary_scoring": "average precision",
            "diagnosis_scoring": "macro F1 over validation-supported classes",
            "kpi_aggregation": (
                "sum positive feature-level mean score decreases; negative "
                "estimates are retained in feature-level CSV but contribute "
                "zero to KPI importance"
            ),
        },
        "models": {
            "binary_detection": binary_report,
            "multiclass_diagnosis": diagnosis_report,
        },
        "artifacts": {
            "attribution_adapter": str(artifact_path),
            "sample_explanation": str(sample_path),
        },
    }
    write_json(reports / "global_explainability_report.json", report)
    return report


def _write_model_reports(
    *,
    output_directory: Path,
    model_name: str,
    feature_names: tuple[str, ...],
    feature_to_kpi: dict[str, str],
    kpi_order: tuple[str, ...],
    features: pd.DataFrame,
    shap_values: np.ndarray,
    permutation: Any,
    sample_features: pd.DataFrame,
    sample_shap: dict[str, Any],
    class_labels: list[str],
    plot_kind: str,
) -> dict[str, Any]:
    folder = output_directory / model_name
    folder.mkdir(parents=True, exist_ok=True)
    values = np.asarray(shap_values, dtype=np.float64)
    if plot_kind == "binary":
        if values.shape != (len(features), len(feature_names)):
            raise ValueError("Binary SHAP matrix does not match the validation features.")
        mean_absolute = np.mean(np.abs(values), axis=0)
    else:
        if (
            values.ndim != 3
            or values.shape[0] != len(features)
            or values.shape[2] != len(feature_names)
        ):
            raise ValueError("Multiclass SHAP matrix does not match features/classes.")
        mean_absolute = np.mean(np.abs(values), axis=(0, 1))

    permutation_mean = np.asarray(permutation.importances_mean, dtype=np.float64)
    permutation_std = np.asarray(permutation.importances_std, dtype=np.float64)
    if (
        permutation_mean.shape != (len(feature_names),)
        or permutation_std.shape != (len(feature_names),)
    ):
        raise ValueError("Permutation importance does not match feature ordering.")
    feature_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "source_kpi": [feature_to_kpi[name] for name in feature_names],
            "mean_absolute_shap": mean_absolute,
            "permutation_mean_score_decrease": permutation_mean,
            "permutation_std": permutation_std,
        }
    ).sort_values("mean_absolute_shap", ascending=False, kind="stable")
    feature_frame.to_csv(folder / "global_feature_importance.csv", index=False)

    kpi_rows = []
    for kpi in kpi_order:
        rows = feature_frame[feature_frame["source_kpi"] == kpi]
        kpi_rows.append(
            {
                "kpi": kpi,
                "mean_absolute_shap": float(rows["mean_absolute_shap"].sum()),
                "permutation_importance": float(
                    np.maximum(
                        rows["permutation_mean_score_decrease"].to_numpy(),
                        0.0,
                    ).sum()
                ),
                "engineered_feature_count": int(len(rows)),
            }
        )
    kpi_rows.sort(key=lambda row: (-row["mean_absolute_shap"], row["kpi"]))
    write_json(
        folder / "kpi_level_importance.json",
        {
            "aggregation_method": AGGREGATION_METHOD,
            "permutation_aggregation": (
                "sum of positive feature-level mean score decreases"
            ),
            "scores": kpi_rows,
        },
    )
    pd.DataFrame(kpi_rows).to_csv(folder / "kpi_level_importance.csv", index=False)
    _plot_feature_importance(
        feature_frame,
        folder / "global_feature_importance.png",
        model_name,
    )
    _plot_kpi_importance(
        kpi_rows,
        folder / "kpi_level_importance.png",
        model_name,
    )
    _plot_permutation_importance(
        feature_frame,
        folder / "permutation_importance.png",
        model_name,
    )
    _plot_shap_summary(
        values,
        features,
        folder / "shap_summary.png",
        model_name,
        class_labels,
        plot_kind,
    )
    return {
        "validation_sample_count": int(len(features)),
        "feature_count": int(len(feature_names)),
        "kpi_count": int(len(kpi_order)),
        "permutation_repeats": _PERMUTATION_REPEATS,
        "importance_method": "mean absolute TreeSHAP by output",
        "sample_explanation_top_kpis": sample_shap["top_kpis"][:5],
        "sample_feature_count": int(len(sample_features.columns)),
        "report_directory": str(folder),
    }


def _plot_feature_importance(
    importance: pd.DataFrame,
    path: Path,
    model_name: str,
) -> None:
    top = importance.nlargest(20, "mean_absolute_shap").sort_values(
        "mean_absolute_shap"
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["feature"], top["mean_absolute_shap"], color="#4472C4")
    axis.set(
        title=f"{model_name.replace('_', ' ').title()} global feature importance",
        xlabel="Mean absolute SHAP value",
        ylabel="Engineered feature",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_kpi_importance(
    rows: list[dict[str, Any]],
    path: Path,
    model_name: str,
) -> None:
    ordered = sorted(rows, key=lambda row: row["mean_absolute_shap"])
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        [row["kpi"] for row in ordered],
        [row["mean_absolute_shap"] for row in ordered],
        color="#70AD47",
    )
    axis.set(
        title=f"{model_name.replace('_', ' ').title()} KPI-level importance",
        xlabel="Sum of mean absolute SHAP across mapped features",
        ylabel="Original KPI",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_permutation_importance(
    importance: pd.DataFrame,
    path: Path,
    model_name: str,
) -> None:
    top = importance.nlargest(20, "permutation_mean_score_decrease").sort_values(
        "permutation_mean_score_decrease"
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        top["feature"],
        top["permutation_mean_score_decrease"],
        xerr=top["permutation_std"],
        color="#ED7D31",
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"{model_name.replace('_', ' ').title()} permutation importance",
        xlabel="Validation score decrease after permutation",
        ylabel="Engineered feature",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_shap_summary(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    path: Path,
    model_name: str,
    class_labels: list[str],
    plot_kind: str,
) -> None:
    import shap

    plt.figure(figsize=(11, 8))
    if plot_kind == "binary":
        shap.summary_plot(
            shap_values,
            features,
            max_display=20,
            show=False,
        )
    else:
        outputs = [shap_values[:, index, :] for index in range(shap_values.shape[1])]
        shap.summary_plot(
            outputs,
            features,
            class_names=class_labels,
            max_display=20,
            show=False,
        )
    plt.title(f"{model_name.replace('_', ' ').title()} TreeSHAP summary")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _write_waterfall(
    explanation: dict[str, Any],
    feature_values: np.ndarray,
    feature_names: list[str],
    contributions: np.ndarray,
    path: Path,
) -> None:
    import shap

    values = np.asarray(contributions, dtype=np.float64)
    if values.shape != (len(feature_names),):
        raise ValueError("Waterfall contributions do not match feature ordering.")
    top_indexes = sorted(
        range(len(values)),
        key=lambda index: (-abs(values[index]), feature_names[index]),
    )[:14]
    remaining = [index for index in range(len(values)) if index not in set(top_indexes)]
    plot_values = [float(values[index]) for index in top_indexes]
    plot_data = [float(feature_values[index]) for index in top_indexes]
    plot_names = [feature_names[index] for index in top_indexes]
    if remaining:
        plot_values.append(float(values[remaining].sum()))
        plot_data.append(0.0)
        plot_names.append("Other features")
    plot_explanation = shap.Explanation(
        values=np.asarray(plot_values, dtype=np.float64),
        base_values=explanation["baseline_value"],
        data=np.asarray(plot_data, dtype=np.float64),
        feature_names=plot_names,
    )
    figure = plt.figure(figsize=(11, 7))
    shap.plots.waterfall(plot_explanation, max_display=15, show=False)
    plt.title(
        f"{explanation['model_kind'].replace('_', ' ').title()} explanation "
        f"for {explanation['predicted_class']}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _positive_class(model: object) -> object:
    classes = list(getattr(model, "classes_", []))
    matches = [
        label
        for label in classes
        if str(label).strip().lower() in {"1", "true", "anomaly", "anomalous"}
    ]
    if len(matches) != 1:
        raise ProjectError("Binary detector must expose one positive anomaly class.")
    return matches[0]


def _positive_probabilities(
    model: object,
    features: pd.DataFrame,
    positive_class: object,
) -> np.ndarray:
    classes = list(getattr(model, "classes_", []))
    index = classes.index(positive_class)
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ProjectError("Binary detector does not expose predict_proba().")
    probabilities = np.asarray(predict_proba(features), dtype=np.float64)
    if (
        probabilities.shape != (len(features), len(classes))
        or not np.isfinite(probabilities).all()
    ):
        raise ProjectError("Binary detector returned invalid probabilities.")
    return probabilities[:, index]


def _class_probability(
    model: object,
    features: pd.DataFrame,
    class_label: object,
) -> float:
    classes = list(getattr(model, "classes_", []))
    if class_label not in classes:
        raise ProjectError("Diagnosis model prediction is not present in classes_.")
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    return float(probabilities[0, classes.index(class_label)])


def _stratified_indexes(
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    indexes = np.arange(values.size)
    if values.size <= max_samples:
        return indexes
    selected, _ = train_test_split(
        indexes,
        train_size=max_samples,
        random_state=seed,
        stratify=values,
    )
    return np.sort(selected)


def _permutation_scores(
    model: object,
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    scoring: str | Any,
    seed: int,
) -> Any:
    try:
        return permutation_importance(
            model,
            features,
            labels,
            scoring=scoring,
            n_repeats=_PERMUTATION_REPEATS,
            random_state=seed,
            n_jobs=1,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProjectError(f"Permutation importance failed: {exc}") from exc


def _load_model_metadata(config: AppConfig) -> dict[str, Any]:
    path = config.resolve(config.inference.metadata_artifact)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not read detection model metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError(["Detection model metadata must be a JSON object."])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to config.yaml.")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split-directory", type=Path)
    parser.add_argument("--report-directory", type=Path)
    args = parser.parse_args()
    try:
        result = run_explainability(
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
