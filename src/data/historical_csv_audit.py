"""Quality audit and visual reports for the dashboard historical CSV."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cl5g.data.telecomts import (
    ALLOWED_PROTOCOL_VALUES,
    DEFAULT_REVISION,
    KPI_NAMES,
    LABEL_MAPPING,
    NUMERIC_KPIS,
    TAXONOMY_LABELS,
    TIME_STEPS,
)

CSV_COLUMNS = ("timestamp", "entity_id", "layer", "anomaly_type", *KPI_NAMES)
_MIB = 1024 * 1024


def audit_historical_csv(
    project_root: Path,
    *,
    revision: str = DEFAULT_REVISION,
    csv_path: Path | None = None,
    metadata_path: Path | None = None,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Audit the exported historical CSV and cross-check labels against ingestion metadata."""
    project_root = project_root.resolve()
    csv_path = (
        csv_path
        or project_root / "data" / "processed" / "telecomts_historical_exploration.csv"
    ).resolve()
    metadata_path = (
        metadata_path
        or project_root
        / "data"
        / "processed"
        / "telecomts"
        / revision
        / "metadata.jsonl.gz"
    ).resolve()
    output_directory = (output_directory or project_root / "reports").resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Historical CSV not found: {csv_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Processed metadata not found: {metadata_path}")

    frame = pd.read_csv(csv_path, keep_default_na=False)
    schema_matches = tuple(frame.columns) == CSV_COLUMNS
    expected_columns_missing = [name for name in CSV_COLUMNS if name not in frame.columns]
    unexpected_columns = [str(name) for name in frame.columns if name not in CSV_COLUMNS]

    numeric_values = pd.DataFrame(index=frame.index)
    numeric_quality: dict[str, dict[str, int | float | None]] = {}
    for name in NUMERIC_KPIS:
        raw = frame[name] if name in frame else pd.Series("", index=frame.index)
        converted = pd.to_numeric(raw, errors="coerce")
        missing_count = int(raw.astype(str).str.strip().eq("").sum())
        nonnumeric_count = int((converted.isna() & ~raw.astype(str).str.strip().eq("")).sum())
        finite_mask = np.isfinite(converted.to_numpy(dtype=np.float64, na_value=np.nan))
        nonfinite_count = int((~finite_mask & ~converted.isna().to_numpy()).sum())
        numeric_values[name] = converted
        numeric_quality[name] = {
            "missing": missing_count,
            "nonnumeric": nonnumeric_count,
            "nonfinite": nonfinite_count,
            "unique_values": int(converted.nunique(dropna=True)),
        }

    categorical_quality: dict[str, dict[str, Any]] = {}
    for name in ("UL_Protocol", "DL_Protocol"):
        values = frame[name].astype(str) if name in frame else pd.Series("", index=frame.index)
        invalid = sorted(set(values) - ALLOWED_PROTOCOL_VALUES)
        categorical_quality[name] = {
            "allowed_values": sorted(ALLOWED_PROTOCOL_VALUES),
            "observed_counts": {
                str(key): int(value)
                for key, value in values.value_counts(dropna=False).items()
            },
            "invalid_value_count": int(values.isin(invalid).sum()),
            "invalid_values": invalid,
        }

    labels = frame["anomaly_type"].astype(str) if "anomaly_type" in frame else pd.Series("", index=frame.index)
    invalid_labels = sorted(set(labels) - set(TAXONOMY_LABELS))
    entity_ids = frame["entity_id"].astype(str) if "entity_id" in frame else pd.Series("", index=frame.index)
    empty_entity_count = int(entity_ids.str.strip().eq("").sum())
    counts_by_entity = frame.groupby("entity_id", sort=False, dropna=False).size() if "entity_id" in frame else pd.Series(dtype=int)
    wrong_length_entities = {
        str(entity_id): int(count)
        for entity_id, count in counts_by_entity.items()
        if count != TIME_STEPS
    }
    labels_per_entity = (
        frame.groupby("entity_id", sort=False, dropna=False)["anomaly_type"]
        .nunique(dropna=False)
        if {"entity_id", "anomaly_type"}.issubset(frame.columns)
        else pd.Series(dtype=int)
    )
    inconsistent_label_entities = [
        str(entity_id)
        for entity_id, count in labels_per_entity.items()
        if count != 1
    ]

    timestamps = (
        pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if "timestamp" in frame
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )
    invalid_timestamp_rows = int(timestamps.isna().sum())
    duplicate_rows = int(frame.duplicated(keep=False).sum())
    duplicate_entity_timestamps = (
        int(frame.duplicated(["entity_id", "timestamp"], keep=False).sum())
        if {"entity_id", "timestamp"}.issubset(frame.columns)
        else 0
    )

    metadata_report = _cross_check_metadata(
        metadata_path,
        selected_ids=set(entity_ids),
        csv_labels=dict(zip(entity_ids, labels)),
    )
    timestamp_schedule_mismatches = _timestamp_schedule_mismatches(
        frame,
        timestamps,
        metadata_report["selected_metadata"],
    )

    class_sample_counts = Counter()
    if {"entity_id", "anomaly_type"}.issubset(frame.columns):
        class_sample_counts.update(
            frame.drop_duplicates("entity_id")["anomaly_type"].astype(str).tolist()
        )
    class_row_counts = Counter(labels.tolist())
    numeric_outliers = _outlier_statistics(numeric_values)
    correlation = numeric_values.corr(method="pearson")
    constant_features = [
        name
        for name, result in numeric_quality.items()
        if result["unique_values"] <= 1
    ]
    binary_sample_counts = {
        "normal": int(class_sample_counts.get("Normal", 0)),
        "anomalous": int(
            sum(count for label, count in class_sample_counts.items() if label != "Normal")
        ),
    }
    anomaly_to_normal_ratio = (
        binary_sample_counts["anomalous"] / binary_sample_counts["normal"]
        if binary_sample_counts["normal"]
        else None
    )

    layer_values = (
        sorted(frame["layer"].astype(str).unique().tolist())
        if "layer" in frame
        else []
    )
    noncanonical_feature_columns = [
        name
        for name in frame.columns
        if name not in {"timestamp", "entity_id", "layer", "anomaly_type", *KPI_NAMES}
    ]
    critical_checks = {
        "canonical_csv_columns": schema_matches,
        "all_numeric_values_valid": all(
            result["missing"] == result["nonnumeric"] == result["nonfinite"] == 0
            for result in numeric_quality.values()
        ),
        "categorical_values_valid": all(
            result["invalid_value_count"] == 0 for result in categorical_quality.values()
        ),
        "labels_in_taxonomy": not invalid_labels,
        "entity_ids_present": empty_entity_count == 0,
        "exactly_128_rows_per_entity": not wrong_length_entities,
        "one_label_per_entity": not inconsistent_label_entities,
        "timestamps_valid": invalid_timestamp_rows == 0,
        "unique_entity_timestamps": duplicate_entity_timestamps == 0,
        "labels_match_ingestion_metadata": (
            metadata_report["selected_samples_matched"] == len(set(entity_ids))
            and metadata_report["label_mapping_mismatches"] == 0
            and not metadata_report["selected_label_mismatches"]
            and metadata_report["duplicate_selected_metadata_ids"] == 0
        ),
        "timestamps_match_source_schedule": not timestamp_schedule_mismatches,
    }
    source_report = _read_historical_export_report(project_root)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": revision,
        "csv_path": str(csv_path),
        "csv_sha256": _sha256_file(csv_path),
        "metadata_path": str(metadata_path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "file_size_bytes": csv_path.stat().st_size,
        "file_size_mib": round(csv_path.stat().st_size / _MIB, 3),
        "canonical_columns": list(CSV_COLUMNS),
        "columns_match_canonical_order": schema_matches,
        "missing_columns": expected_columns_missing,
        "unexpected_columns": unexpected_columns,
        "sample_count": int(len(counts_by_entity)),
        "rows_per_sample": TIME_STEPS,
        "observed_rows_per_sample": sorted(
            {int(value) for value in counts_by_entity.unique().tolist()}
        ),
        "wrong_length_sample_count": len(wrong_length_entities),
        "wrong_length_sample_examples": dict(list(wrong_length_entities.items())[:20]),
        "empty_entity_id_rows": empty_entity_count,
        "inconsistent_label_sample_count": len(inconsistent_label_entities),
        "inconsistent_label_sample_examples": inconsistent_label_entities[:20],
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "duplicate_full_rows": duplicate_rows,
        "duplicate_entity_timestamp_rows": duplicate_entity_timestamps,
        "numeric_kpi_quality": numeric_quality,
        "categorical_kpi_quality": categorical_quality,
        "missing_values_total": int(
            sum(int(result["missing"]) for result in numeric_quality.values())
        ),
        "invalid_labels": invalid_labels,
        "sample_class_distribution": {
            label: int(class_sample_counts.get(label, 0)) for label in TAXONOMY_LABELS
        },
        "row_class_distribution": {
            label: int(class_row_counts.get(label, 0)) for label in TAXONOMY_LABELS
        },
        "binary_sample_distribution": binary_sample_counts,
        "anomalous_to_normal_sample_ratio": anomaly_to_normal_ratio,
        "original_metadata_audit": {
            key: value
            for key, value in metadata_report.items()
            if key != "selected_metadata"
        },
        "timestamp_schedule_mismatch_samples": timestamp_schedule_mismatches,
        "numeric_constant_features": constant_features,
        "numeric_outlier_statistics_tukey_1_5_iqr": numeric_outliers,
        "numeric_pearson_correlation": _json_matrix(correlation),
        "layer_values": layer_values,
        "layer_annotation_note": (
            "The historical CSV assigns ran_du for dashboard compatibility; it is "
            "not a TelecomTS layer-level ground-truth annotation."
        ),
        "leakage_and_sampling_assessment": {
            "target_column": "anomaly_type",
            "predictive_columns": list(KPI_NAMES),
            "excluded_identifier_and_time_columns": ["entity_id", "timestamp"],
            "excluded_layer_column": "layer",
            "layer_is_original_ground_truth": False,
            "extra_metadata_columns_in_csv": noncanonical_feature_columns,
            "entity_grouping_required_for_splits": True,
            "row_level_random_split_is_leaky": True,
            "historical_export_sampling": source_report,
            "warning": (
                "This is a stratified, capped dashboard export, not a prevalence-"
                "representative training set. Treat each entity as one sample and "
                "keep all 128 timesteps in one split."
            ),
        },
        "checks": critical_checks,
        "overall_validation_passed": all(critical_checks.values()),
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory = output_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    _create_figures(frame, numeric_values, labels, class_sample_counts, numeric_outliers, correlation, figures_directory)
    report["figures"] = [
        str(figures_directory / name)
        for name in (
            "anomaly_class_distribution.png",
            "missing_value_distribution.png",
            "kpi_correlation_heatmap.png",
            "normal_vs_anomalous_kpi_distributions.png",
            "kpi_outlier_distributions.png",
        )
    ]
    _write_json(output_directory / "dataset_quality_report.json", report)
    _write_markdown(output_directory / "dataset_quality_report.md", report)
    return report


def _cross_check_metadata(
    metadata_path: Path,
    *,
    selected_ids: set[str],
    csv_labels: dict[str, str],
) -> dict[str, Any]:
    selected_metadata: dict[str, dict[str, Any]] = {}
    full_class_counts: Counter[str] = Counter()
    duplicate_ids = 0
    mapping_mismatches = 0
    invalid_anomaly_intervals = 0
    invalid_affected_kpi_annotations = 0
    selected_mismatches: list[dict[str, Any]] = []
    row_count = 0
    try:
        with gzip.open(metadata_path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row_count += 1
                try:
                    metadata = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed metadata JSON at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(metadata, dict):
                    raise ValueError(f"Metadata at line {line_number} is not an object.")
                sample_id = metadata.get("sample_id")
                label = metadata.get("label")
                if isinstance(label, str):
                    full_class_counts[label] += 1
                anomalies = metadata.get("anomalies")
                anomalies = anomalies if isinstance(anomalies, dict) else {}
                exists = metadata.get("binary_label")
                raw_type = anomalies.get("type")
                mapped_type = (
                    LABEL_MAPPING.get(raw_type.strip().casefold())
                    if isinstance(raw_type, str)
                    else None
                )
                expected_label = "Normal" if exists is False else mapped_type
                mapping_valid = (
                    isinstance(label, str)
                    and label == expected_label
                    and metadata.get("source_anomaly_type") == raw_type
                    and ((label == "Normal") == (exists is False))
                )
                if label != "Normal":
                    duration = anomalies.get("anomaly_duration")
                    start = duration.get("start") if isinstance(duration, dict) else None
                    end = duration.get("end") if isinstance(duration, dict) else None
                    if (
                        isinstance(start, bool)
                        or not isinstance(start, int)
                        or isinstance(end, bool)
                        or not isinstance(end, int)
                        or start < 0
                        or end < start
                        or end >= TIME_STEPS
                    ):
                        invalid_anomaly_intervals += 1
                    affected = anomalies.get("affected_kpis")
                    if (
                        not isinstance(affected, list)
                        or not affected
                        or any(name not in KPI_NAMES for name in affected)
                    ):
                        invalid_affected_kpi_annotations += 1
                if not mapping_valid:
                    mapping_mismatches += 1
                    if isinstance(sample_id, str) and sample_id in selected_ids:
                        selected_mismatches.append(
                            {
                                "sample_id": sample_id,
                                "source_type": raw_type,
                                "metadata_label": label,
                                "expected_label": expected_label,
                            }
                        )
                if not isinstance(sample_id, str):
                    continue
                if sample_id in selected_ids:
                    if sample_id in selected_metadata:
                        duplicate_ids += 1
                    else:
                        selected_metadata[sample_id] = metadata
                    if csv_labels.get(sample_id) != label:
                        selected_mismatches.append(
                            {
                                "sample_id": sample_id,
                                "csv_label": csv_labels.get(sample_id),
                                "metadata_label": label,
                            }
                        )
    except (OSError, EOFError) as exc:
        raise ValueError(f"Could not read processed metadata {metadata_path}: {exc}") from exc

    return {
        "record_count": row_count,
        "class_distribution": {
            label: int(full_class_counts.get(label, 0)) for label in TAXONOMY_LABELS
        },
        "selected_samples_expected": len(selected_ids),
        "selected_samples_matched": len(selected_metadata),
        "selected_sample_ids_missing": sorted(selected_ids - set(selected_metadata)),
        "selected_label_mismatches": selected_mismatches[:100],
        "duplicate_selected_metadata_ids": duplicate_ids,
        "label_mapping_mismatches": mapping_mismatches,
        "invalid_anomaly_interval_count": invalid_anomaly_intervals,
        "invalid_affected_kpi_annotation_count": invalid_affected_kpi_annotations,
        "anomaly_mapped_label_count": sum(
            count for label, count in full_class_counts.items() if label != "Normal"
        ),
        "normal_label_count": int(full_class_counts.get("Normal", 0)),
        "selected_temporal_annotation_count": sum(
            item.get("label") != "Normal"
            and _has_valid_interval(item.get("anomalies"))
            for item in selected_metadata.values()
        ),
        "selected_affected_kpi_annotation_count": sum(
            item.get("label") != "Normal"
            and _has_valid_affected_kpis(item.get("anomalies"))
            for item in selected_metadata.values()
        ),
        "selected_metadata": selected_metadata,
    }


def _timestamp_schedule_mismatches(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    selected_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    if not {"entity_id", "timestamp"}.issubset(frame.columns):
        return ["CSV is missing entity_id or timestamp."]
    mismatches: list[str] = []
    for entity_id, positions in frame.groupby("entity_id", sort=False, dropna=False).indices.items():
        key = str(entity_id)
        metadata = selected_metadata.get(key)
        if metadata is None or len(positions) != TIME_STEPS:
            continue
        try:
            start = pd.Timestamp(metadata["start_time"])
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
            sampling_rate = float(metadata["sampling_rate"])
            actual_times = timestamps.iloc[positions]
            if not math.isfinite(sampling_rate) or sampling_rate <= 0 or actual_times.isna().any():
                raise ValueError("invalid timestamp or sampling rate")
            for timestep, actual in enumerate(actual_times):
                expected = start + pd.to_timedelta(timestep / sampling_rate, unit="s")
                if abs((actual - expected).total_seconds()) > 0.000001:
                    raise ValueError("timestamp does not match source schedule")
        except (KeyError, TypeError, ValueError, OverflowError):
            mismatches.append(key)
    return mismatches


def _has_valid_interval(anomalies: Any) -> bool:
    if not isinstance(anomalies, dict):
        return False
    duration = anomalies.get("anomaly_duration")
    if not isinstance(duration, dict):
        return False
    start, end = duration.get("start"), duration.get("end")
    return (
        not isinstance(start, bool)
        and isinstance(start, int)
        and not isinstance(end, bool)
        and isinstance(end, int)
        and 0 <= start <= end < TIME_STEPS
    )


def _has_valid_affected_kpis(anomalies: Any) -> bool:
    if not isinstance(anomalies, dict):
        return False
    affected = anomalies.get("affected_kpis")
    return (
        isinstance(affected, list)
        and bool(affected)
        and all(isinstance(name, str) and name in KPI_NAMES for name in affected)
    )


def _outlier_statistics(values: pd.DataFrame) -> dict[str, dict[str, int | float | None]]:
    result: dict[str, dict[str, int | float | None]] = {}
    for name in values.columns:
        series = values[name].dropna()
        if series.empty:
            result[name] = {"count": 0, "percent": 0.0, "lower_bound": None, "upper_bound": None}
            continue
        q1, q3 = (float(item) for item in series.quantile([0.25, 0.75]))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        result[name] = {
            "count": count,
            "percent": round(100 * count / len(series), 4),
            "lower_bound": lower,
            "upper_bound": upper,
        }
    return result


def _json_matrix(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {
        str(row_name): {
            str(column_name): (
                float(value) if pd.notna(value) and math.isfinite(float(value)) else None
            )
            for column_name, value in row.items()
        }
        for row_name, row in frame.iterrows()
    }


def _create_figures(
    frame: pd.DataFrame,
    numeric_values: pd.DataFrame,
    labels: pd.Series,
    class_counts: Counter[str],
    outliers: dict[str, dict[str, int | float | None]],
    correlation: pd.DataFrame,
    output_directory: Path,
) -> None:
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})

    fig, axis = plt.subplots(figsize=(13, 6))
    names = list(TAXONOMY_LABELS)
    axis.bar(names, [class_counts.get(name, 0) for name in names], color="#2878b5")
    axis.set(title="Observed historical samples by label", ylabel="Samples", xlabel="Anomaly type")
    axis.tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    _save_figure(fig, output_directory / "anomaly_class_distribution.png")

    missing_counts = {
        name: int(frame[name].astype(str).str.strip().eq("").sum()) if name in frame else 0
        for name in (*NUMERIC_KPIS, "UL_Protocol", "DL_Protocol")
    }
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(list(missing_counts), list(missing_counts.values()), color="#d98132")
    axis.set(title="Missing values by KPI (valid empty protocol strings excluded)", ylabel="Missing values", xlabel="KPI")
    axis.tick_params(axis="x", labelrotation=55)
    if not any(missing_counts.values()):
        axis.text(0.5, 0.8, "No missing KPI values observed", transform=axis.transAxes, ha="center")
        axis.set_ylim(0, 1)
    fig.tight_layout()
    _save_figure(fig, output_directory / "missing_value_distribution.png")

    fig, axis = plt.subplots(figsize=(13, 11))
    matrix = correlation.to_numpy(dtype=float)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation.columns)), labels=correlation.columns, rotation=65, ha="right")
    axis.set_yticks(range(len(correlation.index)), labels=correlation.index)
    axis.set_title("Pearson correlation across numerical KPI rows")
    fig.colorbar(image, ax=axis, label="Correlation")
    fig.tight_layout()
    _save_figure(fig, output_directory / "kpi_correlation_heatmap.png")

    numeric_names = list(numeric_values.columns)
    binary_labels = np.where(labels.to_numpy() == "Normal", "Normal", "Anomalous")
    fig, axes = plt.subplots(4, 4, figsize=(16, 15))
    for index, (axis, name) in enumerate(zip(axes.flat, numeric_names)):
        series = numeric_values[name].to_numpy(dtype=float)
        normal = series[binary_labels == "Normal"]
        anomalous = series[binary_labels == "Anomalous"]
        axis.boxplot(
            [normal[np.isfinite(normal)], anomalous[np.isfinite(anomalous)]],
            tick_labels=["Normal", "Anomalous"],
            showfliers=False,
        )
        axis.set_title(name, fontsize=9)
        axis.tick_params(axis="x", labelrotation=25)
        axis.tick_params(axis="y", labelsize=7)
    fig.suptitle("Numerical KPI distributions by observed binary label", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_directory / "normal_vs_anomalous_kpi_distributions.png")

    fig, axis = plt.subplots(figsize=(13, 6))
    outlier_names = list(numeric_values.columns)
    outlier_rates = [float(outliers[name]["percent"] or 0) for name in outlier_names]
    axis.bar(outlier_names, outlier_rates, color="#8357a8")
    axis.set(title="Tukey outlier rate by numerical KPI (1.5 × IQR)", ylabel="Outlier rows (%)", xlabel="KPI")
    axis.tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    _save_figure(fig, output_directory / "kpi_outlier_distributions.png")


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _read_historical_export_report(project_root: Path) -> dict[str, Any] | None:
    path = project_root / "data" / "reports" / "telecomts" / "historical_csv_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset_revision": report.get("dataset_revision"),
        "total_source_records_aligned": report.get("total_source_records_aligned"),
        "total_valid_samples": report.get("total_valid_samples"),
        "total_selected_samples": report.get("total_selected_samples"),
        "samples_per_class_selected": report.get("samples_per_class_selected"),
        "invalid_samples_skipped": report.get("invalid_samples_skipped"),
        "sampling_method": report.get("sampling_method"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    quality_rows = [
        "| KPI | Missing | Non-numeric | Non-finite | Unique values | Tukey outliers | Outlier rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, quality in report["numeric_kpi_quality"].items():
        outlier = report["numeric_outlier_statistics_tukey_1_5_iqr"][name]
        quality_rows.append(
            f"| {name} | {quality['missing']} | {quality['nonnumeric']} | "
            f"{quality['nonfinite']} | {quality['unique_values']} | "
            f"{outlier['count']} | {outlier['percent']:.4f}% |"
        )
    class_rows = [
        "| Category | Samples | Rows |",
        "|---|---:|---:|",
    ]
    for label in TAXONOMY_LABELS:
        class_rows.append(
            f"| {label} | {report['sample_class_distribution'][label]} | "
            f"{report['row_class_distribution'][label]} |"
        )
    checks = [
        f"- **{'PASS' if passed else 'FAIL'}** `{name}`"
        for name, passed in report["checks"].items()
    ]
    figures = [f"- `{Path(name).name}`" for name in report["figures"]]
    content = "\n".join(
        [
            "# Historical TelecomTS CSV Quality Report",
            "",
            f"- Dataset revision: `{report['dataset_revision']}`",
            f"- CSV: `{report['csv_path']}`",
            f"- SHA-256: `{report['csv_sha256']}`",
            f"- Shape: {report['row_count']:,} rows × {report['column_count']} columns",
            f"- Entities: {report['sample_count']:,}; expected {TIME_STEPS} rows per entity",
            f"- Size: {report['file_size_mib']:.3f} MiB",
            f"- Overall validation: **{'PASS' if report['overall_validation_passed'] else 'FAIL'}**",
            "",
            "## Schema and integrity",
            "",
            f"- Canonical columns and ordering: {'pass' if report['columns_match_canonical_order'] else 'fail'}",
            f"- Observed rows per entity: `{report['observed_rows_per_sample']}`",
            f"- Wrong-length entities: {report['wrong_length_sample_count']}",
            f"- Entities with inconsistent labels: {report['inconsistent_label_sample_count']}",
            f"- Invalid timestamps: {report['invalid_timestamp_rows']}",
            f"- Duplicate complete rows: {report['duplicate_full_rows']}",
            f"- Duplicate `(entity_id, timestamp)` rows: {report['duplicate_entity_timestamp_rows']}",
            f"- Source metadata label mapping mismatches: {report['original_metadata_audit']['label_mapping_mismatches']}",
            f"- Anomalous samples with valid temporal annotations: {report['original_metadata_audit']['selected_temporal_annotation_count']}",
            f"- Anomalous samples with valid affected-KPI annotations: {report['original_metadata_audit']['selected_affected_kpi_annotation_count']}",
            f"- Source schedule mismatches: {len(report['timestamp_schedule_mismatch_samples'])}",
            "",
            "## Class distribution",
            "",
            *class_rows,
            "",
            "## Numerical KPI quality and outliers",
            "",
            *quality_rows,
            "",
            "Outliers are identified descriptively using the Tukey 1.5×IQR rule; they are not automatically invalid measurements.",
            "",
            "## Categorical KPI values",
            "",
            f"- `UL_Protocol`: `{json.dumps(report['categorical_kpi_quality']['UL_Protocol']['observed_counts'], ensure_ascii=False)}`",
            f"- `DL_Protocol`: `{json.dumps(report['categorical_kpi_quality']['DL_Protocol']['observed_counts'], ensure_ascii=False)}`",
            "",
            "## Leakage and interpretation limits",
            "",
            "- `anomaly_type` is a target label; exclude it from model features.",
            "- Keep all 128 rows for an `entity_id` in one group split. A row-random split leaks the same sample across partitions.",
            "- `timestamp` and `entity_id` are identifiers/time metadata, not KPI features.",
            "- `layer=ran_du` is assigned for dashboard compatibility and is not original TelecomTS layer ground truth.",
            "- This capped, stratified CSV is not representative of TelecomTS class prevalence; use the complete sequence dataset for training.",
            "- KPI associations and outlier flags are descriptive, not causal.",
            "",
            "## Figures",
            "",
            *figures,
            "",
            "## Validation checks",
            "",
            *checks,
            "",
            "The original CSV was read-only during this audit.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    report = audit_historical_csv(
        args.project_root,
        revision=args.revision,
        csv_path=args.csv,
        metadata_path=args.metadata,
        output_directory=args.output_directory,
    )
    print(
        f"Historical CSV audit {'passed' if report['overall_validation_passed'] else 'failed'}: "
        f"{report['row_count']:,} rows, {report['sample_count']:,} entities."
    )
    print(f"Report: {args.output_directory or args.project_root / 'reports'}")
    if not report["overall_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
