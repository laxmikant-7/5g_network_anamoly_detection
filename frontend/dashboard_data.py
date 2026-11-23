"""Data loading, KPI-sequence validation, and evaluation-data helpers."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from cl5g.data.telecomts import (
    ALLOWED_PROTOCOL_VALUES,
    CATEGORICAL_KPIS,
    KPI_NAMES,
    NUMERIC_KPIS,
)
from cl5g.pipeline.config import AppConfig

SEQUENCE_ROWS = 128
SEQUENCE_CHANNELS = 18
EVALUATION_COLUMNS = {"actual", "predicted"}


def parse_kpi_sequence(filename: str, content: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse KPI uploads; CSV headers must exactly match the trained channel order."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(content), keep_default_na=False)
            if frame.shape != (SEQUENCE_ROWS, SEQUENCE_CHANNELS):
                raise ValueError(
                    f"CSV must contain exactly {SEQUENCE_ROWS} rows and "
                    f"{SEQUENCE_CHANNELS} KPI columns; received {frame.shape}."
                )
            channel_names = [str(column) for column in frame.columns]
            if tuple(channel_names) != KPI_NAMES:
                raise ValueError(
                    "CSV KPI headers must match the configured names and order exactly: "
                    + ", ".join(KPI_NAMES)
                )
            raw_values = frame.to_numpy(dtype=object)
        elif suffix == ".json":
            payload = json.loads(content.decode("utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("sample")
            raw_values = np.asarray(payload, dtype=object)
            channel_names = [f"Channel {index + 1}" for index in range(SEQUENCE_CHANNELS)]
        else:
            raise ValueError("Upload a .csv or .json sequence file.")
    except (UnicodeError, json.JSONDecodeError, pd.errors.ParserError, TypeError) as exc:
        raise ValueError(f"Could not parse the uploaded sequence: {exc}") from exc
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Could not parse KPI values: {exc}") from exc

    if raw_values.shape != (SEQUENCE_ROWS, SEQUENCE_CHANNELS):
        raise ValueError(
            f"Sequence must have shape ({SEQUENCE_ROWS}, {SEQUENCE_CHANNELS}); "
            f"received {raw_values.shape}."
        )
    category_indexes = {KPI_NAMES.index(name) for name in CATEGORICAL_KPIS}
    values = np.empty(raw_values.shape, dtype=object)
    has_categories = False
    for row_index in range(SEQUENCE_ROWS):
        for channel_index in range(SEQUENCE_CHANNELS):
            value = raw_values[row_index, channel_index]
            if channel_index in category_indexes:
                if value is None:
                    has_categories = True
                    values[row_index, channel_index] = None
                elif isinstance(value, str) and value in ALLOWED_PROTOCOL_VALUES:
                    has_categories = True
                    values[row_index, channel_index] = value
                else:
                    raise ValueError(
                        f"Categorical KPI '{KPI_NAMES[channel_index]}' at time step "
                        f"{row_index} must be a supported protocol string or null."
                    )
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Numeric KPI '{KPI_NAMES[channel_index]}' at time step "
                    f"{row_index} must be numeric."
                ) from exc
            if not np.isfinite(numeric_value):
                raise ValueError(
                    f"Numeric KPI '{KPI_NAMES[channel_index]}' at time step "
                    f"{row_index} must be finite."
                )
            values[row_index, channel_index] = numeric_value
    return (values if has_categories else values.astype(float)), channel_names


def load_historical_csvs(
    config: AppConfig,
) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    """Read available CSV datasets from raw and processed data directories."""
    loaded: list[tuple[str, pd.DataFrame]] = []
    errors: list[str] = []
    for relative_directory in (config.paths.raw, config.paths.processed):
        directory = config.resolve(relative_directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.csv")):
            try:
                loaded.append((path.name, pd.read_csv(path, keep_default_na=False)))
            except (OSError, UnicodeError, pd.errors.ParserError) as exc:
                errors.append(f"{path.name}: {exc}")
    return loaded, errors


def numeric_kpi_columns(frame: pd.DataFrame, config: AppConfig) -> list[str]:
    """Return real numeric columns after excluding identifiers and labels."""
    excluded = {
        config.schema.timestamp_column,
        config.schema.entity_column,
        config.schema.layer_column,
        config.schema.label_column,
    }
    configured = [
        column
        for column in config.schema.numeric_columns
        if column in frame.columns and column not in excluded
    ]
    if configured:
        return configured
    dataset_numeric = [name for name in NUMERIC_KPIS if name in frame.columns]
    if dataset_numeric:
        return dataset_numeric
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]


def evaluation_results(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    predicted_column: str,
    probability_column: str | None,
    normal_label: str = "normal",
) -> dict[str, Any]:
    """Calculate evaluation plots and reports from supplied ground-truth results."""
    if actual_column == predicted_column:
        raise ValueError("Ground-truth and predicted-label columns must be different.")
    if probability_column in {actual_column, predicted_column}:
        raise ValueError("The probability column must be different from the label columns.")
    missing = [
        name
        for name in (actual_column, predicted_column)
        if name not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing evaluation column(s): {', '.join(missing)}.")
    results = frame[[actual_column, predicted_column]].dropna()
    if results.empty:
        raise ValueError("Evaluation file has no rows with both actual and predicted labels.")

    actual = results[actual_column].astype(str)
    predicted = results[predicted_column].astype(str)
    labels = sorted(set(actual) | set(predicted))
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    precision = recall = thresholds = None
    if probability_column is not None:
        if probability_column not in frame.columns:
            raise ValueError(f"Probability column '{probability_column}' is missing.")
        probability_rows = frame.loc[results.index, probability_column]
        try:
            scores = probability_rows.astype(float).to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError("Anomaly probabilities must be numeric.") from exc
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Anomaly probabilities must be finite values from 0 to 1.")
        binary_actual = (actual.str.lower() != normal_label.lower()).astype(int)
        if binary_actual.nunique() > 1:
            precision, recall, thresholds = precision_recall_curve(binary_actual, scores)
    return {
        "labels": labels,
        "confusion_matrix": matrix,
        "classification_report": report,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "rows_evaluated": int(len(results)),
    }


def load_saved_evaluation_reports(
    project_root: Path,
    *,
    dataset_revision: str | None,
    model_versions: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load only held-out reports matching the currently served model versions."""
    if not isinstance(dataset_revision, str) or re.fullmatch(
        r"[0-9a-f]{40}", dataset_revision
    ) is None:
        return {}, []
    if not isinstance(model_versions, dict):
        return {}, []

    report_directory = (
        project_root
        / "data"
        / "reports"
        / "telecomts"
        / dataset_revision
    )
    tasks = {
        "binary_detection": ("detection", "binary_detection"),
        "multiclass_diagnosis": ("diagnosis", "multiclass_diagnosis"),
    }
    reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for report_name, (version_key, directory_name) in tasks.items():
        expected_version = model_versions.get(version_key)
        if not isinstance(expected_version, (str, int, float)):
            continue
        candidates = sorted(
            (report_directory / directory_name).glob("seed-*/final_test_report.json")
        )
        matching: list[tuple[Path, dict[str, Any]]] = []
        for path in candidates:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            if not isinstance(report, dict):
                errors.append(f"{path}: expected a JSON object.")
                continue
            if (
                report.get("dataset_id") == "AliMaatouk/TelecomTS"
                and report.get("dataset_revision") == dataset_revision
                and str(report.get("model_version")) == str(expected_version)
                and isinstance(report.get("metrics"), dict)
            ):
                matching.append((path, report))
        if len(matching) > 1:
            errors.append(
                f"Multiple {report_name} reports match model version "
                f"{expected_version!r}; refusing to choose ambiguously."
            )
        elif matching:
            path, report = matching[0]
            report["report_path"] = str(path.resolve())
            reports[report_name] = report
    return reports, errors
