"""Shared, ordered feature definitions and calculations for training/inference."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cl5g.data.telecomts import (
    CATEGORICAL_KPIS,
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TIME_STEPS,
)
from cl5g.data.telecomts_preprocessing import (
    ENCODED_FEATURE_NAMES,
    MISSING_CATEGORY,
    PROTOCOL_CATEGORIES,
    TelecomTSPreprocessor,
)

BASE_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "first",
    "last",
)
TEMPORAL_STATISTICS = (
    "slope",
    "first_half_mean",
    "second_half_mean",
    "mean_absolute_difference",
    "root_mean_square",
    "skewness",
    "excess_kurtosis",
    "interquartile_range",
)
DISTRIBUTION_STATISTICS = ("skewness", "excess_kurtosis", "interquartile_range")
SUPPORTED_STATISTICS = frozenset(
    (*BASE_STATISTICS, *TEMPORAL_STATISTICS, "category_frequency")
)

FEATURE_EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "base_statistics": BASE_STATISTICS,
    "base_plus_temporal": (*BASE_STATISTICS, *TEMPORAL_STATISTICS),
    "base_plus_distribution": (*BASE_STATISTICS, *DISTRIBUTION_STATISTICS),
    "base_plus_mean_absolute_difference": (
        *BASE_STATISTICS,
        "mean_absolute_difference",
    ),
    "selected_trend": (*BASE_STATISTICS, "slope", "first_half_mean", "second_half_mean"),
    "selected_shape": (
        *BASE_STATISTICS,
        "mean_absolute_difference",
        "root_mean_square",
        "interquartile_range",
    ),
    "selected_temporal_core": (
        *BASE_STATISTICS,
        "slope",
        "first_half_mean",
        "second_half_mean",
        "mean_absolute_difference",
        "root_mean_square",
    ),
}

_CATEGORY_FEATURE_NAMES = {
    (kpi, category): f"{kpi}={category if category else '<empty>'}"
    for kpi in CATEGORICAL_KPIS
    for category in PROTOCOL_CATEGORIES
}


def validate_feature_configuration(value: object) -> dict[str, Any]:
    """Validate and normalize the feature JSON shared by training and inference."""
    if not isinstance(value, Mapping):
        raise ValueError("Feature configuration must be a JSON object.")
    kpi_names = value.get("kpi_names")
    processed_kpi_names = value.get("processed_kpi_names", kpi_names)
    feature_order = value.get("feature_order")
    feature_definitions = value.get("features")
    categorical_kpis = value.get("categorical_kpis", {})
    if not _valid_names(kpi_names, expected_count=18):
        raise ValueError("Feature configuration must define 18 unique input KPI names.")
    if not _valid_names(processed_kpi_names):
        raise ValueError("processed_kpi_names must contain unique channel names.")
    if (
        not isinstance(feature_order, list)
        or not feature_order
        or any(not isinstance(name, str) or not name.strip() for name in feature_order)
        or len(set(feature_order)) != len(feature_order)
    ):
        raise ValueError("feature_order must contain unique feature names.")
    if not isinstance(feature_definitions, list):
        raise ValueError("features must be a list of feature definitions.")
    if not isinstance(categorical_kpis, Mapping):
        raise ValueError("categorical_kpis must map input KPI names to categories.")

    normalized_categories: dict[str, list[str]] = {}
    for name, categories in categorical_kpis.items():
        if (
            not isinstance(name, str)
            or name not in kpi_names
            or name not in CATEGORICAL_KPIS
            or not isinstance(categories, list)
            or not categories
            or any(not isinstance(category, str) for category in categories)
            or len(set(categories)) != len(categories)
        ):
            raise ValueError(f"Invalid categorical KPI definition for '{name}'.")
        normalized_categories[name] = list(categories)
    if normalized_categories and kpi_names != list(KPI_NAMES):
        raise ValueError(
            "Categorical TelecomTS features require the canonical 18-channel KPI order."
        )
    if normalized_categories and (
        set(normalized_categories) != set(CATEGORICAL_KPIS)
        or any(
            categories != list(PROTOCOL_CATEGORIES)
            for categories in normalized_categories.values()
        )
    ):
        raise ValueError(
            "Categorical TelecomTS features require the complete configured protocol domains."
        )

    feature_by_name: dict[str, dict[str, Any]] = {}
    for definition in feature_definitions:
        if not isinstance(definition, Mapping):
            raise ValueError("Each feature must be an object.")
        name, kpi, statistic = (
            definition.get("name"),
            definition.get("kpi"),
            definition.get("statistic"),
        )
        if not all(
            isinstance(item, str) and item.strip()
            for item in (name, kpi, statistic)
        ):
            raise ValueError("Each feature requires name, kpi, and statistic values.")
        if name in feature_by_name:
            raise ValueError(f"Feature configuration repeats '{name}'.")
        if kpi not in processed_kpi_names:
            raise ValueError(f"Feature '{name}' refers to unknown KPI '{kpi}'.")
        if statistic not in SUPPORTED_STATISTICS:
            raise ValueError(f"Feature '{name}' has unsupported statistic '{statistic}'.")
        normalized = dict(definition)
        source_kpi = definition.get("source_kpi", kpi)
        if not isinstance(source_kpi, str) or source_kpi not in kpi_names:
            raise ValueError(f"Feature '{name}' has an invalid source_kpi.")
        if statistic == "category_frequency":
            category = definition.get("category")
            if (
                source_kpi not in normalized_categories
                or not isinstance(category, str)
                or category not in normalized_categories[source_kpi]
                or kpi != _encoded_category_name(source_kpi, category)
            ):
                raise ValueError(
                    f"Feature '{name}' has an invalid categorical frequency definition."
                )
        feature_by_name[name] = normalized
    if set(feature_order) != set(feature_by_name):
        raise ValueError("feature_order must list every feature exactly once.")
    return {
        "feature_set": value.get("feature_set"),
        "kpi_names": list(kpi_names),
        "processed_kpi_names": list(processed_kpi_names),
        "numeric_kpis": list(value.get("numeric_kpis", [])),
        "categorical_kpis": normalized_categories,
        "feature_order": list(feature_order),
        "feature_by_name": feature_by_name,
    }


def extract_features(
    sequence: object,
    feature_configuration: Mapping[str, Any],
) -> pd.DataFrame:
    """Extract one ordered feature row from a processed numeric sequence."""
    spec = (
        feature_configuration
        if "feature_by_name" in feature_configuration
        else validate_feature_configuration(feature_configuration)
    )
    try:
        values = np.asarray(sequence, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Processed sequence must contain numeric values.") from exc
    expected_shape = (TIME_STEPS, len(spec["processed_kpi_names"]))
    if values.shape != expected_shape:
        raise ValueError(
            f"Processed sequence must have shape {expected_shape}; received {values.shape}."
        )
    return extract_features_batch(values[np.newaxis, :, :], spec)


def extract_features_batch(
    sequences: object,
    feature_configuration: Mapping[str, Any],
) -> pd.DataFrame:
    """Extract ordered feature rows from a batch of processed sequences."""
    spec = (
        feature_configuration
        if "feature_by_name" in feature_configuration
        else validate_feature_configuration(feature_configuration)
    )
    try:
        values = np.asarray(sequences, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Processed sequences must contain numeric values.") from exc
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("Processed sequences must be a non-empty 3D batch.")
    expected_shape = (
        values.shape[0],
        TIME_STEPS,
        len(spec["processed_kpi_names"]),
    )
    if values.shape != expected_shape:
        raise ValueError(
            "Processed sequences must have shape "
            f"(samples, {TIME_STEPS}, {len(spec['processed_kpi_names'])}); "
            f"received {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Processed sequences contain missing or non-finite values.")

    channel_indexes = {
        name: index for index, name in enumerate(spec["processed_kpi_names"])
    }
    for kpi, categories in spec["categorical_kpis"].items():
        indicators = values[
            :,
            :,
            [
                channel_indexes[_encoded_category_name(kpi, category)]
                for category in categories
            ],
        ]
        if not np.isin(indicators, (0.0, 1.0)).all() or not np.all(
            np.sum(indicators, axis=2) == 1.0
        ):
            raise ValueError(
                f"Categorical KPI '{kpi}' must be one-hot encoded at every time step."
            )
    columns: dict[str, np.ndarray] = {}
    for feature_name in spec["feature_order"]:
        definition = spec["feature_by_name"][feature_name]
        series = values[:, :, channel_indexes[definition["kpi"]]]
        statistic = definition["statistic"]
        if statistic == "category_frequency":
            if not np.isin(series, (0.0, 1.0)).all():
                raise ValueError(
                    f"Categorical indicator '{definition['kpi']}' must contain only 0 or 1."
                )
        result = _calculate_statistic_batch(series, statistic)
        if not np.isfinite(result).all():
            raise ValueError(f"Feature '{feature_name}' is non-finite.")
        columns[feature_name] = result
    return pd.DataFrame(columns, columns=spec["feature_order"])


def calculate_statistic(values: np.ndarray, statistic: str) -> float:
    """Calculate one statistic using stable centered/scaled operations."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("A feature statistic requires a non-empty 1D sequence.")
    if not np.isfinite(series).all():
        raise ValueError("Feature input contains missing or non-finite values.")
    return float(_calculate_statistic_batch(series[np.newaxis, :], statistic)[0])


def _calculate_statistic_batch(series: np.ndarray, statistic: str) -> np.ndarray:
    if statistic in {"first_half_mean", "second_half_mean"} and series.shape[1] < 2:
        raise ValueError(f"Statistic '{statistic}' requires at least two time steps.")
    if statistic in {"mean", "first_half_mean", "second_half_mean", "category_frequency"}:
        values = (
            series[:, : series.shape[1] // 2]
            if statistic == "first_half_mean"
            else series[:, series.shape[1] // 2 :]
            if statistic == "second_half_mean"
            else series
        )
        return np.sum(values / values.shape[1], axis=1)
    if statistic == "min":
        return np.min(series, axis=1)
    if statistic == "max":
        return np.max(series, axis=1)
    if statistic == "median":
        return np.median(series, axis=1)
    if statistic == "first":
        return series[:, 0]
    if statistic == "last":
        return series[:, -1]

    scale = np.max(np.abs(series), axis=1)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    normalized = series / safe_scale[:, np.newaxis]
    if statistic == "std":
        centered = normalized - np.mean(normalized, axis=1, keepdims=True)
        return scale * np.sqrt(np.mean(centered**2, axis=1))
    if statistic == "slope":
        centered_time = np.arange(series.shape[1], dtype=np.float64)
        centered_time -= centered_time.mean()
        centered = normalized - np.mean(normalized, axis=1, keepdims=True)
        slope = centered @ centered_time / np.dot(centered_time, centered_time)
        return scale * slope
    if statistic == "mean_absolute_difference":
        return scale * np.mean(np.abs(np.diff(normalized, axis=1)), axis=1)
    if statistic == "root_mean_square":
        return scale * np.sqrt(np.mean(normalized**2, axis=1))
    if statistic in {"skewness", "excess_kurtosis"}:
        centered = normalized - np.mean(normalized, axis=1, keepdims=True)
        centered_scale = np.max(np.abs(centered), axis=1)
        safe_centered_scale = np.where(centered_scale == 0.0, 1.0, centered_scale)
        normalized_centered = centered / safe_centered_scale[:, np.newaxis]
        second_moment = np.mean(normalized_centered**2, axis=1)
        safe_second_moment = np.where(second_moment == 0.0, 1.0, second_moment)
        if statistic == "skewness":
            result = (
                np.mean(normalized_centered**3, axis=1)
                / safe_second_moment**1.5
            )
        else:
            result = (
                np.mean(normalized_centered**4, axis=1)
                / safe_second_moment**2
                - 3.0
            )
        return np.where(centered_scale == 0.0, 0.0, result)
    if statistic == "interquartile_range":
        quartiles = np.quantile(normalized, (0.25, 0.75), axis=1)
        return scale * (quartiles[1] - quartiles[0])
    raise ValueError(f"Unsupported feature statistic '{statistic}'.")


def build_feature_configuration(feature_set: str) -> dict[str, Any]:
    """Create an inference-compatible feature JSON for a named experiment."""
    if feature_set not in FEATURE_EXPERIMENTS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. Choose from: "
            + ", ".join(FEATURE_EXPERIMENTS)
        )
    statistics = FEATURE_EXPERIMENTS[feature_set]
    definitions: list[dict[str, Any]] = []
    for kpi in NUMERIC_KPIS:
        for statistic in statistics:
            definitions.append(
                {
                    "name": f"{kpi}__{statistic}",
                    "kpi": kpi,
                    "statistic": statistic,
                    "feature_group": (
                        "base"
                        if statistic in BASE_STATISTICS
                        else "distribution"
                        if statistic in DISTRIBUTION_STATISTICS
                        else "temporal"
                    ),
                    "source_kpi": kpi,
                }
            )
    for kpi in CATEGORICAL_KPIS:
        for category in PROTOCOL_CATEGORIES:
            definitions.append(
                {
                    "name": f"{kpi}__frequency__{_category_label(category)}",
                    "kpi": _encoded_category_name(kpi, category),
                    "statistic": "category_frequency",
                    "category": category,
                    "source_kpi": kpi,
                    "feature_group": "categorical_frequency",
                }
            )
    configuration = {
        "feature_set": feature_set,
        "kpi_names": list(KPI_NAMES),
        "processed_kpi_names": list(ENCODED_FEATURE_NAMES),
        "numeric_kpis": list(NUMERIC_KPIS),
        "categorical_kpis": {
            kpi: list(PROTOCOL_CATEGORIES) for kpi in CATEGORICAL_KPIS
        },
        "feature_order": [definition["name"] for definition in definitions],
        "features": definitions,
    }
    validate_feature_configuration(configuration)
    return configuration


def feature_dimension_report() -> dict[str, Any]:
    """Describe feature names, order, and dimensions for all planned experiments."""
    feature_sets: dict[str, Any] = {}
    for name, statistics in FEATURE_EXPERIMENTS.items():
        configuration = build_feature_configuration(name)
        numeric_count = len(NUMERIC_KPIS) * len(statistics)
        categorical_count = sum(len(categories) for categories in configuration["categorical_kpis"].values())
        feature_sets[name] = {
            "numeric_statistics": list(statistics),
            "numeric_feature_count": numeric_count,
            "categorical_feature_count": categorical_count,
            "total_feature_count": len(configuration["feature_order"]),
            "feature_order": configuration["feature_order"],
        }
    return {
        "input_shape": [TIME_STEPS, len(KPI_NAMES)],
        "processed_shape": [TIME_STEPS, len(ENCODED_FEATURE_NAMES)],
        "numeric_kpis": list(NUMERIC_KPIS),
        "categorical_kpis": {
            name: list(PROTOCOL_CATEGORIES) for name in CATEGORICAL_KPIS
        },
        "observed_protocol_categories": ["", "None", "TCP", "UDP"],
        "reserved_missing_category": MISSING_CATEGORY,
        "feature_sets": feature_sets,
        "feature_names_follow_feature_order": True,
    }


def write_feature_reports(
    project_root: Path,
    revision: str = DEFAULT_REVISION,
) -> dict[str, Any]:
    """Write all experiment configs, dimensions, and one real-sample example."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a 40-character lowercase hexadecimal commit.")
    root = project_root.resolve()
    processed = root / "data" / "processed" / "telecomts" / revision
    reports = root / "data" / "reports" / "telecomts" / revision
    sequences_path = processed / "sequences.jsonl.gz"
    metadata_path = processed / "metadata.jsonl.gz"
    for path in (sequences_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required TelecomTS input is missing: {path}")

    dimension_report = feature_dimension_report()
    output_directory = reports / "features"
    config_directory = output_directory / "feature_configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    for feature_set in FEATURE_EXPERIMENTS:
        _write_json(
            config_directory / f"{feature_set}.json",
            build_feature_configuration(feature_set),
        )
    _write_json(output_directory / "feature_dimensions.json", dimension_report)

    with (
        gzip.open(sequences_path, "rt", encoding="utf-8") as sequence_stream,
        gzip.open(metadata_path, "rt", encoding="utf-8") as metadata_stream,
    ):
        pair = next(
            zip_longest(sequence_stream, metadata_stream, fillvalue=None),
            None,
        )
    if pair is None or pair[0] is None or pair[1] is None:
        raise ValueError("Processed sequences and metadata are empty or misaligned.")
    sequence = json.loads(pair[0])
    metadata = json.loads(pair[1])
    preprocessor = TelecomTSPreprocessor()
    encoded = preprocessor.transform_sample(sequence)
    example_configuration = build_feature_configuration("base_plus_temporal")
    extracted = extract_features(encoded, example_configuration)
    example = {
        "dataset_revision": revision,
        "sample_id": sequence.get("sample_id"),
        "label_for_analysis_only": metadata.get("label"),
        "feature_set": "base_plus_temporal",
        "input_shape": [TIME_STEPS, len(KPI_NAMES)],
        "processed_shape": list(encoded.shape),
        "feature_count": int(extracted.shape[1]),
        "features": {
            name: float(extracted.iloc[0][name])
            for name in example_configuration["feature_order"]
        },
    }
    _write_json(output_directory / "feature_example.json", example)
    return {"dimensions": dimension_report, "example": example}


def _encoded_category_name(kpi: str, category: str) -> str:
    return f"{kpi}={_category_label(category)}"


def _category_label(category: str) -> str:
    return category if category else "<empty>"


def _valid_names(value: object, *, expected_count: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and (expected_count is None or len(value) == expected_count)
        and bool(value)
        and all(isinstance(name, str) and name.strip() for name in value)
        and len(set(value)) == len(value)
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write TelecomTS feature experiment definitions and dimensions."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    write_feature_reports(args.project_root, args.revision)


if __name__ == "__main__":
    main()
