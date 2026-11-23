"""Leakage-safe, reproducible preprocessing for TelecomTS KPI sequences."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cl5g.data.telecomts import (
    ALLOWED_PROTOCOL_VALUES,
    CATEGORICAL_KPIS,
    DEFAULT_REVISION,
    KPI_NAMES,
    NUMERIC_KPIS,
    TIME_STEPS,
)

MISSING_CATEGORY = "__MISSING__"
PROTOCOL_CATEGORIES = ("", "None", "TCP", "UDP", MISSING_CATEGORY)
ENCODED_FEATURE_NAMES = (
    *NUMERIC_KPIS,
    *(
        f"{kpi}={category if category else '<empty>'}"
        for kpi in CATEGORICAL_KPIS
        for category in PROTOCOL_CATEGORIES
    ),
)

LEAKAGE_FIELDS = (
    "label",
    "binary_label",
    "source_anomaly_type",
    "anomalies",
    "labels",
    "anomaly_type",
    "anomaly_present",
    "anomaly_description",
    "anomaly_duration",
    "affected_kpis",
    "zone",
    "application",
    "mobility",
    "congestion",
    "description",
    "start_time",
    "end_time",
    "sampling_rate",
    "sample_id",
    "source_file_id",
    "source_line_number",
    "source_sample_id_present",
)


class TelecomTSPreprocessor:
    """Stateless numeric-preserving and categorical one-hot sequence encoder.

    The fixed protocol vocabulary means there are no dataset-fitted
    transformations. Numeric values remain in their original units; invalid
    or missing numeric values are rejected rather than imputed or scaled.
    """

    def transform_sample(self, sample: Mapping[str, Any]) -> np.ndarray:
        """Encode one processed sample or KPI mapping as a 128-by-26 matrix."""
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping.")
        kpis = sample.get("kpis", sample)
        if not isinstance(kpis, Mapping):
            raise ValueError("sample must contain a KPI mapping.")

        missing = [name for name in KPI_NAMES if name not in kpis]
        extra = [str(name) for name in kpis if name not in KPI_NAMES]
        if missing or extra:
            parts = []
            if missing:
                parts.append("missing KPI(s): " + ", ".join(missing))
            if extra:
                parts.append("unexpected KPI(s): " + ", ".join(extra))
            raise ValueError("; ".join(parts))

        columns: list[list[Any]] = []
        for name in KPI_NAMES:
            values = kpis[name]
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(f"KPI '{name}' must be a sequence.")
            if len(values) != TIME_STEPS:
                raise ValueError(
                    f"KPI '{name}' must contain {TIME_STEPS} values; "
                    f"received {len(values)}."
                )
            columns.append(list(values))

        numeric_columns: list[np.ndarray] = []
        for name in NUMERIC_KPIS:
            values = columns[KPI_NAMES.index(name)]
            normalized: list[float] = []
            for index, value in enumerate(values):
                if value is None:
                    raise ValueError(
                        f"KPI '{name}' at time step {index} is missing."
                    )
                if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                    raise ValueError(
                        f"KPI '{name}' at time step {index} is not numeric."
                    )
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"KPI '{name}' at time step {index} is not finite."
                    )
                normalized.append(numeric_value)
            numeric_columns.append(np.asarray(normalized, dtype=np.float64))

        encoded_categories: list[np.ndarray] = []
        for name in CATEGORICAL_KPIS:
            encoded = np.zeros((TIME_STEPS, len(PROTOCOL_CATEGORIES)), dtype=np.float64)
            values = columns[KPI_NAMES.index(name)]
            for index, value in enumerate(values):
                category = MISSING_CATEGORY if value is None else value
                if not isinstance(category, str):
                    raise ValueError(
                        f"KPI '{name}' at time step {index} must be a string or missing."
                    )
                if category not in ALLOWED_PROTOCOL_VALUES and category != MISSING_CATEGORY:
                    raise ValueError(
                        f"KPI '{name}' at time step {index} has unsupported "
                        f"category {category!r}."
                    )
                encoded[index, PROTOCOL_CATEGORIES.index(category)] = 1.0
            encoded_categories.append(encoded)

        numeric = np.column_stack(numeric_columns)
        return np.concatenate((numeric, *encoded_categories), axis=1)

    def transform(self, sample: Mapping[str, Any] | pd.DataFrame) -> np.ndarray:
        """Transform one mapping or raw-channel DataFrame for shared inference."""
        if isinstance(sample, pd.DataFrame):
            return self.transform_sample(
                {"kpis": {name: sample[name].tolist() for name in sample.columns}}
            )
        return self.transform_sample(sample)


def preprocessing_configuration(
    *,
    revision: str = DEFAULT_REVISION,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Return the versioned preprocessing contract for training and inference."""
    return {
        "preprocessing_version": 1,
        "dataset_id": "AliMaatouk/TelecomTS",
        "dataset_revision": revision,
        "sample_count": sample_count,
        "input": {
            "shape": [TIME_STEPS, len(KPI_NAMES)],
            "kpi_order": list(KPI_NAMES),
            "numeric_kpis": list(NUMERIC_KPIS),
            "categorical_kpis": list(CATEGORICAL_KPIS),
        },
        "output": {
            "shape": [TIME_STEPS, len(ENCODED_FEATURE_NAMES)],
            "feature_order": list(ENCODED_FEATURE_NAMES),
        },
        "numeric_handling": {
            "operation": "preserve raw finite values",
            "missing_values": "reject and report; do not impute",
            "scaling": "none",
        },
        "categorical_handling": {
            "encoding": "one-hot",
            "categories_per_kpi": list(PROTOCOL_CATEGORIES),
            "empty_string_and_literal_None_are_distinct": True,
            "json_null": MISSING_CATEGORY,
            "unknown_categories": "reject and report",
        },
        "excluded_predictors": list(LEAKAGE_FIELDS),
        "fitted_transformations": [],
        "fit_policy": "Stateless encoding; fit any future learned transforms on training data only.",
    }


def write_preprocessing_configuration(
    path: Path,
    *,
    revision: str = DEFAULT_REVISION,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Persist the stateless preprocessing contract atomically as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    configuration = preprocessing_configuration(
        revision=revision,
        sample_count=sample_count,
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
    return configuration
