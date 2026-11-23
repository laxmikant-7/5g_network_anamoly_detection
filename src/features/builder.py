"""Training-facing feature table construction using shared inference logic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cl5g.data.telecomts_preprocessing import TelecomTSPreprocessor
from cl5g.features.extraction import (
    extract_features_batch,
    validate_feature_configuration,
)


def build_features(
    frame: pd.DataFrame,
    feature_configuration: Mapping[str, Any],
    preprocessor: Any | None = None,
) -> pd.DataFrame:
    """Build one ordered feature row per sample, excluding non-KPI metadata.

    The input frame must contain either a ``kpis`` mapping column or one column
    per raw KPI; each cell contains a 128-step sequence. If a transformer is
    supplied, it is applied to every KPI sequence before shared extraction.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    spec = validate_feature_configuration(feature_configuration)
    if frame.empty:
        raise ValueError("frame must contain at least one sample.")
    if preprocessor is None and spec["categorical_kpis"]:
        preprocessor = TelecomTSPreprocessor()

    processed_sequences: list[np.ndarray] = []
    for row_index, row in frame.iterrows():
        if "kpis" in frame.columns:
            raw_kpis = row["kpis"]
            if not isinstance(raw_kpis, Mapping):
                raise ValueError(f"Row {row_index}: 'kpis' must be a mapping.")
            missing = [name for name in spec["kpi_names"] if name not in raw_kpis]
            if missing:
                raise ValueError(
                    f"Row {row_index}: missing KPI(s): " + ", ".join(missing)
                )
            sample = {"kpis": {name: raw_kpis[name] for name in spec["kpi_names"]}}
        else:
            missing = [name for name in spec["kpi_names"] if name not in frame.columns]
            if missing:
                raise ValueError("frame is missing KPI column(s): " + ", ".join(missing))
            sample = {
                "kpis": {name: row[name] for name in spec["kpi_names"]}
            }

        if preprocessor is None:
            try:
                processed = np.column_stack(
                    [
                        np.asarray(sample["kpis"][name], dtype=np.float64)
                        for name in spec["kpi_names"]
                    ]
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Row {row_index}: raw KPI sequences must be numeric."
                ) from exc
        else:
            try:
                raw_frame = pd.DataFrame(
                    {
                        name: sample["kpis"][name]
                        for name in spec["kpi_names"]
                    },
                    columns=spec["kpi_names"],
                )
                processed = preprocessor.transform(raw_frame)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row {row_index}: preprocessing failed: {exc}"
                ) from exc
        processed_sequences.append(np.asarray(processed, dtype=np.float64))
    try:
        batch = np.stack(processed_sequences)
    except ValueError as exc:
        raise ValueError("Processed KPI sequences do not have a consistent shape.") from exc
    result = extract_features_batch(batch, spec)
    result.index = frame.index
    return result
