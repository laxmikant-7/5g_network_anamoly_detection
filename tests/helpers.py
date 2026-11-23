"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cl5g.data.schema import SchemaConfig
from cl5g.pipeline.config import SplitConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def schema_config(**overrides: object) -> SchemaConfig:
    """Return a valid schema, with optional field replacements."""
    values: dict[str, object] = {
        "timestamp_column": "timestamp",
        "entity_column": "entity_id",
        "layer_column": "layer",
        "label_column": "anomaly_type",
        "required_columns": ("timestamp", "entity_id", "layer"),
        "numeric_columns": (),
        "layers": ("phy", "mac", "rrc"),
        "anomaly_types": ("normal", "congestion", "handover_failure"),
    }
    values.update(overrides)
    return SchemaConfig(**values)  # type: ignore[arg-type]


def split_config() -> SplitConfig:
    """Return the reserved split settings used by unit tests."""
    return SplitConfig(strategy="time_ordered", validation_fraction=0.1, test_fraction=0.2)


def valid_frame(**extra: object) -> pd.DataFrame:
    """Return one row that satisfies ``schema_config``."""
    row: dict[str, object] = {
        "timestamp": "2024-01-01T00:00:00Z",
        "entity_id": "gnb-1",
        "layer": "phy",
        "anomaly_type": "normal",
    }
    row.update(extra)
    return pd.DataFrame([row])
