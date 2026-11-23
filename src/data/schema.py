"""Table contract used by later training and inference code."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cl5g.common.exceptions import SchemaValidationError
from cl5g.common.logging_config import get_logger

SAMPLE_FIXTURE_NAME = "schema_fixture.csv"
_MAX_ROW_MESSAGES = 20
_LOGGER = get_logger("data.schema")


@dataclass(frozen=True)
class SchemaConfig:
    """Column contract and controlled vocabularies loaded from configuration."""

    timestamp_column: str
    entity_column: str
    layer_column: str
    label_column: str
    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    layers: tuple[str, ...]
    anomaly_types: tuple[str, ...]


def validate_records(
    frame: pd.DataFrame,
    schema: SchemaConfig,
    *,
    require_labels: bool,
) -> None:
    """Check a table against the configured contract.

    Args:
        frame: Records to validate. The frame is not modified.
        schema: Contract loaded from configuration.
        require_labels: When true, every row needs a known anomaly type.
            Inference allows the label column to be absent.

    Raises:
        TypeError: If ``frame`` or ``schema`` has the wrong type.
        SchemaValidationError: If the table violates the contract.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(schema, SchemaConfig):
        raise TypeError("schema must be a SchemaConfig.")
    errors = _collect_errors(frame, schema, require_labels=require_labels)
    if errors:
        _LOGGER.error("Schema validation failed with %s error(s).", len(errors))
        raise SchemaValidationError(errors)
    _LOGGER.info("Validated %s row(s).", len(frame))


def _collect_errors(
    frame: pd.DataFrame,
    schema: SchemaConfig,
    *,
    require_labels: bool,
) -> list[str]:
    errors: list[str] = []
    if any(not isinstance(column, str) for column in frame.columns):
        errors.append("Column names must be strings.")
        return errors
    if frame.columns.duplicated().any():
        errors.append("The table has duplicate column names.")
    missing = [column for column in schema.required_columns if column not in frame.columns]
    for column in missing:
        errors.append(f"Missing required column '{column}'.")
    if frame.empty:
        errors.append("The table has no rows.")
    if errors:
        return errors

    errors.extend(_timestamp_errors(frame, schema.timestamp_column))
    errors.extend(_entity_errors(frame, schema.entity_column))
    errors.extend(_vocabulary_errors(frame, schema.layer_column, schema.layers, "layer"))
    errors.extend(_label_errors(frame, schema, require_labels=require_labels))
    errors.extend(_numeric_errors(frame, schema.numeric_columns))
    return _limit_row_messages(errors)


def _timestamp_errors(frame: pd.DataFrame, column: str) -> list[str]:
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    messages: list[str] = []
    for index in frame.index[parsed.isna()]:
        messages.append(f"Row index {index}: invalid timestamp.")
    return messages


def _entity_errors(frame: pd.DataFrame, column: str) -> list[str]:
    values = frame[column]
    text = values.astype(str).str.strip()
    missing = values.isna() | text.eq("") | text.str.lower().eq("nan")
    return [f"Row index {index}: entity_id is empty." for index in frame.index[missing]]


def _vocabulary_errors(
    frame: pd.DataFrame,
    column: str,
    allowed: tuple[str, ...],
    label: str,
) -> list[str]:
    values = frame[column]
    text = values.astype(str).str.strip()
    invalid = values.isna() | ~text.isin(allowed)
    messages: list[str] = []
    for index in frame.index[invalid]:
        observed = text.loc[index]
        messages.append(f"Row index {index}: unknown {label} '{observed}'.")
    return messages


def _label_errors(
    frame: pd.DataFrame,
    schema: SchemaConfig,
    *,
    require_labels: bool,
) -> list[str]:
    column = schema.label_column
    if column not in frame.columns:
        if require_labels:
            return [f"Missing label column '{column}'."]
        return []
    values = frame[column]
    text = values.astype(str).str.strip()
    missing = values.isna() | text.eq("") | text.str.lower().eq("nan")
    messages: list[str] = []
    if require_labels:
        for index in frame.index[missing]:
            messages.append(f"Row index {index}: anomaly type is empty.")
    unknown = ~missing & ~text.isin(schema.anomaly_types)
    for index in frame.index[unknown]:
        messages.append(f"Row index {index}: unknown anomaly type '{text.loc[index]}'.")
    return messages


def _numeric_errors(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    messages: list[str] = []
    for column in columns:
        if column not in frame.columns:
            messages.append(f"Missing numeric column '{column}'.")
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            messages.append(f"Column '{column}' must be numeric.")
    return messages


def _limit_row_messages(errors: list[str]) -> list[str]:
    if len(errors) <= _MAX_ROW_MESSAGES:
        return errors
    hidden = len(errors) - _MAX_ROW_MESSAGES
    return [*errors[:_MAX_ROW_MESSAGES], f"{hidden} additional error(s) were omitted."]
