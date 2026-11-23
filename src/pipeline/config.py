"""Load ``config.yaml`` into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cl5g.common.exceptions import ConfigurationError
from cl5g.common.paths import find_project_root, resolve_project_path
from cl5g.data.schema import SchemaConfig

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_SPLIT_STRATEGIES = ("time_ordered", "source_file_grouped")


@dataclass(frozen=True)
class PathConfig:
    """Relative directories from the project configuration."""

    raw: str
    processed: str
    sample: str
    artifacts: str
    logs: str


@dataclass(frozen=True)
class SplitConfig:
    """Fractions and strategy for leakage-aware data partitioning."""

    strategy: str
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class TaskConfig:
    """Settings for one supervised task."""

    enabled: bool
    estimator: str | None
    target: str
    artifact: str


@dataclass(frozen=True)
class InferenceConfig:
    """Artifact paths and attribution settings for the inference pipeline."""

    detection_artifact: str
    preprocessing_artifact: str
    feature_config_artifact: str
    metadata_artifact: str
    temporal_localization_artifact: str | None
    attribution_artifact: str | None
    attribution_top_k: int


@dataclass(frozen=True)
class AppConfig:
    """Fully checked project configuration."""

    project_name: str
    seed: int
    paths: PathConfig
    schema: SchemaConfig
    split: SplitConfig
    inference: InferenceConfig
    localization: TaskConfig
    diagnosis: TaskConfig
    api_host: str
    api_port: int
    log_level: str
    project_root: Path

    def resolve(self, relative: str) -> Path:
        """Resolve a configured relative path against the project root."""
        return resolve_project_path(self.project_root, relative)


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate the project configuration.

    Args:
        config_path: Optional path to a YAML file. When omitted, the loader
            searches for ``config.yaml`` from the project root.

    Raises:
        ConfigurationError: If the file is missing, unreadable, or invalid.
    """
    if config_path is None:
        root = find_project_root()
        path = root / "config.yaml"
    else:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise ConfigurationError("Config file does not exist.")
        root = path.resolve().parent
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError("Config file could not be read.") from exc
    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, dict):
        raise ConfigurationError("Config file must contain a YAML mapping.")
    return _parse_config(loaded, root)


def _parse_config(raw: dict[str, Any], root: Path) -> AppConfig:
    _exact_keys(
        raw,
        {
            "project",
            "paths",
            "schema",
            "split",
            "inference",
            "localization",
            "diagnosis",
            "api",
            "logging",
        },
        "config",
    )
    project = _mapping(_require(raw, "project", "config"), "project")
    paths = _parse_paths(_mapping(_require(raw, "paths", "config"), "paths"))
    schema = _parse_schema(_mapping(_require(raw, "schema", "config"), "schema"))
    split = _parse_split(_mapping(_require(raw, "split", "config"), "split"))
    inference = _parse_inference(
        _mapping(_require(raw, "inference", "config"), "inference")
    )
    localization = _parse_task(
        _mapping(_require(raw, "localization", "config"), "localization"),
        "localization",
        schema.layer_column,
    )
    diagnosis = _parse_task(
        _mapping(_require(raw, "diagnosis", "config"), "diagnosis"),
        "diagnosis",
        schema.label_column,
    )
    api = _mapping(_require(raw, "api", "config"), "api")
    logging_section = _mapping(_require(raw, "logging", "config"), "logging")
    _exact_keys(api, {"host", "port"}, "api")
    _exact_keys(logging_section, {"level"}, "logging")
    config = AppConfig(
        project_name=_text(_require(project, "name", "project"), "project.name"),
        seed=_seed(_require(project, "seed", "project")),
        paths=paths,
        schema=schema,
        split=split,
        inference=inference,
        localization=localization,
        diagnosis=diagnosis,
        api_host=_text(_require(api, "host", "api"), "api.host"),
        api_port=_port(_require(api, "port", "api")),
        log_level=_log_level(_require(logging_section, "level", "logging")),
        project_root=root.resolve(),
    )
    for relative in _configured_paths(config):
        config.resolve(relative)
    return config


def _parse_paths(raw: dict[str, Any]) -> PathConfig:
    _exact_keys(raw, {"raw", "processed", "sample", "artifacts", "logs"}, "paths")
    return PathConfig(
        raw=_relative_path(raw["raw"], "paths.raw"),
        processed=_relative_path(raw["processed"], "paths.processed"),
        sample=_relative_path(raw["sample"], "paths.sample"),
        artifacts=_relative_path(raw["artifacts"], "paths.artifacts"),
        logs=_relative_path(raw["logs"], "paths.logs"),
    )


def _parse_schema(raw: dict[str, Any]) -> SchemaConfig:
    _exact_keys(
        raw,
        {
            "timestamp_column",
            "entity_column",
            "layer_column",
            "label_column",
            "required_columns",
            "numeric_columns",
            "layers",
            "anomaly_types",
        },
        "schema",
    )
    timestamp_column = _text(raw["timestamp_column"], "schema.timestamp_column")
    entity_column = _text(raw["entity_column"], "schema.entity_column")
    layer_column = _text(raw["layer_column"], "schema.layer_column")
    label_column = _text(raw["label_column"], "schema.label_column")
    required = _string_tuple(raw["required_columns"], "schema.required_columns")
    numeric = _string_tuple(raw["numeric_columns"], "schema.numeric_columns", allow_empty=True)
    layers = _string_tuple(raw["layers"], "schema.layers")
    anomaly_types = _string_tuple(raw["anomaly_types"], "schema.anomaly_types")
    identity = {timestamp_column, entity_column, layer_column}
    if not identity.issubset(set(required)):
        raise ConfigurationError(
            "schema.required_columns must include the timestamp, entity, and layer columns."
        )
    if label_column in required:
        raise ConfigurationError(
            "schema.label_column must stay optional so inference rows can omit it."
        )
    overlap = set(numeric) & (identity | {label_column})
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ConfigurationError(f"schema.numeric_columns overlap identifier columns: {names}.")
    return SchemaConfig(
        timestamp_column=timestamp_column,
        entity_column=entity_column,
        layer_column=layer_column,
        label_column=label_column,
        required_columns=required,
        numeric_columns=numeric,
        layers=layers,
        anomaly_types=anomaly_types,
    )


def _parse_split(raw: dict[str, Any]) -> SplitConfig:
    _exact_keys(raw, {"strategy", "validation_fraction", "test_fraction"}, "split")
    strategy = _text(raw["strategy"], "split.strategy")
    if strategy not in _SPLIT_STRATEGIES:
        allowed = ", ".join(_SPLIT_STRATEGIES)
        raise ConfigurationError(f"split.strategy must be one of: {allowed}.")
    validation = _fraction(raw["validation_fraction"], "split.validation_fraction")
    test = _fraction(raw["test_fraction"], "split.test_fraction")
    if validation + test >= 1:
        raise ConfigurationError("split fractions must sum to less than 1.")
    return SplitConfig(strategy=strategy, validation_fraction=validation, test_fraction=test)


def _parse_task(raw: dict[str, Any], name: str, expected_target: str) -> TaskConfig:
    _exact_keys(raw, {"enabled", "estimator", "target", "artifact"}, name)
    target = _text(raw["target"], f"{name}.target")
    if target != expected_target:
        raise ConfigurationError(
            f"{name}.target must match the schema column '{expected_target}'."
        )
    return TaskConfig(
        enabled=_boolean(raw["enabled"], f"{name}.enabled"),
        estimator=_optional_text(raw["estimator"], f"{name}.estimator"),
        target=target,
        artifact=_relative_path(raw["artifact"], f"{name}.artifact"),
    )


def _parse_inference(raw: dict[str, Any]) -> InferenceConfig:
    _exact_keys(
        raw,
        {
            "detection_artifact",
            "preprocessing_artifact",
            "feature_config_artifact",
            "metadata_artifact",
            "temporal_localization_artifact",
            "attribution_artifact",
            "attribution_top_k",
        },
        "inference",
    )
    return InferenceConfig(
        detection_artifact=_relative_path(
            _require(raw, "detection_artifact", "inference"),
            "inference.detection_artifact",
        ),
        preprocessing_artifact=_relative_path(
            _require(raw, "preprocessing_artifact", "inference"),
            "inference.preprocessing_artifact",
        ),
        feature_config_artifact=_relative_path(
            _require(raw, "feature_config_artifact", "inference"),
            "inference.feature_config_artifact",
        ),
        metadata_artifact=_relative_path(
            _require(raw, "metadata_artifact", "inference"),
            "inference.metadata_artifact",
        ),
        temporal_localization_artifact=_optional_relative_path(
            _require(raw, "temporal_localization_artifact", "inference"),
            "inference.temporal_localization_artifact",
        ),
        attribution_artifact=_optional_relative_path(
            _require(raw, "attribution_artifact", "inference"),
            "inference.attribution_artifact",
        ),
        attribution_top_k=_positive_integer(
            _require(raw, "attribution_top_k", "inference"),
            "inference.attribution_top_k",
        ),
    )


def _configured_paths(config: AppConfig) -> tuple[str, ...]:
    return (
        config.paths.raw,
        config.paths.processed,
        config.paths.sample,
        config.paths.artifacts,
        config.paths.logs,
        config.inference.detection_artifact,
        config.inference.preprocessing_artifact,
        config.inference.feature_config_artifact,
        config.inference.metadata_artifact,
        *(
            (config.inference.temporal_localization_artifact,)
            if config.inference.temporal_localization_artifact is not None
            else ()
        ),
        *(
            (config.inference.attribution_artifact,)
            if config.inference.attribution_artifact is not None
            else ()
        ),
        config.localization.artifact,
        config.diagnosis.artifact,
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a mapping.")
    return value


def _require(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{label} is missing required key '{key}'.")
    return mapping[key]


def _exact_keys(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigurationError(f"{label} contains unknown key(s): {names}.")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    candidate = Path(text)
    if candidate.is_absolute() or candidate.anchor or ".." in candidate.parts:
        raise ConfigurationError(f"{label} must be a relative path inside the project.")
    return text.replace("\\", "/")


def _optional_relative_path(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _relative_path(value, label)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{label} must be a positive integer.")
    return value


def _string_tuple(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{label} must be a list of strings.")
    if not value and not allow_empty:
        raise ConfigurationError(f"{label} must not be empty.")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(f"{label} must contain non-empty strings.")
        items.append(item.strip())
    if len(items) != len(set(items)):
        raise ConfigurationError(f"{label} contains duplicate values.")
    return tuple(items)


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("project.seed must be an integer.")
    if value < 0:
        raise ConfigurationError("project.seed must be zero or a positive integer.")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} must be true or false.")
    return value


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a number.")
    number = float(value)
    if number <= 0 or number >= 1:
        raise ConfigurationError(f"{label} must be greater than 0 and less than 1.")
    return number


def _port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("api.port must be an integer.")
    if value < 1 or value > 65535:
        raise ConfigurationError("api.port must be between 1 and 65535.")
    return value


def _log_level(value: Any) -> str:
    text = _text(value, "logging.level").upper()
    if text not in _LOG_LEVELS:
        allowed = ", ".join(_LOG_LEVELS)
        raise ConfigurationError(f"logging.level must be one of: {allowed}.")
    return text
