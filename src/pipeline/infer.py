"""Command-line entry point for the shared inference pipeline."""

from __future__ import annotations

import json

from cl5g.common.exceptions import ConfigurationError, ProjectError
from cl5g.common.logging_config import configure_logging, get_logger
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.pipeline.inference import run_inference

SAMPLE_INPUT_NAME = "inference_sample.json"
_LOGGER = get_logger("pipeline.infer")


def load_sample(config: AppConfig) -> object:
    """Load one JSON-encoded sample from the configured sample directory."""
    path = config.resolve(config.paths.sample) / SAMPLE_INPUT_NAME
    if not path.is_file():
        raise ConfigurationError(
            f"Inference sample is missing: {SAMPLE_INPUT_NAME} "
            f"(expected under {config.paths.sample})."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Inference sample could not be read: {exc}") from exc


def run() -> int:
    """Run inference for ``data/sample/inference_sample.json``."""
    try:
        config = load_config()
        configure_logging(config.log_level)
        result = run_inference(config, load_sample(config))
    except ProjectError as exc:
        _LOGGER.error("%s", exc)
        return 1
    print(json.dumps(result))
    return 0


def main() -> None:
    """Console entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
