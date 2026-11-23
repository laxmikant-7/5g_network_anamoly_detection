"""Logging setup for the ``cl5g`` logger tree."""

from __future__ import annotations

import logging
from pathlib import Path

from cl5g.common.exceptions import ConfigurationError

LOGGER_NAME = "cl5g"
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def configure_logging(level: str, log_file: Path | None = None) -> logging.Logger:
    """Attach stream and optional file handlers to the project logger.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, or CRITICAL.
        log_file: Optional file path. Parent directories are created.

    Raises:
        ConfigurationError: If ``level`` is not recognized.
    """
    if level not in _LEVELS:
        raise ConfigurationError(
            f"logging.level must be one of {', '.join(_LEVELS)}. Received '{level}'."
        )
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``cl5g`` namespace."""
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
