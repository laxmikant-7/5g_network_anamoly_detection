"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from cl5g.common.exceptions import ConfigurationError
from cl5g.pipeline.config import AppConfig


def get_config(request: Request) -> AppConfig:
    """Return the configuration stored on the application."""
    config = getattr(request.app.state, "config", None)
    if not isinstance(config, AppConfig):
        raise ConfigurationError("Application config is missing.")
    return config
