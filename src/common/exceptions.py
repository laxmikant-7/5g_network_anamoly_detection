"""Project-specific exceptions."""

from __future__ import annotations


class ProjectError(Exception):
    """Base error for expected project failures."""


class ConfigurationError(ProjectError):
    """Raised when configuration is missing or inconsistent."""


class SchemaValidationError(ProjectError):
    """Raised when a table does not match the configured data contract."""

    def __init__(self, errors: list[str]) -> None:
        if not errors:
            raise ValueError("SchemaValidationError requires at least one error.")
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


class ArtifactNotFoundError(ProjectError):
    """Raised when a configured model artifact file is absent."""
