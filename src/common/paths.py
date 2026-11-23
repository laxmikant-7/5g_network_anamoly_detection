"""Project-root discovery and relative path checks."""

from __future__ import annotations

import os
from pathlib import Path

from cl5g.common.exceptions import ConfigurationError

_PROJECT_ROOT_ENV = "CL5G_PROJECT_ROOT"


def resolve_project_path(root: Path, relative: str) -> Path:
    """Resolve a configured path and keep it inside ``root``.

    Args:
        root: Repository directory used as the base.
        relative: Path written in configuration. Absolute paths and ``..``
            segments are rejected.

    Returns:
        The resolved path.

    Raises:
        ConfigurationError: If the value is absolute, empty, or escapes ``root``.
    """
    if not isinstance(relative, str) or not relative.strip():
        raise ConfigurationError("A configured path must be a non-empty relative string.")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.anchor or ".." in candidate.parts:
        raise ConfigurationError(
            f"Configured path '{relative}' must be a relative path inside the project."
        )
    root_resolved = root.resolve()
    resolved = (root_resolved / candidate).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ConfigurationError(
            f"Configured path '{relative}' escapes the project root."
        )
    return resolved


def display_path(root: Path, path: Path) -> str:
    """Return a project-relative POSIX path for messages and logs."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking parents and checking an environment variable.

    The root contains ``config.yaml`` and ``src/__init__.py``. ``CL5G_PROJECT_ROOT``
    overrides directory walking when it points at such a root.

    Raises:
        ConfigurationError: If no project root can be identified.
    """
    override = os.environ.get(_PROJECT_ROOT_ENV)
    if override:
        root = Path(override).expanduser()
        if not _is_project_root(root):
            raise ConfigurationError(
                f"{_PROJECT_ROOT_ENV} does not point at a project root "
                "(expected config.yaml and src/__init__.py)."
            )
        return root.resolve()

    seeds: list[Path] = []
    if start is not None:
        seeds.append(start)
    seeds.append(Path.cwd())
    seeds.append(Path(__file__).resolve())
    seen: set[Path] = set()
    for seed in seeds:
        for candidate in (seed, *seed.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if _is_project_root(candidate):
                return candidate.resolve()
    raise ConfigurationError(
        f"Could not find the project root. Set {_PROJECT_ROOT_ENV} to the repository directory."
    )


def _is_project_root(candidate: Path) -> bool:
    return (candidate / "config.yaml").is_file() and (candidate / "src" / "__init__.py").is_file()
