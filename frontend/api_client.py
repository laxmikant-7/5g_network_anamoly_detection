"""HTTP client for the research API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from frontend.settings import deployment_setting


class ApiClientError(Exception):
    """The backend could not be queried."""


def fetch_health(base_url: str, timeout: float = 2.0) -> dict[str, Any]:
    """Call ``GET /health``.

    Args:
        base_url: API origin, for example ``http://127.0.0.1:8000``.
        timeout: Socket timeout in seconds.

    Returns:
        The decoded JSON object.

    Raises:
        ApiClientError: If the request fails or the body is not a JSON object.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ApiClientError("API base URL must be a non-empty string.")
    url = base_url.rstrip("/") + "/health"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ApiClientError(f"Health check failed with status {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ApiClientError("Could not reach the backend health endpoint.") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiClientError("Health check returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiClientError("Health check returned an unexpected payload.")
    return payload


def fetch_model_info(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Call ``GET /model-info`` and return the saved model configuration."""
    return _get_json(base_url, "/model-info", "Model information", timeout)


def predict_sample(
    base_url: str,
    sample: list[list[float | str | None]],
    timeout: float = 10.0,
    *,
    kpi_names: list[str] | None = None,
) -> dict[str, Any]:
    """Submit one 128-by-18 sample to the backend inference pipeline."""
    return _post_sample(
        base_url,
        "/predict",
        sample,
        "Prediction",
        timeout,
        kpi_names=kpi_names,
    )


def analyze_sample(
    base_url: str,
    sample: list[list[float | str | None]],
    timeout: float = 15.0,
    *,
    sampling_rate_hz: float = 10.0,
    kpi_names: list[str] | None = None,
) -> dict[str, Any]:
    """Submit one sequence for complete backend diagnosis and localization."""
    return _post_sample(
        base_url,
        "/analyze",
        sample,
        "Analysis",
        timeout,
        sampling_rate_hz=sampling_rate_hz,
        kpi_names=kpi_names,
    )


def _get_json(
    base_url: str,
    path: str,
    label: str,
    timeout: float,
) -> dict[str, Any]:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ApiClientError("API base URL must be a non-empty string.")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers=_auth_headers(base_url),
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ApiClientError(f"{label} request failed with status {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ApiClientError(f"Could not reach the backend {label.lower()} endpoint.") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiClientError(f"{label} endpoint returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiClientError(f"{label} endpoint returned an unexpected payload.")
    return payload


def _post_sample(
    base_url: str,
    path: str,
    sample: list[list[float | str | None]],
    label: str,
    timeout: float,
    *,
    sampling_rate_hz: float | None = None,
    kpi_names: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ApiClientError("API base URL must be a non-empty string.")
    payload: dict[str, Any] = {"sample": sample}
    if kpi_names is not None:
        payload["kpi_names"] = kpi_names
    if sampling_rate_hz is not None:
        payload["sampling_rate_hz"] = sampling_rate_hz
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **_auth_headers(base_url)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            payload = None
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str):
            message = detail
        elif isinstance(detail, list):
            message = "; ".join(str(item) for item in detail)
        else:
            message = f"{label} failed with status {exc.code}."
        raise ApiClientError(message) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ApiClientError(f"Could not reach the backend {label.lower()} endpoint.") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiClientError(f"{label} endpoint returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiClientError(f"{label} endpoint returned an unexpected payload.")
    return payload


def _auth_headers(base_url: str) -> dict[str, str]:
    api_key = deployment_setting("CL5G_API_KEY")
    if api_key is None:
        return {}
    configured_url = deployment_setting("CL5G_API_BASE_URL")
    if configured_url is None or base_url.rstrip("/") != configured_url.rstrip("/"):
        raise ApiClientError(
            "Authenticated requests must use the configured CL5G_API_BASE_URL."
        )
    parsed = urlsplit(base_url)
    is_local_http = (
        parsed.scheme == "http"
        and parsed.hostname is not None
        and parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    )
    if (
        (parsed.scheme != "https" and not is_local_http)
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ApiClientError(
            "Authenticated API requests require HTTPS or local loopback HTTP."
        )
    return {"X-API-Key": api_key}
