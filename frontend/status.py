"""Status payload shared by the Streamlit page."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from cl5g.common.exceptions import ConfigurationError
from cl5g.pipeline.capabilities import (
    EVALUATION_IMPLEMENTED,
    FEATURES_IMPLEMENTED,
    INFERENCE_IMPLEMENTED,
    TRAINING_IMPLEMENTED,
)
from cl5g.pipeline.config import AppConfig
from cl5g.pipeline.inference import artifact_presence
from frontend.settings import deployment_setting


def api_base_url(config: AppConfig) -> str:
    """Resolve the separate API host from deployment settings or local config."""
    value = deployment_setting("CL5G_API_BASE_URL")
    if value is None:
        return f"http://{config.api_host}:{config.api_port}"
    parsed = urlsplit(value)
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
        raise ConfigurationError(
            "CL5G_API_BASE_URL must be HTTPS (or local loopback HTTP) without "
            "credentials, query parameters, or a fragment."
        )
    return value.rstrip("/")


def build_status(config: AppConfig) -> dict[str, Any]:
    """Describe configured model artifacts without scoring a sample."""
    presence = artifact_presence(config)
    return {
        "project_name": config.project_name,
        "training_implemented": TRAINING_IMPLEMENTED,
        "inference_implemented": INFERENCE_IMPLEMENTED,
        "features_implemented": FEATURES_IMPLEMENTED,
        "evaluation_implemented": EVALUATION_IMPLEMENTED,
        "detection_artifact_present": presence["detection"],
        "preprocessing_artifact_present": presence["preprocessing"],
        "feature_config_artifact_present": presence["feature_config"],
        "metadata_artifact_present": presence["metadata"],
        "localization_artifact_present": presence["localization"],
        "temporal_localization_artifact_present": presence["temporal_localization"],
        "diagnosis_artifact_present": presence["diagnosis"],
        "attribution_artifact_present": presence["attribution"],
        "layers": list(config.schema.layers),
        "anomaly_types": list(config.schema.anomaly_types),
        "api_base_url": api_base_url(config),
    }
