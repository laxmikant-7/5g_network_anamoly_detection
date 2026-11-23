"""Request and response models for the research API."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from cl5g.data.telecomts import ALLOWED_PROTOCOL_VALUES, CATEGORICAL_KPIS, KPI_NAMES

CATEGORICAL_CHANNEL_INDEXES = {
    KPI_NAMES.index(name) for name in CATEGORICAL_KPIS
}


class HealthResponse(BaseModel):
    """Application liveness and validated binary-prediction readiness."""

    status: Literal["ok"] = "ok"
    liveness_status: Literal["alive"] = "alive"
    prediction_ready: bool
    readiness_status: Literal["ready", "missing_artifacts", "invalid_artifacts"]
    readiness_error: str | None = None
    model_loading_status: Literal["loaded", "unavailable"]
    loaded_components: dict[str, bool]
    artifacts_present: dict[str, bool]


class SampleRequest(BaseModel):
    """One 128-step, 18-channel sample; protocol channels may contain strings or null."""

    sample: list[list[float | str | None]]
    kpi_names: list[str] | None = None
    sampling_rate_hz: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def validate_dimensions_and_values(self) -> SampleRequest:
        if self.kpi_names is not None and tuple(self.kpi_names) != KPI_NAMES:
            raise ValueError(
                "kpi_names must match the configured 18-channel KPI order exactly."
            )
        if not math.isfinite(self.sampling_rate_hz):
            raise ValueError("sampling_rate_hz must be finite.")
        if len(self.sample) != 128:
            raise ValueError(f"sample must contain exactly 128 time steps; received {len(self.sample)}.")
        for index, row in enumerate(self.sample):
            if len(row) != 18:
                raise ValueError(
                    f"sample time step {index} must contain exactly 18 KPI values; "
                    f"received {len(row)}."
                )
            for channel_index, value in enumerate(row):
                if value is None and channel_index in CATEGORICAL_CHANNEL_INDEXES:
                    continue
                if isinstance(value, str):
                    if (
                        channel_index not in CATEGORICAL_CHANNEL_INDEXES
                        or value not in ALLOWED_PROTOCOL_VALUES
                    ):
                        raise ValueError(
                            f"sample time step {index} contains an unsupported "
                            "categorical KPI value."
                        )
                elif not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(
                        f"sample time step {index} contains a missing or non-finite "
                        "numeric KPI value."
                    )
        return self


class PredictionResponse(BaseModel):
    """Binary prediction response."""

    status: Literal["normal", "anomaly"]
    anomaly_probability: float = Field(ge=0, le=1)
    model_version: str


class TemporalLocalizationResponse(BaseModel):
    """Inclusive interval with optional time mapping and model score."""

    start: int = Field(ge=0, lt=128)
    end: int = Field(ge=0, lt=128)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, gt=0)
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    localization_score: float | None = Field(default=None, ge=0, le=1)
    score_calibrated: bool | None = None
    method: str | None = None
    affected_kpis: list[str] | None = None

    @model_validator(mode="after")
    def validate_order(self) -> TemporalLocalizationResponse:
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start.")
        return self


class FeatureAttributionResponse(BaseModel):
    """Local model contribution for one engineered feature."""

    feature: str
    value: float
    contribution: float
    absolute_contribution: float
    source_kpi: str


class KPIAttributionResponse(BaseModel):
    """Feature contributions aggregated to an original KPI channel."""

    kpi: str
    importance: float = Field(ge=0)
    signed_contribution: float


class ModelAttributionResponse(BaseModel):
    """Class-specific TreeSHAP explanation for one model stage."""

    model_kind: Literal["binary_detection", "multiclass_diagnosis"]
    method: str
    predicted_class: str | int
    prediction_probability: float = Field(ge=0, le=1)
    baseline_value: float
    output_value: float
    aggregation_method: str
    interpretation: str
    top_features: list[FeatureAttributionResponse]
    top_kpis: list[KPIAttributionResponse]


class AttributionResponse(BaseModel):
    """Separate binary and multiclass contributions for one anomaly result."""

    binary_detection: ModelAttributionResponse
    multiclass_diagnosis: ModelAttributionResponse
    aggregation_method: str | None
    interpretation: str | None


class AnalysisResponse(PredictionResponse):
    """Complete anomaly analysis."""

    model_versions: dict[str, str] | None = None
    predicted_anomaly_type: str | None
    diagnosis_confidence: float | None = Field(default=None, ge=0, le=1)
    diagnosis_probabilities: dict[str, float] | None = None
    important_kpis: list[str] | None
    attributions: AttributionResponse | None = None
    temporal_localization: TemporalLocalizationResponse | None
    explanation: str


class ModelInfoResponse(BaseModel):
    """Model metadata and feature configuration."""

    model_versions: dict[str, str]
    supported_anomaly_categories: list[str]
    feature_configuration: dict[str, Any]
    decision_threshold: float | None = Field(default=None, ge=0, le=1)
    dataset_revision: str | None = None
