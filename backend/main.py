"""FastAPI application for saved-model 5G anomaly inference."""

from __future__ import annotations

import os
import secrets
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    AnalysisResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SampleRequest,
)
from cl5g.common.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    ProjectError,
    SchemaValidationError,
)
from cl5g.common.logging_config import configure_logging, get_logger
from cl5g.pipeline.config import AppConfig, load_config
from cl5g.pipeline.inference import InferencePipeline, artifact_presence

_LOGGER = get_logger("backend")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create the API and one cached, non-training inference pipeline."""
    resolved = load_config() if config is None else config
    _validate_deployment_security()
    configure_logging(resolved.log_level)
    app = FastAPI(
        title="Cross-Layer 5G Anomaly API",
        version="0.2.0",
        summary="Binary detection and complete analysis from saved model artifacts.",
        description=(
            "Accepts a single 128-time-step, 18-KPI sequence. Models and "
            "preprocessing artifacts are loaded by the shared inference pipeline; "
            "requests never train or retrain models."
        ),
    )
    app.state.config = resolved
    app.state.inference_pipeline = InferencePipeline(resolved)

    cors_origins = [
        origin.strip()
        for origin in os.environ.get("CL5G_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    if os.environ.get("CL5G_ENV", "development").strip().lower() == "production":
        if not cors_origins or "*" in cors_origins:
            raise ConfigurationError(
                "Production requires explicit origins in CL5G_CORS_ORIGINS."
            )
        for origin in cors_origins:
            parsed_origin = urlsplit(origin)
            if (
                parsed_origin.scheme != "https"
                or not parsed_origin.netloc
                or parsed_origin.path
                or parsed_origin.query
                or parsed_origin.fragment
            ):
                raise ConfigurationError(
                    "CL5G_CORS_ORIGINS must contain only HTTPS origins."
                )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials="*" not in cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    _register_exception_handlers(app)
    _register_routes(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SchemaValidationError)
    async def handle_schema_error(_request: Request, exc: SchemaValidationError) -> JSONResponse:
        _LOGGER.warning("Rejected KPI sample: %s", exc)
        return JSONResponse(status_code=422, content={"detail": list(exc.errors)})

    @app.exception_handler(ArtifactNotFoundError)
    async def handle_missing_artifact(
        _request: Request, exc: ArtifactNotFoundError
    ) -> JSONResponse:
        _LOGGER.error("Inference artifact unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def handle_configuration_error(
        _request: Request, exc: ConfigurationError
    ) -> JSONResponse:
        _LOGGER.error("Inference configuration error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(ProjectError)
    async def handle_project_error(_request: Request, exc: ProjectError) -> JSONResponse:
        _LOGGER.exception("Inference request failed: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        _LOGGER.exception("Unhandled backend error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected inference service error occurred."},
        )


def _validate_deployment_security() -> None:
    environment = os.environ.get("CL5G_ENV", "development").strip().lower()
    api_key = os.environ.get("CL5G_API_KEY")
    if environment == "production" and (
        not api_key or len(api_key) < 32 or api_key != api_key.strip()
    ):
        raise ConfigurationError(
            "Production requires CL5G_API_KEY with at least 32 non-whitespace characters."
        )


def _require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = os.environ.get("CL5G_API_KEY")
    if expected is not None and (
        x_api_key is None or not secrets.compare_digest(x_api_key, expected)
    ):
        raise HTTPException(status_code=401, detail="A valid API key is required.")


def _register_routes(app: FastAPI) -> None:
    @app.get("/", tags=["service"], summary="Service index")
    def root() -> dict[str, str]:
        return {
            "service": "cross-layer-5g-anomaly",
            "health": "/health",
            "predict": "/predict",
            "analyze": "/analyze",
            "model_info": "/model-info",
            "docs": "/docs",
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["service"],
        summary="Application liveness and prediction readiness",
        description=(
            "HTTP 200 confirms the API process is alive. Check prediction_ready "
            "and readiness_status separately to confirm compatible required "
            "binary-inference artifacts are present and loadable."
        ),
    )
    def health() -> HealthResponse:
        """Return liveness independently from verified model prediction readiness."""
        pipeline: InferencePipeline = app.state.inference_pipeline
        status = pipeline.loading_status()
        if not status["prediction_ready"]:
            _LOGGER.warning(
                "API is live but prediction readiness is %s: %s",
                status["readiness_status"],
                status["readiness_error"],
            )
        return HealthResponse(**status)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        summary="Predict normal or anomaly",
        responses={
            422: {"description": "The input is not a finite 128-by-18 KPI sequence."},
            503: {"description": "A required model or preprocessing artifact is missing."},
            401: {"description": "A valid X-API-Key header is required when API authentication is enabled."},
        },
        dependencies=[Depends(_require_api_key)],
    )
    def predict(body: SampleRequest) -> dict[str, object]:
        """Return binary detection without requiring diagnosis artifacts."""
        _LOGGER.info("Binary prediction requested.")
        pipeline: InferencePipeline = app.state.inference_pipeline
        return pipeline.predict_binary(body.sample, kpi_names=body.kpi_names)

    @app.post(
        "/analyze",
        response_model=AnalysisResponse,
        response_model_exclude_unset=True,
        tags=["inference"],
        summary="Run complete anomaly analysis",
        responses={
            422: {"description": "The input is not a finite 128-by-18 KPI sequence."},
            503: {"description": "A required model or preprocessing artifact is missing."},
            401: {"description": "A valid X-API-Key header is required when API authentication is enabled."},
        },
        dependencies=[Depends(_require_api_key)],
    )
    def analyze(body: SampleRequest) -> dict[str, object]:
        """Return diagnosis, attribution, and temporal localization when available."""
        _LOGGER.info("Complete anomaly analysis requested.")
        pipeline: InferencePipeline = app.state.inference_pipeline
        return pipeline.predict(
            body.sample,
            sampling_rate_hz=body.sampling_rate_hz,
            kpi_names=body.kpi_names,
        )

    @app.get(
        "/model-info",
        response_model=ModelInfoResponse,
        tags=["models"],
        summary="Model versions and feature configuration",
        responses={503: {"description": "Model metadata or feature configuration is missing."}},
        dependencies=[Depends(_require_api_key)],
    )
    def model_info() -> dict[str, object]:
        """Return configured categories and saved feature/model version metadata."""
        pipeline: InferencePipeline = app.state.inference_pipeline
        return pipeline.model_info()


app = create_app()


def main() -> None:
    """Run the ASGI application with Uvicorn."""
    import uvicorn

    config = app.state.config
    host = os.environ.get("CL5G_API_HOST", config.api_host)
    raw_port = os.environ.get("PORT", os.environ.get("CL5G_API_PORT"))
    try:
        port = config.api_port if raw_port is None else int(raw_port)
    except ValueError as exc:
        raise ConfigurationError("PORT or CL5G_API_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ConfigurationError("PORT or CL5G_API_PORT must be between 1 and 65535.")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
