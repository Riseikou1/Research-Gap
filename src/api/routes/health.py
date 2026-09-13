"""Local process and database health route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.application.analysis_service import PIPELINE_VERSION
from src.api.models import HealthResponse
from src.persistence.database import DATABASE_ERRORS

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> HealthResponse:
    return HealthResponse(
        pipeline_version=PIPELINE_VERSION, api_schema_version=PIPELINE_VERSION,
        build_version=request.app.state.components.settings.operations.build_version,
    )


@router.get("/health", response_model=HealthResponse)
@router.get("/health/ready", response_model=HealthResponse)
def readiness(request: Request) -> HealthResponse:
    try:
        request.app.state.components.database.check()
        worker = request.app.state.components.runner.diagnostics()
        if not worker.get("available"):
            raise RuntimeError("worker unavailable")
        settings = request.app.state.components.settings
        if settings.web.app_url.startswith("https://") and not settings.openai_api_key:
            raise RuntimeError("required provider unavailable")
    except DATABASE_ERRORS + (RuntimeError,):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready.",
        )
    return HealthResponse(
        pipeline_version=PIPELINE_VERSION,
        api_schema_version=PIPELINE_VERSION,
        build_version=settings.operations.build_version,
    )
