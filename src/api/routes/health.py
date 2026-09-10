"""Local process and database health route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from src.api.models import HealthResponse
from src.persistence.database import DATABASE_ERRORS

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    try:
        request.app.state.components.database.check()
    except DATABASE_ERRORS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable.",
        )
    return HealthResponse()
