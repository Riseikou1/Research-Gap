"""Local process and database health route."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request, status

from src.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    try:
        request.app.state.components.database.check()
    except (OSError, sqlite3.DatabaseError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable.",
        )
    return HealthResponse()
