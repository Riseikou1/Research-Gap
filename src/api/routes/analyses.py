"""Analysis creation, history, retrieval, and deletion routes."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.models import AnalysisCreated, AnalysisDetail, AnalysisSummary, CreateAnalysisRequest
from src.persistence.models import NewAnalysis

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisCreated, status_code=status.HTTP_201_CREATED)
def create_analysis(payload: CreateAnalysisRequest, request: Request) -> AnalysisCreated:
    components = request.app.state.components
    if payload.paper_limit > components.settings.openalex.max_candidates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "paper_limit cannot exceed configured RESEARCH_GAP_MAX_CANDIDATES "
                f"({components.settings.openalex.max_candidates})"
            ),
        )
    analysis_id = str(uuid4())
    configuration = components.service.configuration_snapshot(
        decomposer=payload.decomposer,
        query_generator=payload.query_generator,
        paper_limit=payload.paper_limit,
    )
    record = components.repository.create(
        NewAnalysis(
            analysis_id=analysis_id,
            research_idea=payload.research_idea,
            decomposer=payload.decomposer,
            query_generator=payload.query_generator,
            paper_limit=payload.paper_limit,
            configuration=configuration,
        )
    )
    try:
        components.runner.submit(analysis_id)
    except Exception:
        components.repository.mark_failed(analysis_id, "Analysis worker is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analysis worker is unavailable.",
        )
    return AnalysisCreated(analysis_id=record.analysis_id, status=record.status)


@router.get("", response_model=list[AnalysisSummary])
def list_analyses(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> list[AnalysisSummary]:
    return [
        AnalysisSummary.from_record(record)
        for record in request.app.state.components.repository.list_recent(limit)
    ]


@router.get("/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str, request: Request) -> AnalysisDetail:
    record = request.app.state.components.repository.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    return AnalysisDetail.from_record(record)


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str, request: Request) -> None:
    components = request.app.state.components
    record = components.repository.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    if not record.terminal and not components.runner.cancel(analysis_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A running analysis cannot be deleted until it finishes.",
        )
    if not components.repository.delete(analysis_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
