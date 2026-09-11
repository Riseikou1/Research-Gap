"""Strict public request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from src.persistence.models import AnalysisHistoryItem, AnalysisRecord, AnalysisStatus
from src.api.safety import public_analysis_result, public_job_error


class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class CreateAnalysisRequest(StrictApiModel):
    research_idea: str = Field(min_length=2, max_length=5000)
    decomposer: Literal["deterministic", "openai"] = "deterministic"
    query_generator: Literal["deterministic", "openai"] = "deterministic"
    paper_limit: int = Field(default=20, ge=1, le=100)
    full_text: bool = False
    mode: Literal["quick", "full"] | None = None


class AnalysisCreated(StrictApiModel):
    analysis_id: str
    status: AnalysisStatus


class AnalysisSummary(StrictApiModel):
    analysis_id: str
    research_idea: str
    status: AnalysisStatus
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    mode: Literal["quick", "full"] = "full"
    stage: str = "preparing"

    @classmethod
    def from_record(cls, record: AnalysisRecord | AnalysisHistoryItem) -> "AnalysisSummary":
        return cls(
            analysis_id=record.analysis_id,
            research_idea=record.research_idea,
            status=record.status,
            created_at=record.created_at,
            completed_at=record.completed_at,
            mode=record.mode,
            stage=record.stage,
        )


class AnalysisDetail(AnalysisSummary):
    started_at: AwareDatetime | None = None
    decomposer: Literal["deterministic", "openai"]
    query_generator: Literal["deterministic", "openai"]
    paper_limit: int
    configuration: dict[str, object]
    result: dict[str, object] | None = None
    error_message: str | None = None
    progress: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_record(cls, record: AnalysisRecord) -> "AnalysisDetail":
        return cls(
            analysis_id=record.analysis_id,
            research_idea=record.research_idea,
            status=record.status,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            decomposer=record.decomposer,
            query_generator=record.query_generator,
            paper_limit=record.paper_limit,
            configuration=record.configuration,
            result=public_analysis_result(record.result) if record.result else None,
            error_message=(
                public_job_error(
                    record.error_message,
                    credit_was_reserved=record.reservation_id is not None,
                )
                if record.error_message
                else None
            ),
            mode=record.mode,
            stage=record.stage,
            progress=record.progress,
        )


class HealthResponse(StrictApiModel):
    status: Literal["ok"] = "ok"
