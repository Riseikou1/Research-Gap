"""Typed records stored by the Milestone 8 persistence layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

AnalysisStatus = Literal["pending", "running", "completed", "failed"]
AnalysisMode = Literal["quick", "full"]
OwnerKind = Literal["user", "guest", "local"]


class StrictPersistenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class NewAnalysis(StrictPersistenceModel):
    analysis_id: str = Field(min_length=1, max_length=64)
    research_idea: str = Field(min_length=2, max_length=5000)
    decomposer: Literal["deterministic", "openai"]
    query_generator: Literal["deterministic", "openai"]
    paper_limit: int = Field(ge=1, le=100)
    configuration: dict[str, object]
    owner_kind: OwnerKind | None = None
    owner_id: str | None = None
    mode: AnalysisMode = "full"
    stage: str = "preparing"
    progress: dict[str, object] = Field(default_factory=dict)
    reservation_id: str | None = None


class AnalysisRecord(NewAnalysis):
    status: AnalysisStatus
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    result: dict[str, object] | None = None
    error_message: str | None = Field(default=None, max_length=1000)

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed"}


class AnalysisHistoryItem(StrictPersistenceModel):
    analysis_id: str
    research_idea: str
    status: AnalysisStatus
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    mode: AnalysisMode = "full"
    stage: str = "preparing"


def parse_timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
