"""Provider-neutral paper-retrieval contracts."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.models.paper import Paper
from src.models.query import RetrievalMode, SearchQuery


class RetrievalError(RuntimeError):
    """Raised when a provider cannot return usable candidates."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when a requested provider feature is not configured."""


class RetrievalRequest(BaseModel):
    """A bounded provider-neutral candidate request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    query: SearchQuery
    mode: RetrievalMode
    limit: int = Field(ge=1, le=100)


class PaperRetriever(Protocol):
    provider_name: str

    def search(self, request: RetrievalRequest) -> list[Paper]: ...
