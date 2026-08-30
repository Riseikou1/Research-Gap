"""Provider-neutral decomposition and query-generation interfaces."""

from typing import Protocol

from src.models.idea import ResearchIdea
from src.models.query import SearchQuery


class QueryDecomposer(Protocol):
    def decompose(self, idea: str) -> ResearchIdea: ...


class QueryGenerator(Protocol):
    def generate(self, idea: ResearchIdea) -> list[SearchQuery]: ...
