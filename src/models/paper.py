"""Normalized paper and retrieval-provenance models."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from src.models.query import RetrievalMode, SearchQuery


class RetrievalProvenance(BaseModel):
    """One concrete route by which a provider returned a paper."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    query: SearchQuery
    provider: str = Field(min_length=1, max_length=80)
    mode: RetrievalMode
    retrieved_at: AwareDatetime
    provider_rank: int = Field(ge=1)
    provider_score: float | None = None
    serialized_query: str | None = Field(default=None, max_length=1000)
    fallback_used: bool = False

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


class Paper(BaseModel):
    """Provider-independent paper used by retrieval and ranking."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str | None = None

    authors: list[str] = Field(default_factory=list)

    publication_year: int | None = Field(
        default=None,
        ge=1000,
        le=3000,
    )
    publication_date: date | None = None

    doi: str | None = None
    openalex_id: str | None = None

    citation_count: int = Field(default=0, ge=0)

    source: str | None = None
    url: str | None = None

    provenance: list[RetrievalProvenance] = Field(default_factory=list)

    lexical_raw_score: float | None = None
    semantic_raw_score: float | None = None

    lexical_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    semantic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    constraint_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    final_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    ranking_mode: Literal["hybrid", "lexical_only"] | None = None

    @field_validator(
        "id",
        "title",
        "abstract",
        "doi",
        "openalex_id",
        "source",
        "url",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = " ".join(value.split())
            return normalized or None
        return value

    @field_validator("authors", mode="before")
    @classmethod
    def normalize_authors(cls, value: object) -> object:
        if not isinstance(value, list):
            return value

        result: list[str] = []
        seen: set[str] = set()

        for author in value:
            if not isinstance(author, str):
                continue

            normalized = " ".join(author.split())
            key = normalized.casefold()

            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)

        return result

    @computed_field
    @property
    def matched_queries(self) -> list[str]:
        """Return unique retrieval queries in first-seen order."""

        result: list[str] = []
        seen: set[str] = set()

        for item in self.provenance:
            key = item.query.comparison_key

            if key not in seen:
                seen.add(key)
                result.append(item.query.text)

        return result

    @computed_field
    @property
    def retrieval_modes(self) -> list[str]:
        """Return unique retrieval modes in first-seen order."""

        return list(
            dict.fromkeys(
                item.mode.value
                for item in self.provenance
            )
        )

    @computed_field
    @property
    def retrieved_by(self) -> list[str]:
        """Return unique retrieval providers in first-seen order."""

        result: list[str] = []
        seen: set[str] = set()

        for item in self.provenance:
            key = item.provider.casefold()

            if key not in seen:
                seen.add(key)
                result.append(item.provider)

        return result

    def embedding_text(self) -> str:
        """Return the focused title-and-abstract representation for embeddings."""

        if self.abstract:
            return f"{self.title}\n\n{self.abstract}"

        return self.title

    def to_legacy_dict(self) -> dict[str, object]:
        """Serialize the normalized model using the Milestone 2 paper shape."""

        return {
            "openalex_id": self.openalex_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "year": self.publication_year,
            "doi": self.doi,
            "url": self.url,
            "citation_count": self.citation_count,
            "provenance": [
                {
                    "query": item.query.text,
                    "provider": item.provider,
                    "strategy": item.mode.value,
                    "query_strategy": item.query.strategy,
                    "query_source": item.query.source,
                    "query_provider": item.query.provider,
                    "retrieved_at": item.retrieved_at.isoformat(),
                    "rank": item.provider_rank,
                    "provider_score": item.provider_score,
                }
                for item in self.provenance
            ],
        }
