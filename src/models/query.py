"""Provider-independent search-query models and provenance."""

from __future__ import annotations

import unicodedata
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QuerySource = Literal["deterministic", "llm"]


class RetrievalMode(StrEnum):
    """Supported literature-retrieval strategies."""

    BROAD_LEXICAL = "broad_lexical"
    TITLE_ABSTRACT = "title_abstract"
    SEMANTIC = "semantic"


class QueryOrigin(BaseModel):
    """The generator and strategy that produced a query."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    strategy: str = Field(min_length=1, max_length=80)
    source: QuerySource
    provider: str | None = Field(default=None, min_length=1, max_length=80)

    @field_validator("strategy", "provider", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value

    @property
    def comparison_key(self) -> tuple[str, str, str | None]:
        """Return a normalized key used to deduplicate query origins."""

        return (
            self.strategy.casefold(),
            self.source,
            self.provider.casefold() if self.provider else None,
        )


class SearchQuery(BaseModel):
    """A normalized search query with complete generation provenance."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    text: str = Field(min_length=1, max_length=1000)
    strategy: str = Field(min_length=1, max_length=80)
    source: QuerySource
    provider: str | None = Field(default=None, min_length=1, max_length=80)
    origins: list[QueryOrigin] = Field(default_factory=list)

    @field_validator("text", "strategy", "provider", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value

    @model_validator(mode="after")
    def retain_primary_origin(self) -> SearchQuery:
        """Ensure the query's primary provenance is always retained once."""

        primary = QueryOrigin(
            strategy=self.strategy,
            source=self.source,
            provider=self.provider,
        )

        unique: list[QueryOrigin] = []
        seen: set[tuple[str, str, str | None]] = set()

        for origin in (primary, *self.origins):
            key = origin.comparison_key

            if key not in seen:
                seen.add(key)
                unique.append(origin)

        self.origins = unique
        return self

    @property
    def comparison_key(self) -> str:
        """Return the normalized key used for query equivalence."""

        normalized = unicodedata.normalize("NFKC", self.text)
        return normalized.casefold()

    def merged_with(self, other: SearchQuery) -> SearchQuery:
        """Merge provenance from another equivalent search query."""

        if self.comparison_key != other.comparison_key:
            raise ValueError("only equivalent queries can be merged")

        return SearchQuery(
            text=self.text,
            strategy=self.strategy,
            source=self.source,
            provider=self.provider,
            origins=[*self.origins, *other.origins],
        )