"""Structured representation of an unstructured research idea."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchIdea(BaseModel):
    """Provider-independent, validated query-decomposition result."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    original_text: str = Field(min_length=1)
    problem: list[str] = Field(default_factory=list)
    population: list[str] = Field(default_factory=list)
    intervention_or_method: list[str] = Field(default_factory=list)
    data_or_modality: list[str] = Field(default_factory=list)
    comparison: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    domain: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    # Surface facet value -> provider-derived canonical concept.  The map is
    # intentionally generic so matching does not need a scientific ontology.
    canonical_facets: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("original_text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.split())
        return value
