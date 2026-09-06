"""Structured Milestone-5 literature-landscape models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperFeatures(BaseModel):
    """Normalized, comparison-oriented features derived from one paper."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    paper_id: str = Field(min_length=1)
    title: str | None = None
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    problems: list[str] = Field(default_factory=list)
    populations_or_settings: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    method_families: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    dataset_types: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    performance_metrics: list[str] = Field(default_factory=list)
    efficiency_metrics: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    study_type: Literal["empirical", "review", "survey", "methodological", "other"] = "other"


class FeatureFrequency(BaseModel):
    """Frequency of a normalized feature across distinct papers."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    dimension: str = Field(min_length=1)
    value: str = Field(min_length=1)
    count: int = Field(ge=0)
    prevalence: float = Field(ge=0.0, le=1.0)
    paper_ids: list[str] = Field(default_factory=list)


class CombinationPattern(BaseModel):
    """An observed combination of normalized feature dimensions."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    dimensions: dict[str, str] = Field(min_length=2)
    count: int = Field(ge=0)
    prevalence: float = Field(ge=0.0, le=1.0)
    paper_ids: list[str] = Field(default_factory=list)


ConflictStatus = Literal["comparable_conflict", "insufficient_comparability"]


class EvidenceConflict(BaseModel):
    """A conservative conflict assessment for comparable findings."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    paper_ids: list[str] = Field(min_length=2)
    topic: str = Field(min_length=1)
    status: ConflictStatus
    reason: str = Field(min_length=1)


class LiteratureLandscape(BaseModel):
    """Complete deterministic Milestone-5 result for an analyzed paper set."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    total_papers: int = Field(ge=0)
    papers: list[PaperFeatures] = Field(default_factory=list)
    frequencies: list[FeatureFrequency] = Field(default_factory=list)
    combinations: list[CombinationPattern] = Field(default_factory=list)
    missing_field_counts: dict[str, int] = Field(default_factory=dict)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
