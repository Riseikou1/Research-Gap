"""Provider-independent structured evidence models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvidenceSource = Literal["title", "abstract"]


class EvidenceItem(BaseModel):
    """One claim and the source text that supports it."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    value: str = Field(min_length=1)
    evidence_text: str | None = None
    source: EvidenceSource
    confidence: float = Field(ge=0.0, le=1.0)


class LimitationEvidence(EvidenceItem):
    """A limitation explicitly attributed to the paper's authors."""

    author_stated: Literal[True] = True


class PaperEvidence(BaseModel):
    """Structured, provenance-bearing evidence extracted from one paper."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    research_objective: EvidenceItem | None = None
    population_or_setting: list[EvidenceItem] = Field(default_factory=list)
    method_or_intervention: list[EvidenceItem] = Field(default_factory=list)
    comparison_or_baseline: list[EvidenceItem] = Field(default_factory=list)
    datasets: list[EvidenceItem] = Field(default_factory=list)
    sample_size: EvidenceItem | None = None
    evaluation_metrics: list[EvidenceItem] = Field(default_factory=list)
    main_findings: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[LimitationEvidence] = Field(default_factory=list)
    future_work: list[EvidenceItem] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_missing_fields(self) -> "PaperEvidence":
        fields = (
            ("research_objective", self.research_objective),
            ("population_or_setting", self.population_or_setting),
            ("method_or_intervention", self.method_or_intervention),
            ("comparison_or_baseline", self.comparison_or_baseline),
            ("datasets", self.datasets),
            ("sample_size", self.sample_size),
            ("evaluation_metrics", self.evaluation_metrics),
            ("main_findings", self.main_findings),
            ("limitations", self.limitations),
            ("future_work", self.future_work),
        )
        self.missing_fields = [name for name, value in fields if not value]
        return self
