"""Provider-independent structured evidence models."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvidenceSource = Literal["title", "abstract"]
StudyType = Literal["empirical", "review", "survey", "methodological", "other"]


def canonical_evidence_key(value: str) -> str:
    """Return a conservative key for harmless evidence spelling variants."""

    normalized = value.casefold().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


class EvidenceItem(BaseModel):
    """One claim and the source text that directly supports it."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    value: str = Field(min_length=1)
    canonical_value: str | None = None
    evidence_text: str = Field(min_length=1)
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
    study_type: StudyType
    research_objective: EvidenceItem | None = None
    population_or_setting: list[EvidenceItem] = Field(default_factory=list)
    method_or_intervention: list[EvidenceItem] = Field(default_factory=list)
    comparison_or_baseline: list[EvidenceItem] = Field(default_factory=list)
    data_or_modality: list[EvidenceItem] = Field(default_factory=list)
    datasets: list[EvidenceItem] = Field(default_factory=list)
    sample_size: EvidenceItem | None = None
    evaluation_metrics: list[EvidenceItem] = Field(default_factory=list)
    main_findings: list[EvidenceItem] = Field(default_factory=list)
    constraints: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[LimitationEvidence] = Field(default_factory=list)
    future_work: list[EvidenceItem] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_evidence(self) -> "PaperEvidence":
        list_fields = (
            "population_or_setting",
            "method_or_intervention",
            "comparison_or_baseline",
            "data_or_modality",
            "datasets",
            "evaluation_metrics",
            "main_findings",
            "constraints",
            "limitations",
            "future_work",
        )

        for field_name in list_fields:
            setattr(self, field_name, _deduplicate_items(getattr(self, field_name)))

        fields = (
            ("research_objective", self.research_objective),
            ("population_or_setting", self.population_or_setting),
            ("method_or_intervention", self.method_or_intervention),
            ("comparison_or_baseline", self.comparison_or_baseline),
            ("data_or_modality", self.data_or_modality),
            ("datasets", self.datasets),
            ("sample_size", self.sample_size),
            ("evaluation_metrics", self.evaluation_metrics),
            ("main_findings", self.main_findings),
            ("constraints", self.constraints),
            ("limitations", self.limitations),
            ("future_work", self.future_work),
        )

        self.missing_fields = [name for name, value in fields if not value]
        return self

def _deduplicate_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    result: list[EvidenceItem] = []
    seen: set[str] = set()

    for item in items:
        key = canonical_evidence_key(item.canonical_value or item.value)
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result
