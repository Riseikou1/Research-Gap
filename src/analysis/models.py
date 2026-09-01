"""Provider-independent models for literature analysis and gap verification."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.extraction.evidence import canonical_evidence_key


GapCategory = Literal[
    "population_or_setting",
    "dataset",
    "method",
    "comparison",
    "evaluation",
    "limitation",
    "future_work",
    "contradiction",
    "combination",
]

GapPattern = Literal[
    "underrepresented_population",
    "narrow_dataset_setting",
    "missing_comparison",
    "limited_external_validation",
    "limited_real_world_validation",
    "repeated_limitation",
    "repeated_future_work",
    "comparable_conflict",
    "method_domain_transfer",
    "evaluation_gap",
    "replication_gap",
    "combination_gap",
]

VerificationPattern = GapPattern | Literal["direct_idea_assessment"]

EvidenceRole = Literal[
    "direct_support",
    "contextual_support",
    "potential_contradiction",
    "confirmed_contradiction",
    "confirmed_direct_match",
    "contextual_or_partial_support",
    "potential_match",
]

GapAssessmentLabel = Literal[
    "well_studied",
    "uncertain",
    "promising_gap",
]

VerificationStatus = Literal[
    "not_run",
    "verified",
]

VerificationSource = Literal[
    "deterministic",
    "llm",
]


def canonical_gap_key(value: str) -> str:
    """Create a stable compact identifier from candidate wording."""

    normalized = " ".join(
        re.findall(
            r"[a-z0-9]+",
            value.casefold(),
        )
    )

    return hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()[:16]


def _unique_strings(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for value in values
            if value
        )
    )


class GapEvidence(BaseModel):
    """One provenance-bearing claim used in gap reasoning."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    paper_id: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    study_type: str | None = Field(
        default=None,
        max_length=40,
    )
    role: EvidenceRole = "direct_support"


class LandscapeBasis(BaseModel):
    """One observed literature-landscape feature grounding a candidate."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    dimension: str = Field(
        min_length=1,
        max_length=80,
    )
    value: str = Field(
        min_length=1,
        max_length=300,
    )
    count: int = Field(ge=0)
    total: int = Field(ge=0)
    prevalence: float = Field(
        ge=0.0,
        le=1.0,
    )
    paper_ids: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def normalize_basis(self) -> "LandscapeBasis":
        self.paper_ids = _unique_strings(
            self.paper_ids
        )

        if self.count > self.total:
            raise ValueError(
                "landscape basis count cannot exceed total"
            )

        return self


class VerificationQuery(BaseModel):
    """One bounded query used for direct or counterexample verification."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    candidate_id: str = Field(min_length=1)
    query: str = Field(
        min_length=2,
        max_length=1000,
    )
    pattern_type: VerificationPattern
    strategy: str = Field(
        min_length=1,
        max_length=80,
    )
    source: VerificationSource = "deterministic"


class VerificationFailure(BaseModel):
    """A provider or extraction failure during verification."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    error: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Temporary cross-paper synthesis models
#
# These are still retained because EvidenceSynthesizer and ResearchResult
# currently expose synthesis data. They can be removed together if that
# legacy synthesis path is deleted from the pipeline.
# ---------------------------------------------------------------------------


class GapVerification(BaseModel):
    """Evidence and coverage outcome of verifying one candidate."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    candidate_id: str = Field(min_length=1)

    verification_queries: list[VerificationQuery] = Field(
        default_factory=list,
    )

    searched_paper_ids: list[str] = Field(
        default_factory=list,
    )
    supporting_paper_ids: list[str] = Field(
        default_factory=list,
    )
    potential_contradiction_paper_ids: list[str] = Field(
        default_factory=list,
    )
    contradicting_paper_ids: list[str] = Field(
        default_factory=list,
    )

    evidence: list[GapEvidence] = Field(
        default_factory=list,
    )
    coverage_notes: list[str] = Field(
        default_factory=list,
    )
    failures: list[VerificationFailure] = Field(
        default_factory=list,
    )

    label: GapAssessmentLabel
    reason: str = Field(
        min_length=1,
        max_length=2000,
    )

    @model_validator(mode="after")
    def normalize_verification(self) -> "GapVerification":
        self.searched_paper_ids = _unique_strings(
            self.searched_paper_ids
        )
        self.supporting_paper_ids = _unique_strings(
            self.supporting_paper_ids
        )
        self.potential_contradiction_paper_ids = _unique_strings(
            self.potential_contradiction_paper_ids
        )
        self.contradicting_paper_ids = _unique_strings(
            self.contradicting_paper_ids
        )

        self.evidence = _unique_gap_evidence(
            self.evidence
        )

        return self


class GapCandidate(BaseModel):
    """A cautious, evidence-backed research-gap hypothesis."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    id: str = Field(
        default="",
        max_length=120,
    )

    title: str = Field(
        min_length=1,
        max_length=300,
    )
    description: str = Field(
        min_length=1,
        max_length=1500,
    )
    category: GapCategory
    rationale: str = Field(
        min_length=1,
        max_length=2000,
    )

    supporting_paper_ids: list[str] = Field(
        default_factory=list,
    )
    supporting_evidence: list[GapEvidence] = Field(
        default_factory=list,
    )

    pattern_type: GapPattern = "evaluation_gap"
    landscape_basis: list[LandscapeBasis] = Field(
        default_factory=list,
    )

    verification_queries: list[VerificationQuery] = Field(
        default_factory=list,
    )
    verification: GapVerification | None = None

    final_label: GapAssessmentLabel | None = None

    potentially_contradicting_paper_ids: list[str] = Field(
        default_factory=list,
    )
    contradicting_paper_ids: list[str] = Field(
        default_factory=list,
    )

    potentially_contradicting_evidence: list[GapEvidence] = Field(
        default_factory=list,
    )
    contradicting_evidence: list[GapEvidence] = Field(
        default_factory=list,
    )

    verification_status: VerificationStatus = "not_run"

    # Internal bookkeeping only.
    #
    # These are not novelty probabilities and should not be displayed as
    # scientific confidence. They remain temporarily because existing
    # candidate-generation code still populates them.
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    idea_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    support_count: int = Field(
        default=0,
        ge=0,
    )

    @model_validator(mode="after")
    def normalize_candidate(self) -> "GapCandidate":
        if not self.id:
            self.id = (
                "gap-"
                + canonical_gap_key(
                    f"{self.pattern_type}:{self.title}"
                )
            )

        self.supporting_evidence = _unique_gap_evidence(
            self.supporting_evidence
        )

        evidence_ids = [
            item.paper_id
            for item in self.supporting_evidence
        ]

        self.supporting_paper_ids = _unique_strings(
            [
                *self.supporting_paper_ids,
                *evidence_ids,
            ]
        )

        self.potentially_contradicting_paper_ids = _unique_strings(
            self.potentially_contradicting_paper_ids
        )
        self.contradicting_paper_ids = _unique_strings(
            self.contradicting_paper_ids
        )

        self.potentially_contradicting_evidence = _unique_gap_evidence(
            self.potentially_contradicting_evidence
        )
        self.contradicting_evidence = _unique_gap_evidence(
            self.contradicting_evidence
        )

        self.support_count = len(
            self.supporting_paper_ids
        )

        return self


class IdeaAssessment(BaseModel):
    """Direct verification result for the complete proposed research idea."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    label: GapAssessmentLabel
    rationale: str = Field(
        min_length=1,
        max_length=2000,
    )

    supporting_paper_ids: list[str] = Field(
        default_factory=list,
    )
    supporting_evidence: list[GapEvidence] = Field(
        default_factory=list,
    )

    counterexample_paper_ids: list[str] = Field(
        default_factory=list,
    )
    counterexample_evidence: list[GapEvidence] = Field(
        default_factory=list,
    )

    partial_match_paper_ids: list[str] = Field(
        default_factory=list,
    )
    potential_match_paper_ids: list[str] = Field(
        default_factory=list,
    )

    matched_facets: dict[str, list[str]] = Field(
        default_factory=dict,
    )

    verification_queries: list[VerificationQuery] = Field(
        default_factory=list,
    )
    searched_paper_ids: list[str] = Field(
        default_factory=list,
    )

    coverage_notes: list[str] = Field(
        default_factory=list,
    )
    failures: list[VerificationFailure] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def normalize_assessment(self) -> "IdeaAssessment":
        self.supporting_paper_ids = _unique_strings(
            self.supporting_paper_ids
        )
        self.counterexample_paper_ids = _unique_strings(
            self.counterexample_paper_ids
        )
        self.partial_match_paper_ids = _unique_strings(
            self.partial_match_paper_ids
        )
        self.potential_match_paper_ids = _unique_strings(
            self.potential_match_paper_ids
        )
        self.searched_paper_ids = _unique_strings(
            self.searched_paper_ids
        )

        self.supporting_evidence = _unique_gap_evidence(
            self.supporting_evidence
        )
        self.counterexample_evidence = _unique_gap_evidence(
            self.counterexample_evidence
        )

        self.matched_facets = {
            paper_id: _unique_strings(facets)
            for paper_id, facets in self.matched_facets.items()
            if paper_id
        }

        return self


def _unique_gap_evidence(
    items: list[GapEvidence],
) -> list[GapEvidence]:
    result: list[GapEvidence] = []
    seen: set[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for item in items:
        key = (
            item.paper_id,
            item.evidence_type,
            canonical_evidence_key(
                item.value
            ),
            item.role,
        )

        if key in seen:
            continue

        seen.add(
            key
        )
        result.append(
            item
        )

    return result