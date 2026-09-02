"""Strict, provider-independent models for judged evaluation data and reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["well_studied", "uncertain", "promising_gap"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class RetrievalJudgment(StrictModel):
    paper_id: str = Field(min_length=1)
    relevance: int = Field(ge=0)


class RetrievalEvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    relevant_papers: list[RetrievalJudgment] = Field(min_length=1)


class ExtractionEvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = ""
    gold: dict[str, list[str]]


class VerificationEvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    expected_label: Label
    known_counterexample_ids: list[str] = Field(default_factory=list)
    expected_candidate_labels: dict[str, Label] = Field(default_factory=dict)


class DeduplicationEvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    gold_duplicate_pairs: list[list[str]] = Field(default_factory=list)


class FieldMetrics(StrictModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)
    exact_accuracy: float | None = Field(default=None, ge=0, le=1)


class ExtractionMetrics(StrictModel):
    per_field: dict[str, FieldMetrics] = Field(default_factory=dict)
    macro: FieldMetrics | None = None
    micro: FieldMetrics | None = None


class RetrievalAggregateMetrics(StrictModel):
    cases: int = Field(ge=0)
    recall_at_10: float | None = Field(default=None, ge=0, le=1)
    recall_at_50: float | None = Field(default=None, ge=0, le=1)
    mrr: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_10: float | None = Field(default=None, ge=0, le=1)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


class DeduplicationMetrics(StrictModel):
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    gold_pairs: int = Field(ge=0)
    predicted_pairs: int = Field(ge=0)


class AttributionMetrics(StrictModel):
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)
    supported_claim_rate: float = Field(ge=0, le=1)
    unsupported_claim_rate: float = Field(ge=0, le=1)
    attribution_accuracy: float = Field(ge=0, le=1)


class VerificationMetrics(StrictModel):
    cases_total: int = Field(ge=0)
    cases_scored: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    per_label: dict[str, FieldMetrics] = Field(default_factory=dict)
    counterexample_cases: int = Field(ge=0)
    counterexamples_discovered: int = Field(ge=0)
    counterexample_discovery_rate: float | None = Field(default=None, ge=0, le=1)
    false_promising_gap_count: int = Field(ge=0)
    false_promising_gap_rate: float | None = Field(default=None, ge=0, le=1)
    candidate_accuracy: float | None = Field(default=None, ge=0, le=1)


class ModelPricing(StrictModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


class PerformanceMetrics(StrictModel):
    cases_total: int = Field(ge=0, default=0)
    cases_completed: int = Field(ge=0, default=0)
    cases_failed: int = Field(ge=0, default=0)
    cache_mode: Literal["cold", "warm", "unknown"] = "unknown"
    total_seconds: float | None = Field(default=None, ge=0)
    latency: dict[str, float | None] = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    request_counts: dict[str, int] = Field(default_factory=dict)
    token_usage: dict[str, int] | Literal["unavailable"] = "unavailable"
    estimated_cost: float | None = Field(default=None, ge=0)
    cache_hit_rates: dict[str, float] = Field(default_factory=dict)
    work_counts: dict[str, int] = Field(default_factory=dict)


class EvaluationFailure(StrictModel):
    case_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error: str = Field(min_length=1)


class EvaluationReport(StrictModel):
    schema_version: str = "m7-v1"
    dataset_version: str = "unspecified"
    timestamp: str
    cases_total: int = Field(ge=0, default=0)
    cases_completed: int = Field(ge=0, default=0)
    cases_failed: int = Field(ge=0, default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval: RetrievalAggregateMetrics | None = None
    deduplication: DeduplicationMetrics | None = None
    extraction: ExtractionMetrics | None = None
    attribution: AttributionMetrics | None = None
    verification: VerificationMetrics | None = None
    performance: PerformanceMetrics | None = None
    failures: list[EvaluationFailure] = Field(default_factory=list)
