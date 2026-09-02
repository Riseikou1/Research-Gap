"""Offline judged-evaluation building blocks."""

from .ablation import AblationVariant
from .dataset import EvaluationDataset, load_cases, load_jsonl, write_jsonl
from .extraction import aggregate_extraction, evaluate_attribution, evaluate_claim_attribution, evaluate_extraction
from .human import AnnotationRecord, aggregate_ratings, export_gap_annotations, write_annotations_csv
from .metrics import RetrievalMetrics, evaluate_ranking, evaluate_retrieval
from .models import (
    AttributionMetrics, DeduplicationEvaluationCase, DeduplicationMetrics, EvaluationFailure, EvaluationReport, ExtractionEvaluationCase,
    ExtractionMetrics, FieldMetrics, ModelPricing, PerformanceMetrics,
    RetrievalAggregateMetrics, RetrievalEvaluationCase, RetrievalJudgment, VerificationEvaluationCase, VerificationMetrics,
)
from .performance import aggregate_performance, cache_hit_rate, performance_from_result
from .retrieval import evaluate_deduplication, normalize_pair, pairwise_deduplication_metrics
from .verification import counterexample_discovery_rate, evaluate_verification


def __getattr__(name: str):
    # Keep ``python -m src.evaluation.runner`` free of runpy's re-import warning
    # while retaining the convenient package-level import.
    if name == "EvaluationRunner":
        from .runner import EvaluationRunner
        return EvaluationRunner
    raise AttributeError(name)

__all__ = [
    "AblationVariant", "AttributionMetrics", "EvaluationDataset", "EvaluationFailure",
    "EvaluationReport", "EvaluationRunner", "ExtractionEvaluationCase", "ExtractionMetrics",
    "DeduplicationEvaluationCase", "DeduplicationMetrics", "FieldMetrics", "ModelPricing", "PerformanceMetrics", "RetrievalAggregateMetrics", "RetrievalEvaluationCase",
    "RetrievalJudgment", "RetrievalMetrics", "VerificationEvaluationCase", "VerificationMetrics",
    "aggregate_extraction", "aggregate_performance", "cache_hit_rate", "counterexample_discovery_rate",
    "AnnotationRecord", "aggregate_ratings", "export_gap_annotations", "write_annotations_csv",
    "evaluate_attribution", "evaluate_claim_attribution", "evaluate_deduplication", "evaluate_extraction", "evaluate_ranking",
    "evaluate_retrieval", "evaluate_verification", "load_cases", "load_jsonl", "normalize_pair",
    "performance_from_result", "pairwise_deduplication_metrics", "write_jsonl",
]
