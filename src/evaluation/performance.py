"""Aggregation of existing pipeline timings and work-accounting snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median

from .models import ModelPricing, PerformanceMetrics

_REQUEST_KEYS = (
    "retrieval_provider_requests", "openai_decomposition_requests",
    "openai_query_generation_requests", "openai_extraction_requests",
)
_CACHE_KEYS = {
    "retrieval": ("retrieval_cache_hits", "retrieval_cache_misses"),
    "evidence": ("memory_evidence_cache_hits", "persistent_evidence_cache_hits", "new_evidence_extractions"),
    "embedding": ("embedding_cache_hits", "new_embeddings"),
    "planning": ("planning_cache_hits", "openai_decomposition_requests", "openai_query_generation_requests"),
}


def cache_hit_rate(hits: int, requests: int) -> float:
    return hits / requests if requests else 0.0


def _work(result: object) -> Mapping[str, int]:
    return getattr(result, "work_metrics", {}) or {}


def performance_from_result(result: object, *, total_seconds: float | None = None, pricing: ModelPricing | None = None, cache_mode: str = "unknown") -> PerformanceMetrics:
    timings = dict(getattr(result, "stage_timings", {}) or {})
    stage_aliases = {
        "planning": "planning_seconds",
        "initial_retrieval": "retrieval_seconds",
        "ranking_embeddings": "ranking_seconds",
        "evidence_lookup_extraction": "extraction_seconds",
        "direct_verification": "verification_seconds",
        "candidate_verification": "verification_seconds",
    }
    normalized_timings: dict[str, float] = {}
    for key, value in timings.items():
        output_key = stage_aliases.get(key, key if key.endswith("_seconds") else f"{key}_seconds")
        normalized_timings[output_key] = normalized_timings.get(output_key, 0.0) + float(value)
    work = dict(_work(result))
    requests = {key: int(work.get(key, 0)) for key in _REQUEST_KEYS if key in work}
    rates: dict[str, float] = {}
    for name, keys in _CACHE_KEYS.items():
        if name == "evidence":
            hits = sum(work.get(key, 0) for key in keys[:2]); total = hits + work.get(keys[2], 0)
        elif name == "planning":
            hits = work.get(keys[0], 0); total = hits + work.get(keys[1], 0) + work.get(keys[2], 0)
        else:
            hits = work.get(keys[0], 0); total = hits + work.get(keys[1], 0)
        rates[name] = cache_hit_rate(hits, total)
    token_keys = ("input_tokens", "output_tokens", "total_tokens")
    tokens = {key: int(work[key]) for key in token_keys if key in work} or "unavailable"
    cost = None
    if pricing and isinstance(tokens, dict):
        cost = (tokens.get("input_tokens", 0) * pricing.input_per_million + tokens.get("output_tokens", 0) * pricing.output_per_million) / 1_000_000
    if total_seconds is None:
        candidate_total = getattr(result, "total_seconds", None)
        total_seconds = float(candidate_total) if candidate_total is not None else sum(normalized_timings.values()) or None
    latency = {"mean": total_seconds, "median": total_seconds, "min": total_seconds, "max": total_seconds}
    return PerformanceMetrics(
        cases_total=1, cases_completed=1, total_seconds=total_seconds, cache_mode=cache_mode,
        latency=latency,
        stage_seconds=normalized_timings, request_counts=requests, token_usage=tokens,
        estimated_cost=cost, cache_hit_rates=rates, work_counts={k: int(v) for k, v in work.items()},
    )


def aggregate_performance(metrics: Sequence[PerformanceMetrics]) -> PerformanceMetrics:
    if not metrics:
        return PerformanceMetrics()
    seconds = [m.total_seconds for m in metrics if m.total_seconds is not None]
    stage_names = sorted({key for m in metrics for key in m.stage_seconds})
    stage_seconds = {key: sum(m.stage_seconds.get(key, 0.0) for m in metrics) for key in stage_names}
    request_names = sorted({key for m in metrics for key in m.request_counts})
    requests = {key: sum(m.request_counts.get(key, 0) for m in metrics) for key in request_names}
    work_names = sorted({key for m in metrics for key in m.work_counts})
    work = {key: sum(m.work_counts.get(key, 0) for m in metrics) for key in work_names}
    rates = {key: sum(m.cache_hit_rates.get(key, 0.0) for m in metrics) / len(metrics) for key in sorted({k for m in metrics for k in m.cache_hit_rates})}
    token_values = [m.token_usage for m in metrics if isinstance(m.token_usage, dict)]
    tokens: dict[str, int] | str = "unavailable"
    if token_values:
        tokens = {key: sum(item.get(key, 0) for item in token_values) for key in ("input_tokens", "output_tokens", "total_tokens") if any(key in item for item in token_values)}
    return PerformanceMetrics(
        cases_total=sum(m.cases_total for m in metrics), cases_completed=sum(m.cases_completed for m in metrics),
        cases_failed=sum(m.cases_failed for m in metrics), total_seconds=sum(seconds) if seconds else None,
        cache_mode=metrics[0].cache_mode if len({m.cache_mode for m in metrics}) == 1 else "unknown",
        latency=latency_summary(metrics),
        stage_seconds=stage_seconds, request_counts=requests, token_usage=tokens,
        estimated_cost=sum(m.estimated_cost or 0 for m in metrics) if any(m.estimated_cost is not None for m in metrics) else None,
        cache_hit_rates=rates, work_counts=work,
    )


def latency_summary(metrics: Sequence[PerformanceMetrics]) -> dict[str, float | None]:
    values = [m.total_seconds for m in metrics if m.total_seconds is not None]
    return {"mean": sum(values) / len(values) if values else None, "median": median(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None}
