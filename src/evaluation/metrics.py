"""Standard information-retrieval metrics for judged evaluation sets."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Standard binary-relevance metrics for one ranked retrieval result."""

    recall_at_10: float
    recall_at_50: float
    mrr: float
    ndcg_at_10: float


def evaluate_ranking(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> RetrievalMetrics:
    """Evaluate one binary-relevance ranking without provider assumptions.

    Ranked identifiers are normalized and deduplicated in first-seen order so
    the same paper cannot receive relevance credit multiple times.

    Relevant identifiers are normalized into a set because relevance judgments
    are binary: a paper is either judged relevant or it is not.
    """

    relevant = {
        normalized
        for identifier in relevant_ids
        if (normalized := _normalize_identifier(identifier))
    }

    if not relevant:
        raise ValueError("at least one judged relevant paper is required")

    ranked = _unique_normalized_ids(ranked_ids)

    return RetrievalMetrics(
        recall_at_10=_recall_at(ranked, relevant, 10),
        recall_at_50=_recall_at(ranked, relevant, 50),
        mrr=_reciprocal_rank(ranked, relevant),
        ndcg_at_10=_ndcg_at(ranked, relevant, 10),
    )


def _normalize_identifier(identifier: str) -> str:
    """Normalize an identifier for case-insensitive comparison."""

    return " ".join(identifier.split()).casefold()


def _unique_normalized_ids(identifiers: Sequence[str]) -> list[str]:
    """Normalize and deduplicate identifiers while preserving ranking order."""

    result: list[str] = []
    seen: set[str] = set()

    for identifier in identifiers:
        normalized = _normalize_identifier(identifier)

        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def _recall_at(
    ranked: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Return the fraction of all relevant papers retrieved within top-k."""

    retrieved_relevant = relevant.intersection(ranked[:k])
    return len(retrieved_relevant) / len(relevant)


def _reciprocal_rank(
    ranked: Sequence[str],
    relevant: set[str],
) -> float:
    """Return reciprocal rank of the first relevant retrieved paper."""

    for rank, identifier in enumerate(ranked, start=1):
        if identifier in relevant:
            return 1.0 / rank

    return 0.0


def _ndcg_at(
    ranked: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Return binary normalized discounted cumulative gain at top-k."""

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, identifier in enumerate(ranked[:k], start=1)
        if identifier in relevant
    )

    ideal_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg if idcg else 0.0