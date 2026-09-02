"""Standard information-retrieval metrics for judged evaluation sets."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
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
        _normalize_identifier(identifier): 1.0
        for identifier in relevant_ids
        if _normalize_identifier(identifier)
    }

    return evaluate_retrieval(ranked_ids, relevant)


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    judgments: Mapping[str, int | float] | Iterable[object],
) -> RetrievalMetrics:
    """Score one ranked list against binary or graded relevance judgments.

    ``judgments`` may be a mapping of paper ID to grade, an iterable of IDs,
    or an iterable of objects exposing ``paper_id`` and ``relevance``.  IDs
    are normalized and duplicate retrieved IDs count only once.
    """

    relevant = _relevance_map(judgments)

    if not relevant:
        raise ValueError("at least one judged relevant paper is required")

    ranked = _unique_normalized_ids(retrieved_ids)

    positive = {identifier for identifier, grade in relevant.items() if grade > 0}
    return RetrievalMetrics(
        recall_at_10=_recall_at(ranked, positive, 10),
        recall_at_50=_recall_at(ranked, positive, 50),
        mrr=_reciprocal_rank(ranked, positive),
        ndcg_at_10=_ndcg_at(ranked, relevant, 10),
    )


def _relevance_map(judgments: Mapping[str, int | float] | Iterable[object]) -> dict[str, float]:
    if isinstance(judgments, Mapping):
        pairs = judgments.items()
    else:
        pairs = []
        for item in judgments:
            if isinstance(item, str):
                pairs.append((item, 1.0))
            else:
                paper_id = getattr(item, "paper_id", None)
                relevance = getattr(item, "relevance", None)
                if paper_id is None and isinstance(item, Mapping):
                    paper_id = item.get("paper_id")
                    relevance = item.get("relevance", 1.0)
                if paper_id is None:
                    raise TypeError("judgments must contain paper_id and relevance")
                pairs.append((paper_id, relevance if relevance is not None else 1.0))

    result: dict[str, float] = {}
    for identifier, relevance in pairs:
        normalized = _normalize_identifier(str(identifier))
        if not normalized:
            continue
        try:
            grade = float(relevance)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid relevance grade for {identifier!r}") from exc
        if grade < 0:
            raise ValueError("relevance grades must be non-negative")
        result[normalized] = max(result.get(normalized, 0.0), grade)
    if not result or not any(value > 0 for value in result.values()):
        raise ValueError("at least one judged relevant paper is required")
    return result


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
    relevant: Mapping[str, float],
    k: int,
) -> float:
    """Return graded normalized discounted cumulative gain at top-k."""

    dcg = sum(
        (2.0 ** relevant[identifier] - 1.0) / math.log2(rank + 1)
        for rank, identifier in enumerate(ranked[:k], start=1)
        if identifier in relevant and relevant[identifier] > 0
    )

    idcg = sum(
        (2.0 ** grade - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(
            sorted((grade for grade in relevant.values() if grade > 0), reverse=True)[:k],
            start=1,
        )
    )

    return dcg / idcg if idcg else 0.0
