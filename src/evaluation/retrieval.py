"""Retrieval and pairwise deduplication evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .metrics import RetrievalMetrics, evaluate_retrieval


def evaluate_retrieval_case(retrieved_ids: Sequence[str], judgments: Iterable[object]) -> RetrievalMetrics:
    return evaluate_retrieval(retrieved_ids, judgments)


def normalize_pair(pair: Sequence[str]) -> tuple[str, str]:
    if len(pair) != 2:
        raise ValueError("duplicate relationships must contain exactly two paper IDs")
    values = tuple(" ".join(str(item).split()).casefold() for item in pair)
    if not all(values) or values[0] == values[1]:
        raise ValueError("duplicate relationships require two distinct non-empty IDs")
    return tuple(sorted(values))  # type: ignore[return-value]


def duplicate_pairs(pairs: Iterable[Sequence[str]]) -> set[tuple[str, str]]:
    return {normalize_pair(pair) for pair in pairs}


def evaluate_deduplication(gold_pairs: Iterable[Sequence[str]], predicted_pairs: Iterable[Sequence[str]]) -> dict[str, float | int]:
    gold, predicted = duplicate_pairs(gold_pairs), duplicate_pairs(predicted_pairs)
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "gold_pairs": len(gold), "predicted_pairs": len(predicted)}


pairwise_deduplication_metrics = evaluate_deduplication
