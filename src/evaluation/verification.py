"""Offline scoring for conservative direct and candidate verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import FieldMetrics, Label, VerificationEvaluationCase, VerificationMetrics

LABELS = ("well_studied", "uncertain", "promising_gap")


def _normalize_ids(values: object) -> set[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        return set()
    return {
        normalized
        for value in values
        if (normalized := " ".join(str(value).split()).casefold())
    }


def _prediction(value: object) -> tuple[str | None, set[str], set[str], dict[str, str]]:
    if isinstance(value, str):
        return value, set(), set(), {}
    if isinstance(value, Mapping):
        label = value.get("label") or value.get("final_label")
        searched_ids = _normalize_ids(value.get("searched_paper_ids", []))
        confirmed_ids = (
            _normalize_ids(value.get("counterexample_paper_ids", []))
            | _normalize_ids(value.get("contradicting_paper_ids", []))
        )
        candidates = value.get("candidate_labels") or {}
        return label if isinstance(label, str) else None, searched_ids, confirmed_ids, dict(candidates)
    label = getattr(value, "label", None) or getattr(value, "final_label", None)
    searched_ids = _normalize_ids(getattr(value, "searched_paper_ids", []))
    confirmed_ids = (
        _normalize_ids(getattr(value, "counterexample_paper_ids", []))
        | _normalize_ids(getattr(value, "contradicting_paper_ids", []))
    )
    return label, searched_ids, confirmed_ids, {}


def counterexample_discovery_rate(known_ids: Sequence[str], discovered_ids: Sequence[str]) -> float | None:
    known = _normalize_ids(known_ids)
    if not known:
        return None
    return len(known & _normalize_ids(discovered_ids)) / len(known)


def counterexample_confirmation_rate(known_ids: Sequence[str], confirmed_ids: Sequence[str]) -> float | None:
    known = _normalize_ids(known_ids)
    if not known:
        return None
    return len(known & _normalize_ids(confirmed_ids)) / len(known)


def evaluate_verification(
    cases: Sequence[VerificationEvaluationCase],
    predictions: Mapping[str, object],
) -> VerificationMetrics:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    scored = 0
    discovered = 0
    counterexample_cases = 0
    known_counterexamples = 0
    confirmed = 0
    false_positive = 0
    well_studied_cases = 0
    candidate_total = candidate_correct = 0
    for case in cases:
        label, searched_ids, confirmed_ids, candidate_labels = _prediction(predictions.get(case.id))
        if label in LABELS:
            matrix[case.expected_label][label] += 1
            scored += 1
        known_ids = _normalize_ids(case.known_counterexample_ids)
        if known_ids:
            counterexample_cases += 1
            known_counterexamples += len(known_ids)
            discovered += len(known_ids & searched_ids)
            confirmed += len(known_ids & confirmed_ids)
        false_positive += int(case.expected_label == "well_studied" and label == "promising_gap")
        well_studied_cases += int(case.expected_label == "well_studied")
        for candidate_id, expected in case.expected_candidate_labels.items():
            candidate_total += 1
            candidate_correct += int(candidate_labels.get(candidate_id) == expected)
    per_label: dict[str, FieldMetrics] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = FieldMetrics(
            precision=p,
            recall=r,
            f1=2 * p * r / (p + r) if p + r else 0.0,
            support=tp + fn,
        )
    return VerificationMetrics(
        cases_total=len(cases), cases_scored=scored,
        accuracy=sum(matrix[x][x] for x in LABELS) / scored if scored else None,
        confusion_matrix=matrix, per_label=per_label,
        counterexample_cases=counterexample_cases,
        known_counterexamples=known_counterexamples,
        counterexamples_discovered=discovered,
        counterexample_discovery_rate=discovered / known_counterexamples if known_counterexamples else None,
        counterexamples_confirmed=confirmed,
        counterexample_confirmation_rate=confirmed / known_counterexamples if known_counterexamples else None,
        false_promising_gap_count=false_positive,
        false_promising_gap_rate=false_positive / well_studied_cases if well_studied_cases else None,
        candidate_accuracy=candidate_correct / candidate_total if candidate_total else None,
    )
