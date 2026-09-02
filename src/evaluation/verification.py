"""Offline scoring for conservative direct and candidate verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import FieldMetrics, Label, VerificationEvaluationCase, VerificationMetrics

LABELS = ("well_studied", "uncertain", "promising_gap")


def _prediction(value: object) -> tuple[str | None, set[str], dict[str, str]]:
    if isinstance(value, str):
        return value, set(), {}
    if isinstance(value, Mapping):
        label = value.get("label") or value.get("final_label")
        ids = value.get("counterexample_paper_ids") or value.get("contradicting_paper_ids") or []
        candidates = value.get("candidate_labels") or {}
        return label if isinstance(label, str) else None, set(str(x) for x in ids), dict(candidates)
    label = getattr(value, "label", None) or getattr(value, "final_label", None)
    ids = getattr(value, "counterexample_paper_ids", None) or getattr(value, "contradicting_paper_ids", None) or []
    return label, set(str(x) for x in ids), {}


def counterexample_discovery_rate(known_ids: Sequence[str], discovered_ids: Sequence[str]) -> float | None:
    if not known_ids:
        return None
    return float(bool(set(known_ids) & set(discovered_ids)))


def evaluate_verification(
    cases: Sequence[VerificationEvaluationCase],
    predictions: Mapping[str, object],
) -> VerificationMetrics:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    scored = 0
    discovered = 0
    counterexample_cases = 0
    false_positive = 0
    well_studied_cases = 0
    candidate_total = candidate_correct = 0
    for case in cases:
        label, ids, candidate_labels = _prediction(predictions.get(case.id))
        if label in LABELS:
            matrix[case.expected_label][label] += 1
            scored += 1
        if case.known_counterexample_ids:
            counterexample_cases += 1
            discovered += int(bool(set(case.known_counterexample_ids) & ids))
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
        counterexamples_discovered=discovered,
        counterexample_discovery_rate=discovered / counterexample_cases if counterexample_cases else None,
        false_promising_gap_count=false_positive,
        false_promising_gap_rate=false_positive / well_studied_cases if well_studied_cases else None,
        candidate_accuracy=candidate_correct / candidate_total if candidate_total else None,
    )
