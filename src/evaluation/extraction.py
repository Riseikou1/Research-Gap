"""Offline structured-evidence and provenance scoring."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.extraction.evidence import EvidenceItem, PaperEvidence

from .models import AttributionMetrics, ExtractionMetrics, FieldMetrics

EVIDENCE_FIELDS = (
    "research_objective", "population_or_setting", "method_or_intervention",
    "comparison_or_baseline", "data_or_modality", "datasets", "sample_size",
    "evaluation_metrics", "main_findings", "constraints", "limitations", "future_work",
)


def normalize_value(value: str) -> str:
    value = value.casefold().replace("-", " ").replace("_", " ")
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def _values(record: PaperEvidence | Mapping[str, object], field: str) -> list[str]:
    value = record.get(field) if isinstance(record, Mapping) else getattr(record, field, None)
    if value is None:
        return []
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    output: list[str] = []
    for item in values:
        if isinstance(item, EvidenceItem):
            text = item.value
        elif isinstance(item, Mapping):
            text = item.get("value")
        else:
            text = item
        if isinstance(text, str) and text.strip():
            output.append(normalize_value(text))
    return list(dict.fromkeys(output))


def _gold_values(gold: Mapping[str, Sequence[str]], field: str) -> list[str]:
    values = gold.get(field, [])
    return list(
        dict.fromkeys(
            normalized
            for value in values
            if (normalized := normalize_value(str(value)))
        )
    )


@dataclass(frozen=True, slots=True)
class _Counts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def active(self) -> bool:
        return bool(self.true_positive or self.false_positive or self.false_negative)


def _counts(predicted: Sequence[str], gold: Sequence[str]) -> _Counts:
    predicted_set, gold_set = set(predicted), set(gold)
    return _Counts(
        true_positive=len(predicted_set & gold_set),
        false_positive=len(predicted_set - gold_set),
        false_negative=len(gold_set - predicted_set),
    )


def _field_metrics(predicted: list[str], gold: list[str]) -> FieldMetrics:
    predicted_set, gold_set = set(predicted), set(gold)
    counts = _counts(predicted, gold)
    tp, fp, fn = counts.true_positive, counts.false_positive, counts.false_negative
    # Empty/empty is an unscored field, not a perfect prediction.  Returning
    # zero here keeps direct per-field output explicit; macro and micro use
    # ``active`` below and therefore exclude it entirely.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FieldMetrics(
        precision=precision, recall=recall, f1=f1, support=len(set(gold)),
        exact_accuracy=(1.0 if predicted_set == gold_set else 0.0) if counts.active else None,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def evaluate_extraction(predicted: PaperEvidence | Mapping[str, object], gold: Mapping[str, Sequence[str]]) -> ExtractionMetrics:
    per_field = {
        field: _field_metrics(_values(predicted, field), _gold_values(gold, field))
        for field in EVIDENCE_FIELDS
    }
    total = _sum_counts(per_field.values())
    micro = _metrics_from_counts(total, exact_accuracy=None)
    nonempty = [metric for metric in per_field.values() if _metric_is_active(metric)]
    macro = None
    if nonempty:
        macro = _average_metrics(nonempty)
    return ExtractionMetrics(per_field=per_field, macro=macro, micro=micro)


def _contains(haystack: str, needle: str) -> bool:
    return normalize_value(needle) in normalize_value(haystack)


def evaluate_attribution(
    evidence: PaperEvidence | Mapping[str, object] | Sequence[EvidenceItem],
    *,
    title: str,
    abstract: str | None,
    human_judgments: Mapping[str, bool] | None = None,
) -> AttributionMetrics:
    """Check evidence spans against their declared title/abstract source.

    Human judgments are keyed by evidence text and override the deterministic
    span check when supplied; canonical values are deliberately ignored.
    """
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, Mapping)):
        items = list(evidence)
    else:
        items = []
        record = evidence
        for field in EVIDENCE_FIELDS:
            value = record.get(field) if isinstance(record, Mapping) else getattr(record, field, None)
            values = value if isinstance(value, list) else ([value] if value is not None else [])
            items.extend(item for item in values if isinstance(item, EvidenceItem) or isinstance(item, Mapping))
    supported = 0
    for item in items:
        if isinstance(item, EvidenceItem):
            span, source = item.evidence_text, item.source
        else:
            span, source = item.get("evidence_text", ""), item.get("source")
        if not isinstance(span, str):
            continue
        if human_judgments and span in human_judgments:
            is_supported = human_judgments[span]
        elif source == "title":
            is_supported = _contains(title, span)
        elif source == "abstract":
            is_supported = bool(abstract) and _contains(abstract or "", span)
        else:
            is_supported = False
        supported += int(is_supported)
    total = len(items)
    unsupported = total - supported
    rate = supported / total if total else 0.0
    return AttributionMetrics(
        total_claims=total, supported_claims=supported, unsupported_claims=unsupported,
        supported_claim_rate=rate, unsupported_claim_rate=unsupported / total if total else 0.0,
        attribution_accuracy=rate,
    )


def evaluate_claim_attribution(*args, **kwargs) -> AttributionMetrics:
    """Compatibility name emphasizing that provenance, not canonical text, is scored."""
    return evaluate_attribution(*args, **kwargs)


def aggregate_extraction(metrics: Sequence[ExtractionMetrics]) -> ExtractionMetrics:
    if not metrics:
        return ExtractionMetrics()
    fields = sorted({field for item in metrics for field in item.per_field})
    per_field: dict[str, FieldMetrics] = {}
    for field in fields:
        values = [item.per_field[field] for item in metrics if field in item.per_field]
        counts = _sum_counts(values)
        exact_values = [x.exact_accuracy for x in values if x.exact_accuracy is not None and _metric_is_active(x)]
        per_field[field] = _metrics_from_counts(
            counts,
            exact_accuracy=sum(exact_values) / len(exact_values) if exact_values else None,
        )
    active = [metric for metric in per_field.values() if _metric_is_active(metric)]
    macro = None
    if active:
        macro = _average_metrics(active)
    micro = _metrics_from_counts(_sum_counts(per_field.values()), exact_accuracy=None)
    return ExtractionMetrics(per_field=per_field, macro=macro, micro=micro)


def _sum_counts(metrics: Sequence[FieldMetrics]) -> _Counts:
    return _Counts(
        true_positive=sum(metric.true_positive for metric in metrics),
        false_positive=sum(metric.false_positive for metric in metrics),
        false_negative=sum(metric.false_negative for metric in metrics),
    )


def _metric_is_active(metric: FieldMetrics) -> bool:
    return bool(metric.true_positive or metric.false_positive or metric.false_negative)


def _metrics_from_counts(counts: _Counts, *, exact_accuracy: float | None) -> FieldMetrics:
    tp, fp, fn = counts.true_positive, counts.false_positive, counts.false_negative
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FieldMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        support=tp + fn,
        exact_accuracy=exact_accuracy,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def _average_metrics(metrics: Sequence[FieldMetrics]) -> FieldMetrics:
    exact_values = [metric.exact_accuracy for metric in metrics if metric.exact_accuracy is not None]
    counts = _sum_counts(metrics)
    return FieldMetrics(
        precision=sum(metric.precision for metric in metrics) / len(metrics),
        recall=sum(metric.recall for metric in metrics) / len(metrics),
        f1=sum(metric.f1 for metric in metrics) / len(metrics),
        support=sum(metric.support for metric in metrics),
        exact_accuracy=sum(exact_values) / len(exact_values) if exact_values else None,
        true_positive=counts.true_positive,
        false_positive=counts.false_positive,
        false_negative=counts.false_negative,
    )
