"""Offline structured-evidence and provenance scoring."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

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
    values = value if isinstance(value, list) else [value]
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


def _field_metrics(predicted: list[str], gold: list[str]) -> FieldMetrics:
    predicted_set, gold_set = set(predicted), set(gold)
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FieldMetrics(
        precision=precision, recall=recall, f1=f1, support=len(gold_set),
        exact_accuracy=1.0 if predicted_set == gold_set else 0.0,
    )


def evaluate_extraction(predicted: PaperEvidence | Mapping[str, object], gold: Mapping[str, Sequence[str]]) -> ExtractionMetrics:
    per_field = {
        field: _field_metrics(_values(predicted, field), [normalize_value(str(v)) for v in gold.get(field, [])])
        for field in EVIDENCE_FIELDS
    }
    tp = fp = fn = 0
    for field, metrics in per_field.items():
        pred, expected = set(_values(predicted, field)), {normalize_value(str(v)) for v in gold.get(field, [])}
        tp += len(pred & expected); fp += len(pred - expected); fn += len(expected - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    micro = FieldMetrics(precision=precision, recall=recall, f1=f1, support=tp + fn)
    nonempty = [metric for metric in per_field.values() if metric.support]
    macro = None
    if nonempty:
        macro = FieldMetrics(
            precision=sum(m.precision for m in nonempty) / len(nonempty),
            recall=sum(m.recall for m in nonempty) / len(nonempty),
            f1=sum(m.f1 for m in nonempty) / len(nonempty),
            support=sum(m.support for m in nonempty),
            exact_accuracy=sum(m.exact_accuracy or 0 for m in nonempty) / len(nonempty),
        )
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
        per_field[field] = FieldMetrics(
            precision=sum(x.precision for x in values) / len(values),
            recall=sum(x.recall for x in values) / len(values),
            f1=sum(x.f1 for x in values) / len(values),
            support=sum(x.support for x in values),
            exact_accuracy=sum(x.exact_accuracy or 0 for x in values) / len(values),
        )
    active = [metric for metric in per_field.values() if metric.support]
    macro = None
    if active:
        macro = FieldMetrics(
            precision=sum(x.precision for x in active) / len(active),
            recall=sum(x.recall for x in active) / len(active),
            f1=sum(x.f1 for x in active) / len(active),
            support=sum(x.support for x in active),
            exact_accuracy=sum(x.exact_accuracy or 0 for x in active) / len(active),
        )
    total_support = sum(x.support for x in per_field.values())
    micro = FieldMetrics(
        precision=sum(x.precision * max(x.support, 1) for x in per_field.values()) / max(sum(max(x.support, 1) for x in per_field.values()), 1),
        recall=sum(x.recall * x.support for x in per_field.values()) / max(total_support, 1),
        f1=sum(x.f1 * max(x.support, 1) for x in per_field.values()) / max(sum(max(x.support, 1) for x in per_field.values()), 1),
        support=total_support,
    )
    return ExtractionMetrics(per_field=per_field, macro=macro, micro=micro)
