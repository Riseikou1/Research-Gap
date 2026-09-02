"""Stable JSON and human-readable rendering for evaluation reports."""

from __future__ import annotations

import json
from typing import Any

from .models import EvaluationReport


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return report.model_dump(mode="json", exclude_none=False)


def report_to_json(report: EvaluationReport, *, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=indent, sort_keys=True)


def format_report(report: EvaluationReport) -> str:
    lines = ["Research GAP Evaluation", "=======================", f"Dataset: {report.dataset_version}", f"Cases: {report.cases_completed}/{report.cases_total} completed; {report.cases_failed} failed"]
    retrieval = report.retrieval
    if retrieval:
        lines += ["", "Retrieval", "---------"]
        for key in ("cases", "recall_at_10", "recall_at_50", "mrr", "ndcg_at_10"):
            value = getattr(retrieval, key, None)
            if value is not None:
                lines.append(f"{key}: {value}")
    extraction = report.extraction
    if extraction:
        lines += ["", "Extraction", "----------", "Field                     Precision  Recall  F1"]
        for field, metric in extraction.per_field.items():
            lines.append(f"{field:25} {metric.precision:9.3f} {metric.recall:7.3f} {metric.f1:5.3f}")
    attribution = report.attribution
    if attribution:
        lines += ["", "Attribution", "-----------", f"Supported claims: {attribution.supported_claim_rate:.1%}", f"Unsupported claim rate: {attribution.unsupported_claim_rate:.1%}"]
    verification = report.verification
    if verification:
        lines += ["", "Verification", "------------", f"Cases scored: {verification.cases_scored}/{verification.cases_total}", f"Assessment accuracy: {verification.accuracy}", f"Counterexample discovery rate: {verification.counterexample_discovery_rate}", f"False promising-gap rate: {verification.false_promising_gap_rate}"]
    performance = report.performance
    if performance:
        lines += ["", "Performance", "-----------", f"Total latency: {performance.total_seconds}", f"Cache mode: {performance.cache_mode}", f"Cache hit rates: {performance.cache_hit_rates}"]
        if performance.latency:
            lines.append(f"Latency summary: {performance.latency}")
    if report.failures:
        lines += ["", f"Failures: {len(report.failures)}"]
    return "\n".join(lines)
