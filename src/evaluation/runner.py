"""Small offline evaluation orchestrator with optional prediction execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dataset import load_jsonl
from .extraction import aggregate_extraction, evaluate_attribution, evaluate_extraction
from .metrics import evaluate_retrieval
from .models import (
    DeduplicationEvaluationCase, EvaluationFailure, EvaluationReport,
    ExtractionEvaluationCase, RetrievalAggregateMetrics, RetrievalEvaluationCase,
    VerificationEvaluationCase,
)
from .performance import aggregate_performance, performance_from_result
from .reporting import format_report, report_to_json
from .retrieval import evaluate_deduplication
from .verification import evaluate_verification


class EvaluationRunner:
    """Score saved predictions, or execute an injected offline/stub executor."""

    def __init__(self, *, dataset_version: str = "unspecified", metadata: dict[str, Any] | None = None) -> None:
        self.dataset_version = dataset_version
        self.metadata = metadata or {}
        self.failures: list[EvaluationFailure] = []

    def evaluate_retrieval(self, cases: Sequence[RetrievalEvaluationCase], predictions: Mapping[str, Sequence[str]]) -> dict[str, Any]:
        values = []
        for case in cases:
            if case.id not in predictions:
                continue
            try:
                values.append(evaluate_retrieval(predictions[case.id], case.relevant_papers))
            except Exception as exc:
                self.failures.append(EvaluationFailure(case_id=case.id, stage="retrieval_scoring", error=str(exc)))
        if not values:
            return RetrievalAggregateMetrics(cases=0)
        return RetrievalAggregateMetrics(cases=len(values), **{name: sum(getattr(item, name) for item in values) / len(values) for name in ("recall_at_10", "recall_at_50", "mrr", "ndcg_at_10")})

    def evaluate_extraction(self, cases: Sequence[ExtractionEvaluationCase], predictions: Mapping[str, object]) -> tuple[object, object]:
        extraction_scores = []
        attribution_scores = []
        for case in cases:
            prediction = predictions.get(case.id)
            if prediction is None:
                continue
            try:
                extraction_scores.append(evaluate_extraction(prediction, case.gold))
                attribution_scores.append(evaluate_attribution(prediction, title=case.title, abstract=case.abstract))
            except Exception as exc:
                self.failures.append(EvaluationFailure(case_id=case.id, stage="extraction_scoring", error=str(exc)))
        attribution = None
        if attribution_scores:
            total = sum(x.total_claims for x in attribution_scores)
            supported = sum(x.supported_claims for x in attribution_scores)
            from .models import AttributionMetrics
            attribution = AttributionMetrics(total_claims=total, supported_claims=supported, unsupported_claims=total-supported, supported_claim_rate=supported/total if total else 0.0, unsupported_claim_rate=(total-supported)/total if total else 0.0, attribution_accuracy=supported/total if total else 0.0)
        return aggregate_extraction(extraction_scores), attribution

    def evaluate_verification(self, cases: Sequence[VerificationEvaluationCase], predictions: Mapping[str, object]):
        return evaluate_verification(cases, predictions)

    def evaluate_deduplication(self, cases: Sequence[DeduplicationEvaluationCase], predictions: Mapping[str, object]):
        gold = [pair for case in cases for pair in case.gold_duplicate_pairs]
        predicted = [pair for case in cases for pair in (predictions.get(case.id) or [])]
        return evaluate_deduplication(gold, predicted)

    def evaluate_performance(self, results: Sequence[object]):
        return aggregate_performance([performance_from_result(result) for result in results])

    def run(self, *, retrieval_cases: Sequence[RetrievalEvaluationCase] = (), retrieval_predictions: Mapping[str, Sequence[str]] | None = None, deduplication_cases: Sequence[DeduplicationEvaluationCase] = (), deduplication_predictions: Mapping[str, object] | None = None, extraction_cases: Sequence[ExtractionEvaluationCase] = (), extraction_predictions: Mapping[str, object] | None = None, verification_cases: Sequence[VerificationEvaluationCase] = (), verification_predictions: Mapping[str, object] | None = None, executor: Callable[[object], object] | None = None, performance_results: Sequence[object] = ()) -> EvaluationReport:
        self.failures = []
        failures: list[EvaluationFailure] = []
        retrieval_values = dict(retrieval_predictions or {})
        deduplication_values = dict(deduplication_predictions or {})
        extraction_values = dict(extraction_predictions or {})
        verification_values = dict(verification_predictions or {})
        if executor:
            all_cases: list[object] = [*retrieval_cases, *deduplication_cases, *extraction_cases, *verification_cases]
            for case in all_cases:
                try:
                    prediction = executor(case)
                    if isinstance(case, RetrievalEvaluationCase) and case.id not in retrieval_values:
                        retrieval_values[case.id] = prediction
                    elif isinstance(case, DeduplicationEvaluationCase) and case.id not in deduplication_values:
                        deduplication_values[case.id] = prediction
                    elif isinstance(case, ExtractionEvaluationCase) and case.id not in extraction_values:
                        extraction_values[case.id] = prediction
                    elif isinstance(case, VerificationEvaluationCase) and case.id not in verification_values:
                        verification_values[case.id] = prediction
                except Exception as exc:
                    failures.append(EvaluationFailure(case_id=case.id, stage="execution", error=str(exc)))
        extraction = attribution = verification = None
        retrieval = self.evaluate_retrieval(retrieval_cases, retrieval_values) if retrieval_cases else None
        deduplication = self.evaluate_deduplication(deduplication_cases, deduplication_values) if deduplication_cases else None
        if extraction_cases:
            extraction, attribution = self.evaluate_extraction(extraction_cases, extraction_values)
        if verification_cases:
            verification = self.evaluate_verification(verification_cases, verification_values)
        performance = self.evaluate_performance(performance_results) if performance_results else None
        total = len(retrieval_cases) + len(deduplication_cases) + len(extraction_cases) + len(verification_cases)
        provided = set(retrieval_values) | set(deduplication_values) | set(extraction_values) | set(verification_values)
        case_ids = {case.id for case in (*retrieval_cases, *deduplication_cases, *extraction_cases, *verification_cases)}
        failed_ids = {failure.case_id for failure in (*self.failures, *failures)}
        cases_failed = len((case_ids - provided) | failed_ids)
        if executor and performance is None:
            from .models import PerformanceMetrics
            performance = PerformanceMetrics(cases_total=total, cases_completed=total-cases_failed, cases_failed=cases_failed)
        return EvaluationReport(dataset_version=self.dataset_version, timestamp=datetime.now(timezone.utc).isoformat(), cases_total=total, cases_completed=total-cases_failed, cases_failed=cases_failed, metadata=self.metadata, retrieval=retrieval, deduplication=deduplication, extraction=extraction, attribution=attribution, verification=verification, performance=performance, failures=[*self.failures, *failures])


def _prediction_records(path: Path, field: str) -> dict[str, Any]:
    records: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        import json
        for line in handle:
            if not line.strip(): continue
            raw = json.loads(line)
            if "_meta" in raw:
                continue
            case_id = raw.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("prediction rows require a non-empty case_id")
            if case_id in records:
                raise ValueError(f"duplicate prediction case ID {case_id!r}")
            records[case_id] = raw.get(field, raw.get("prediction"))
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Research GAP evaluation")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--evaluation-type", choices=("retrieval", "deduplication", "extraction", "verification"), required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-version", default="unspecified")
    args = parser.parse_args(argv)
    from .models import ExtractionEvaluationCase, RetrievalEvaluationCase, VerificationEvaluationCase
    model_type, field = {"retrieval": (RetrievalEvaluationCase, "retrieved_ids"), "deduplication": (DeduplicationEvaluationCase, "duplicate_pairs"), "extraction": (ExtractionEvaluationCase, "evidence"), "verification": (VerificationEvaluationCase, "assessment")} [args.evaluation_type]
    dataset = load_jsonl(args.dataset, model_type, dataset_version=args.dataset_version)
    runner = EvaluationRunner(dataset_version=dataset.version)
    raw = _prediction_records(args.predictions, field)
    kwargs: dict[str, Any] = {f"{args.evaluation_type}_cases": dataset.cases, f"{args.evaluation_type}_predictions": raw}
    report = runner.run(**kwargs)
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
