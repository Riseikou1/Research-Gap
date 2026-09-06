"""Offline evaluation with explicit failures and complete metric denominators."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .ablation import AblationPrediction, AblationPredictionGenerator, AblationVariant
from .dataset import load_jsonl
from .extraction import aggregate_extraction, evaluate_attribution, evaluate_extraction
from .metrics import RetrievalMetrics, evaluate_retrieval
from .models import (
    AttributionMetrics, DeduplicationEvaluationCase, EvaluationFailure, EvaluationReport,
    EvaluationType, ExtractionEvaluationCase, ExtractionMetrics, PerformanceMetrics,
    RetrievalAggregateMetrics, RetrievalEvaluationCase, VerificationEvaluationCase,
    VerificationMetrics,
)
from .performance import aggregate_performance, performance_from_result
from .reporting import format_report, report_to_json
from .retrieval import aggregate_deduplication, evaluate_deduplication
from .verification import evaluate_verification, parse_verification_prediction

EvaluationCase = (
    RetrievalEvaluationCase | DeduplicationEvaluationCase
    | ExtractionEvaluationCase | VerificationEvaluationCase
)


class EvaluationRunner:
    """Score saved predictions; execute only cases without a supplied prediction."""

    def __init__(self, *, dataset_version: str = "unspecified", metadata: dict[str, Any] | None = None) -> None:
        self.dataset_version = dataset_version
        self.metadata = metadata or {}
        self.failures: list[EvaluationFailure] = []

    def _failure(self, kind: EvaluationType, case_id: str, stage: str, error: object) -> None:
        # Keep the original execution error instead of also reporting its
        # resulting missing prediction as a second failure.
        if not any(item.evaluation_type == kind and item.case_id == case_id for item in self.failures):
            self.failures.append(EvaluationFailure(
                case_id=case_id, evaluation_type=kind, stage=stage, error=str(error),
            ))

    def evaluate_retrieval(
        self, cases: Sequence[RetrievalEvaluationCase], predictions: Mapping[str, object],
    ) -> RetrievalAggregateMetrics:
        scores: list[RetrievalMetrics] = []
        scored = 0
        for case in cases:
            try:
                value = predictions.get(case.id)
                if value is None:
                    raise ValueError("missing or null retrieval prediction")
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ValueError("retrieved_ids must be a sequence of paper IDs")
                if any(not isinstance(item, str) or not item.strip() for item in value):
                    raise ValueError("retrieved_ids must contain non-empty strings")
                score = evaluate_retrieval(value, case.relevant_papers)
                scored += 1
            except (TypeError, ValueError) as exc:
                self._failure("retrieval", case.id, "retrieval_scoring", exc)
                score = RetrievalMetrics(0.0, 0.0, 0.0, 0.0)
            scores.append(score)
        averages = {
            name: sum(getattr(item, name) for item in scores) / len(cases)
            for name in ("recall_at_10", "recall_at_50", "mrr", "ndcg_at_10")
        } if cases else {}
        return RetrievalAggregateMetrics(cases=len(cases), cases_scored=scored, **averages)

    def generate_retrieval_ablation_predictions(
        self, cases: Sequence[RetrievalEvaluationCase], generator: AblationPredictionGenerator,
        variant: AblationVariant | str,
    ) -> dict[str, AblationPrediction]:
        """Keep unavailable ablations explicit rather than interpreting them as empty results."""
        results: dict[str, AblationPrediction] = {}
        for case in cases:
            try:
                result = generator.generate(case.idea, variant)
                results[case.id] = result
                if not result.available:
                    self._failure("retrieval", case.id, "ablation_unavailable",
                                  result.unavailable_reason or "ablation dependencies unavailable")
            except Exception as exc:
                self._failure("retrieval", case.id, "ablation_generation", exc)
        return results

    def generate_retrieval_predictions(
        self, cases: Sequence[RetrievalEvaluationCase], generator: AblationPredictionGenerator,
        variant: AblationVariant | str,
    ) -> dict[str, Sequence[str]]:
        """Return available ranked IDs; absent cases remain failures when scored."""
        results = self.generate_retrieval_ablation_predictions(cases, generator, variant)
        return {case_id: result.retrieved_ids for case_id, result in results.items() if result.available}

    def evaluate_extraction(
        self, cases: Sequence[ExtractionEvaluationCase], predictions: Mapping[str, object],
    ) -> tuple[ExtractionMetrics, AttributionMetrics | None]:
        extraction_scores, attribution_scores = [], []
        for case in cases:
            try:
                value = predictions.get(case.id)
                if value is None:
                    raise ValueError("missing or null extraction prediction")
                predicted_id = value.get("paper_id") if isinstance(value, Mapping) else getattr(value, "paper_id", None)
                if predicted_id is not None and predicted_id != case.paper_id:
                    raise ValueError("extraction paper_id does not match the evaluation case")
                extraction = evaluate_extraction(value, case.gold)
                attribution = evaluate_attribution(value, title=case.title, abstract=case.abstract)
            except (TypeError, ValueError) as exc:
                self._failure("extraction", case.id, "extraction_scoring", exc)
                extraction = evaluate_extraction({}, case.gold)
                attribution = evaluate_attribution({}, title=case.title, abstract=case.abstract)
            extraction_scores.append(extraction)
            attribution_scores.append(attribution)
        attribution = None
        if attribution_scores:
            total = sum(item.total_claims for item in attribution_scores)
            supported = sum(item.supported_claims for item in attribution_scores)
            rate = supported / total if total else 0.0
            attribution = AttributionMetrics(
                total_claims=total, supported_claims=supported, unsupported_claims=total - supported,
                supported_claim_rate=rate, unsupported_claim_rate=(total - supported) / total if total else 0.0,
                attribution_accuracy=rate,
            )
        return aggregate_extraction(extraction_scores), attribution

    def evaluate_verification(
        self, cases: Sequence[VerificationEvaluationCase], predictions: Mapping[str, object],
    ) -> VerificationMetrics:
        valid = {}
        for case in cases:
            try:
                value = predictions.get(case.id)
                if value is None:
                    raise ValueError("missing or null verification prediction")
                parse_verification_prediction(value)
                valid[case.id] = value
            except (TypeError, ValueError) as exc:
                self._failure("verification", case.id, "verification_scoring", exc)
        return evaluate_verification(cases, valid)

    def evaluate_deduplication(
        self, cases: Sequence[DeduplicationEvaluationCase], predictions: Mapping[str, object],
    ) -> dict[str, float | int]:
        scores = []
        for case in cases:
            try:
                value = predictions.get(case.id)
                if value is None:
                    raise ValueError("missing or null deduplication prediction")
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ValueError("duplicate_pairs must be a sequence of paper-ID pairs")
                score = evaluate_deduplication(case.gold_duplicate_pairs, value)
            except (TypeError, ValueError) as exc:
                self._failure("deduplication", case.id, "deduplication_scoring", exc)
                score = evaluate_deduplication(case.gold_duplicate_pairs, [])
            scores.append(score)
        return aggregate_deduplication(scores)

    def evaluate_performance(self, results: Sequence[object]) -> PerformanceMetrics:
        return aggregate_performance([performance_from_result(result) for result in results])

    def _prepare_predictions(
        self, kind: EvaluationType, cases: Sequence[EvaluationCase], predictions: Mapping[str, object],
        executor: Callable[[object], object] | None,
    ) -> dict[str, object]:
        values = dict(predictions)
        for case in cases:
            if case.id not in values and executor is not None:
                try:
                    values[case.id] = executor(case)
                except Exception as exc:
                    self._failure(kind, case.id, "execution", exc)
            if values.get(case.id) is None:
                self._failure(kind, case.id, "prediction", "missing or null prediction")
        return values

    def run(
        self, *, retrieval_cases: Sequence[RetrievalEvaluationCase] = (),
        retrieval_predictions: Mapping[str, Sequence[str]] | None = None,
        deduplication_cases: Sequence[DeduplicationEvaluationCase] = (),
        deduplication_predictions: Mapping[str, object] | None = None,
        extraction_cases: Sequence[ExtractionEvaluationCase] = (),
        extraction_predictions: Mapping[str, object] | None = None,
        verification_cases: Sequence[VerificationEvaluationCase] = (),
        verification_predictions: Mapping[str, object] | None = None,
        executor: Callable[[object], object] | None = None,
        performance_results: Sequence[object] = (),
    ) -> EvaluationReport:
        self.failures = []
        groups = (
            ("retrieval", retrieval_cases, retrieval_predictions or {}),
            ("deduplication", deduplication_cases, deduplication_predictions or {}),
            ("extraction", extraction_cases, extraction_predictions or {}),
            ("verification", verification_cases, verification_predictions or {}),
        )
        # Reject dataset mixups before an executor can make any provider calls.
        for kind, cases, predictions in groups:
            ids = {case.id for case in cases}
            if len(ids) != len(cases):
                raise ValueError(f"duplicate {kind} case IDs")
            unknown = set(predictions) - ids
            if unknown:
                raise ValueError(f"unknown {kind} prediction case IDs: {sorted(unknown, key=str)}")
        prepared = {kind: self._prepare_predictions(kind, cases, predictions, executor)
                    for kind, cases, predictions in groups}
        retrieval = self.evaluate_retrieval(retrieval_cases, prepared["retrieval"]) if retrieval_cases else None
        deduplication = self.evaluate_deduplication(deduplication_cases, prepared["deduplication"]) if deduplication_cases else None
        extraction = attribution = None
        if extraction_cases:
            extraction, attribution = self.evaluate_extraction(extraction_cases, prepared["extraction"])
        verification = self.evaluate_verification(verification_cases, prepared["verification"]) if verification_cases else None
        performance = self.evaluate_performance(performance_results) if performance_results else None
        total = sum(len(cases) for _, cases, _ in groups)
        failed = len({(item.evaluation_type, item.case_id) for item in self.failures})
        if executor is not None and performance is None:
            performance = PerformanceMetrics(cases_total=total, cases_completed=total - failed, cases_failed=failed)
        return EvaluationReport(
            dataset_version=self.dataset_version, timestamp=datetime.now(timezone.utc).isoformat(),
            cases_total=total, cases_completed=total - failed, cases_failed=failed, metadata=self.metadata,
            retrieval=retrieval, deduplication=deduplication, extraction=extraction, attribution=attribution,
            verification=verification, performance=performance, failures=list(self.failures),
        )


def _prediction_records(path: Path, field: str) -> dict[str, Any]:
    records: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            prefix = f"{path}:{line_number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{prefix}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{prefix}: prediction row must be an object")
            if "_meta" in raw:
                if set(raw) != {"_meta"} or not isinstance(raw["_meta"], dict):
                    raise ValueError(f"{prefix}: metadata must be a separate object row")
                continue
            case_id = raw.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"{prefix}: prediction rows require a non-empty case_id")
            case_id = case_id.strip()
            if case_id in records:
                raise ValueError(f"{prefix}: duplicate prediction case ID {case_id!r}")
            if field not in raw and "prediction" not in raw:
                raise ValueError(f"{prefix}: prediction row requires {field!r} or 'prediction'")
            records[case_id] = raw[field] if field in raw else raw["prediction"]
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Research GAP evaluation")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--evaluation-type", choices=("retrieval", "deduplication", "extraction", "verification"), required=True)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-version", default="unspecified")
    args = parser.parse_args(argv)
    model_type, field = {
        "retrieval": (RetrievalEvaluationCase, "retrieved_ids"),
        "deduplication": (DeduplicationEvaluationCase, "duplicate_pairs"),
        "extraction": (ExtractionEvaluationCase, "evidence"),
        "verification": (VerificationEvaluationCase, "assessment"),
    }[args.evaluation_type]
    try:
        dataset = load_jsonl(args.dataset, model_type, dataset_version=args.dataset_version)
        if not dataset.cases:
            raise ValueError("evaluation dataset must contain at least one case")
        raw = _prediction_records(args.predictions, field)
        runner = EvaluationRunner(dataset_version=dataset.version)
        report = runner.run(**{
            f"{args.evaluation_type}_cases": dataset.cases,
            f"{args.evaluation_type}_predictions": raw,
        })
        if args.output:
            if args.output.resolve() in {args.dataset.resolve(), args.predictions.resolve()}:
                raise ValueError("--output must not overwrite the dataset or predictions")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report_to_json(report) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    return 1 if report.cases_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
