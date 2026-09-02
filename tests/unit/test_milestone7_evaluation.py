import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    AnnotationRecord,
    EvaluationRunner,
    ExtractionEvaluationCase,
    RetrievalEvaluationCase,
    RetrievalJudgment,
    VerificationEvaluationCase,
    aggregate_ratings,
    evaluate_attribution,
    evaluate_deduplication,
    evaluate_extraction,
    evaluate_retrieval,
    evaluate_verification,
    load_jsonl,
)
from src.evaluation.performance import cache_hit_rate, performance_from_result
from src.evaluation.reporting import report_to_dict, report_to_json
from src.extraction.evidence import EvidenceItem, PaperEvidence


class RetrievalTest(unittest.TestCase):
    def test_metrics_deduplicate_ids_and_support_grades(self):
        metrics = evaluate_retrieval(
            ["noise", "A", "a", "B"],
            [RetrievalJudgment(paper_id="a", relevance=3), RetrievalJudgment(paper_id="b", relevance=1)],
        )
        self.assertEqual(metrics.recall_at_10, 1.0)
        self.assertEqual(metrics.mrr, 0.5)
        self.assertLess(metrics.ndcg_at_10, 1.0)

    def test_empty_and_no_hit_cases(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval([], [])
        metrics = evaluate_retrieval(["noise"], {"gold": 1})
        self.assertEqual(metrics.mrr, 0.0)


class DeduplicationTest(unittest.TestCase):
    def test_pairwise_order_and_duplicate_invariance(self):
        result = evaluate_deduplication([("A", "B"), ("C", "D")], [("b", "a"), ("A", "C"), ("A", "C")])
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)


def evidence_item(value: str, text: str = "The method works.", source: str = "abstract") -> EvidenceItem:
    return EvidenceItem(value=value, canonical_value="unrelated canonical label", evidence_text=text, source=source, confidence=1.0)


class ExtractionTest(unittest.TestCase):
    def test_field_scores_and_attribution_ignore_canonical_value(self):
        record = PaperEvidence(
            paper_id="p1", title="A study", study_type="empirical",
            research_objective=evidence_item("Task A", "A study" , "title"),
            method_or_intervention=[evidence_item("Method A", "The method works.")],
            extraction_confidence=1.0,
        )
        score = evaluate_extraction(record, {"research_objective": ["Task A"], "method_or_intervention": ["Method-A"]})
        self.assertEqual(score.per_field["method_or_intervention"].f1, 1.0)
        attribution = evaluate_attribution(record, title="A study", abstract="The method works.")
        self.assertEqual(attribution.supported_claims, 2)
        self.assertEqual(attribution.unsupported_claim_rate, 0.0)

    def test_unsupported_span_and_missing_source_are_counted(self):
        item = evidence_item("claim", "not present", "abstract")
        attribution = evaluate_attribution([item], title="Title", abstract="Abstract")
        self.assertEqual(attribution.unsupported_claim_rate, 1.0)


class VerificationTest(unittest.TestCase):
    def test_uncertain_and_dangerous_positive_are_scored(self):
        cases = [
            VerificationEvaluationCase(id="a", idea="x", expected_label="well_studied", known_counterexample_ids=["p1"]),
            VerificationEvaluationCase(id="b", idea="y", expected_label="uncertain"),
        ]
        result = evaluate_verification(cases, {"a": {"label": "promising_gap", "counterexample_paper_ids": ["p1"]}, "b": "uncertain"})
        self.assertEqual(result.counterexample_discovery_rate, 1.0)
        self.assertEqual(result.false_promising_gap_count, 1)
        self.assertEqual(result.accuracy, 0.5)


class DatasetAndRunnerTest(unittest.TestCase):
    def test_jsonl_version_validation_and_end_to_end_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.jsonl"
            path.write_text(json.dumps({"_meta": {"dataset_version": "m7-v1"}}) + "\n" + json.dumps({"id": "c1", "idea": "x", "relevant_papers": [{"paper_id": "p1", "relevance": 1}]}) + "\n", encoding="utf-8")
            dataset = load_jsonl(path, RetrievalEvaluationCase)
            self.assertEqual(dataset.version, "m7-v1")
            report = EvaluationRunner(dataset_version=dataset.version).run(retrieval_cases=dataset.cases, retrieval_predictions={"c1": ["p1"]})
            self.assertEqual(report.retrieval["mrr"], 1.0)
            self.assertEqual(report.dataset_version, "m7-v1")

    def test_stub_executor_produces_report_and_records_failed_case(self):
        cases = [
            RetrievalEvaluationCase(id="ok", idea="x", relevant_papers=[RetrievalJudgment(paper_id="p", relevance=1)]),
            RetrievalEvaluationCase(id="bad", idea="x", relevant_papers=[RetrievalJudgment(paper_id="q", relevance=1)]),
        ]

        def executor(case):
            if case.id == "bad":
                raise RuntimeError("stub failure")
            return ["p"]

        report = EvaluationRunner(dataset_version="m7-v1").run(retrieval_cases=cases, executor=executor)
        self.assertEqual(report.retrieval["cases"], 1)
        self.assertEqual(report.performance.cases_failed, 1)
        self.assertEqual(report.failures[0].stage, "execution")

    def test_duplicate_case_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            row = {"id": "same", "idea": "x", "relevant_papers": [{"paper_id": "p1", "relevance": 1}]}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path, RetrievalEvaluationCase)

    def test_report_json_round_trip_keeps_numeric_metrics(self):
        case = RetrievalEvaluationCase(id="c", idea="x", relevant_papers=[RetrievalJudgment(paper_id="p", relevance=1)])
        report = EvaluationRunner(dataset_version="m7-v1").run(retrieval_cases=[case], retrieval_predictions={"c": ["p"]})
        payload = json.loads(report_to_json(report))
        self.assertIsInstance(payload["retrieval"]["mrr"], float)
        self.assertEqual(payload, report_to_dict(report))


class HumanSupportTest(unittest.TestCase):
    def test_rating_aggregation_is_not_automatic_scoring(self):
        records = [AnnotationRecord(case_id="c", candidate_id="g", rating=3, reviewer_id="r1"), AnnotationRecord(case_id="c", candidate_id="g", rating=5, reviewer_id="r2")]
        summary = aggregate_ratings(records)["g"]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean"], 4.0)


class PerformanceTest(unittest.TestCase):
    def test_existing_work_accounting_is_reused(self):
        result = type("Result", (), {
            "stage_timings": {"planning": 1.0, "initial_retrieval": 2.0},
            "work_metrics": {"retrieval_cache_hits": 1, "retrieval_cache_misses": 1, "planning_cache_hits": 1, "openai_decomposition_requests": 1, "retrieved_papers": 3},
        })()
        metrics = performance_from_result(result, total_seconds=3.0)
        self.assertEqual(metrics.stage_seconds["planning_seconds"], 1.0)
        self.assertEqual(metrics.cache_hit_rates["retrieval"], 0.5)
        self.assertEqual(metrics.token_usage, "unavailable")
        self.assertEqual(cache_hit_rate(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
