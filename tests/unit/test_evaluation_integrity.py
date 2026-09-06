"""Regression tests for failed cases and independent evaluation examples."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.extraction import evaluate_attribution
from src.evaluation.models import (
    DeduplicationEvaluationCase,
    ExtractionEvaluationCase,
    RetrievalEvaluationCase,
    VerificationEvaluationCase,
)
from src.evaluation.runner import EvaluationRunner, main
from src.evaluation.verification import evaluate_verification


def retrieval_case(case_id):
    return RetrievalEvaluationCase(
        id=case_id, idea="example idea", relevant_papers=[{"paper_id": "p", "relevance": 1}],
    )


class EvaluationIntegrityTest(unittest.TestCase):
    def test_missing_and_malformed_retrieval_results_remain_in_denominator(self):
        report = EvaluationRunner().run(
            retrieval_cases=[retrieval_case(name) for name in ("ok", "missing", "malformed")],
            retrieval_predictions={"ok": ["p"], "malformed": "p"},
        )
        self.assertEqual(report.retrieval.cases, 3)
        self.assertEqual(report.retrieval.cases_scored, 1)
        self.assertAlmostEqual(report.retrieval.mrr, 1 / 3)
        self.assertEqual((report.cases_completed, report.cases_failed), (1, 2))
        self.assertEqual({item.case_id for item in report.failures}, {"missing", "malformed"})

    def test_failed_extraction_keeps_gold_claims_in_recall(self):
        cases = [ExtractionEvaluationCase(
            id=name, paper_id=name, title="Study", gold={"datasets": ["dataset a"]},
        ) for name in ("ok", "missing", "malformed")]
        report = EvaluationRunner().run(
            extraction_cases=cases,
            extraction_predictions={"ok": {"datasets": ["dataset a"]}, "malformed": 42},
        )
        self.assertEqual(report.extraction.micro.support, 3)
        self.assertEqual(report.extraction.micro.false_negative, 2)
        self.assertAlmostEqual(report.extraction.micro.recall, 1 / 3)
        self.assertEqual(report.cases_failed, 2)

    def test_blank_evidence_span_is_not_support(self):
        result = evaluate_attribution(
            {"datasets": [{"value": "invented", "source": "abstract", "evidence_text": "!!!"}]},
            title="Study", abstract="Some source text.",
        )
        self.assertEqual(result.unsupported_claims, 1)

    def test_missing_verification_is_a_miss_not_a_correct_uncertain_label(self):
        cases = [VerificationEvaluationCase(id=name, idea="x", expected_label="uncertain")
                 for name in ("ok", "missing")]
        result = evaluate_verification(cases, {"ok": "uncertain"})
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.cases_scored, 1)
        self.assertEqual(result.per_label["uncertain"].support, 2)
        self.assertEqual(result.per_label["uncertain"].recall, 0.5)

    def test_duplicate_pairs_cannot_match_across_cases(self):
        report = EvaluationRunner().run(
            deduplication_cases=[
                DeduplicationEvaluationCase(id="a", gold_duplicate_pairs=[["p1", "p2"]]),
                DeduplicationEvaluationCase(id="b", gold_duplicate_pairs=[]),
            ],
            deduplication_predictions={"a": [], "b": [["p1", "p2"]]},
        )
        metrics = report.deduplication
        self.assertEqual((metrics.true_positive, metrics.false_positive, metrics.false_negative), (0, 1, 1))

    def test_malformed_pair_records_failure_instead_of_crashing(self):
        for prediction in (["ab"], [[1, 2]], [["p", "p"]]):
            with self.subTest(prediction=prediction):
                report = EvaluationRunner().run(
                    deduplication_cases=[DeduplicationEvaluationCase(id="a", gold_duplicate_pairs=[["p", "q"]])],
                    deduplication_predictions={"a": prediction},
                )
                self.assertEqual(report.cases_failed, 1)
                self.assertEqual(report.deduplication.false_negative, 1)

    def test_case_ids_are_scoped_to_evaluation_type(self):
        report = EvaluationRunner().run(
            retrieval_cases=[retrieval_case("same")], retrieval_predictions={"same": ["p"]},
            verification_cases=[VerificationEvaluationCase(id="same", idea="x", expected_label="uncertain")],
        )
        self.assertEqual((report.cases_total, report.cases_completed, report.cases_failed), (2, 1, 1))
        self.assertEqual(report.failures[0].evaluation_type, "verification")

    def test_saved_predictions_are_not_executed_again(self):
        executor = Mock(side_effect=AssertionError("unnecessary provider call"))
        report = EvaluationRunner().run(
            retrieval_cases=[retrieval_case("a")], retrieval_predictions={"a": ["p"]}, executor=executor,
        )
        executor.assert_not_called()
        self.assertEqual(report.cases_failed, 0)

    def test_none_and_invalid_verification_predictions_are_failures(self):
        for prediction in (None, "maybe", {"label": "uncertain", "candidate_labels": "bad"}):
            with self.subTest(prediction=prediction):
                report = EvaluationRunner().run(
                    verification_cases=[VerificationEvaluationCase(id="a", idea="x", expected_label="uncertain")],
                    verification_predictions={"a": prediction},
                )
                self.assertEqual(report.cases_failed, 1)
                self.assertEqual(report.verification.accuracy, 0.0)

    def test_unknown_prediction_ids_and_duplicate_case_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            EvaluationRunner().run(retrieval_cases=[retrieval_case("a")], retrieval_predictions={"typo": ["p"]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            EvaluationRunner().run(retrieval_cases=[retrieval_case("a"), retrieval_case("a")])

    def test_failed_cli_evaluation_writes_report_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, predictions, output = root / "gold.jsonl", root / "predictions.jsonl", root / "report.json"
            dataset.write_text(retrieval_case("a").model_dump_json() + "\n", encoding="utf-8")
            predictions.write_text("", encoding="utf-8")
            with patch("sys.stdout", io.StringIO()):
                status = main(["--dataset", str(dataset), "--evaluation-type", "retrieval",
                               "--predictions", str(predictions), "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.read_text())["cases_failed"], 1)

    def test_malformed_prediction_file_reports_line_and_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, predictions = root / "gold.jsonl", root / "predictions.jsonl"
            dataset.write_text(retrieval_case("a").model_dump_json() + "\n", encoding="utf-8")
            for row in ("[]", "{broken", '{"case_id":"a"}'):
                with self.subTest(row=row):
                    predictions.write_text(row + "\n", encoding="utf-8")
                    stderr = io.StringIO()
                    with patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                        status = main(["--dataset", str(dataset), "--evaluation-type", "retrieval",
                                       "--predictions", str(predictions)])
                    self.assertEqual(status, 2)
                    self.assertIn(f"{predictions}:1:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
