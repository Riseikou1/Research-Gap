import unittest
import json
from io import BytesIO
from types import SimpleNamespace

from src.analysis.comparison import classify_study_type, to_paper_features
from src.analysis.models import GapCandidate, LandscapeBasis
from src.analysis.verification import GapVerifier
from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.openalex import OpenAlexRetriever, _serialize_provider_query
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import RetrievalRequest


def claim(value: str) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        evidence_text=value,
        source="abstract",
        confidence=0.9,
    )


class NoCallRetriever:
    provider_name = "test"

    def search(self, request):
        raise AssertionError("external retrieval should not execute")


class NoCallExtractor:
    failures = []

    def extract_many(self, papers, limit=None):
        raise AssertionError("evidence extraction should not execute")


class FinalMilestone6Test(unittest.TestCase):
    def test_initial_full_match_skips_direct_provider_work(self):
        record = PaperEvidence(
            paper_id="p1",
            title="Study p1",
            study_type="empirical",
            research_objective=claim("problem_a"),
            method_or_intervention=[claim("method_b")],
            constraints=[claim("constraint_c")],
            extraction_confidence=0.9,
        )
        idea = ResearchIdea(
            original_text="problem_a method_b constraint_c",
            problem=["problem_a"],
            intervention_or_method=["method_b"],
            constraints=["constraint_c"],
        )
        verifier = GapVerifier(
            MultiQueryRetriever(NoCallRetriever(), max_candidates=20),
            NoCallExtractor(),
        )

        assessment = verifier.assess_idea(idea, evidence=[record])

        self.assertEqual(assessment.label, "well_studied")
        self.assertEqual(assessment.verification_queries, [])
        self.assertIn("p1", assessment.supporting_paper_ids)

    def test_unanchored_multi_component_candidate_is_rejected_before_retrieval(self):
        idea = ResearchIdea(
            original_text="problem_a method_b",
            problem=["problem_a"],
            intervention_or_method=["method_b"],
        )
        candidate = GapCandidate(
            title="Unrelated combination",
            description="A landscape-only combination",
            category="combination",
            pattern_type="combination_gap",
            rationale="test",
            supporting_paper_ids=["p1", "p2"],
            landscape_basis=[
                LandscapeBasis(
                    dimension="problem",
                    value="problem_x",
                    count=2,
                    total=4,
                    prevalence=0.5,
                ),
                LandscapeBasis(
                    dimension="method",
                    value="method_y",
                    count=2,
                    total=4,
                    prevalence=0.5,
                ),
            ],
        )
        verifier = GapVerifier(
            MultiQueryRetriever(NoCallRetriever(), max_candidates=20),
            NoCallExtractor(),
        )

        result = verifier.verify_many(idea, [candidate], [])

        self.assertEqual(result, [])
        self.assertEqual(
            verifier.metrics_snapshot()["candidate_rejected_unanchored"],
            1,
        )

    def test_study_type_remains_metadata_not_problem(self):
        record = PaperEvidence(
            paper_id="p1",
            title="A survey of problem_a",
            study_type="empirical",
            research_objective=claim("problem_a"),
            extraction_confidence=0.9,
        )

        features = to_paper_features([record])

        self.assertEqual(classify_study_type(record), "survey")
        self.assertEqual(features[0].problems, ["problem a"])
        self.assertEqual(features[0].study_type, "survey")
        self.assertNotIn("survey", features[0].problems)

    def test_provider_query_serialization_removes_operator_syntax(self):
        serialized, fallback = _serialize_provider_query(
            '"problem_a" AND (method_b OR constraint_c)'
        )

        self.assertTrue(fallback)
        self.assertNotIn('"', serialized)
        self.assertNotIn("(", serialized)
        self.assertNotIn(")", serialized)
        self.assertNotIn("_", serialized)
        self.assertIn("problem", serialized)

    def test_provider_fallback_preserves_original_query_provenance(self):
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "display_name": "Study one",
                    "publication_year": 2025,
                }
            ]
        }
        retriever = OpenAlexRetriever(
            api_key="test-key",
            opener=lambda request, timeout: BytesIO(
                json.dumps(payload).encode("utf-8")
            ),
        )
        original = SearchQuery(
            text='"problem_a" AND method_b',
            strategy="original",
            source="deterministic",
        )

        papers = retriever.search(
            RetrievalRequest(
                query=original,
                mode=RetrievalMode.BROAD_LEXICAL,
                limit=1,
            )
        )

        provenance = papers[0].provenance[0]
        self.assertEqual(provenance.query, original)
        self.assertTrue(provenance.fallback_used)
        self.assertEqual(provenance.serialized_query, "problem a AND method b")


if __name__ == "__main__":
    unittest.main()
