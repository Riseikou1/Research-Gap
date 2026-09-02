import unittest
from types import SimpleNamespace

from src.analysis.verification import GapVerifier, _pre_screen_papers
from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.extraction.paper_extractor import PaperExtractor, _BatchExtractionResult
from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.ranking.lexical import LexicalScorer
from src.retrieval.multi_query import MultiQueryRetriever


def claim(value: str, text: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        evidence_text=text or value,
        source="abstract",
        confidence=0.9,
    )


class BatchResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["text_format"] is _BatchExtractionResult:
            ids = [
                line.removeprefix("Paper ID: ")
                for line in kwargs["input"].splitlines()
                if line.startswith("Paper ID: ")
            ]
            payload = {
                "papers": [
                    {
                        "paper_id": paper_id,
                        "evidence": {"extraction_confidence": 0.5},
                    }
                    for paper_id in ids
                ]
            }
        else:
            payload = {"extraction_confidence": 0.5}
        return SimpleNamespace(
            output_parsed=kwargs["text_format"].model_validate(payload),
            status="completed",
            output=[],
        )


class ReturningRetriever:
    provider_name = "test-provider"

    def __init__(self, paper: Paper):
        self.paper = paper

    def search(self, request):
        return [self.paper]


class CountingExtractor:
    failures = []

    def __init__(self, record: PaperEvidence):
        self.record = record
        self.calls = 0

    def extract_many(self, papers, limit=None):
        self.calls += 1
        return [self.record for _ in papers[:limit]]


class ColdPathTest(unittest.TestCase):
    def test_oversized_abstracts_split_batches_by_input_budget(self):
        responses = BatchResponses()
        extractor = PaperExtractor(
            client=SimpleNamespace(responses=responses),
            batch_size=3,
            max_batch_input_chars=900,
        )
        papers = [
            Paper(id=f"p{index}", title=f"Paper {index}", abstract="term " * 220)
            for index in range(3)
        ]

        result = extractor.extract_many(papers)

        self.assertEqual([item.paper_id for item in result], ["p0", "p1", "p2"])
        self.assertEqual(len(responses.calls), 3)
        self.assertTrue(all("Paper ID:" in call["input"] for call in responses.calls))

    def test_verification_reuses_exact_initial_evidence_before_extraction(self):
        paper = Paper(
            id="p1",
            title="Study p1",
            abstract="problem alpha uses method beta",
        )
        record = PaperEvidence(
            paper_id="p1",
            title="Study p1",
            study_type="empirical",
            research_objective=claim("problem alpha"),
            method_or_intervention=[claim("method beta")],
            extraction_confidence=0.9,
        )
        extractor = CountingExtractor(record)
        verifier = GapVerifier(
            MultiQueryRetriever(
                ReturningRetriever(paper),
                max_candidates=20,
                per_route_limit=5,
                max_workers=1,
            ),
            extractor,
        )
        idea = ResearchIdea(
            original_text="problem alpha method beta",
            problem=["problem alpha"],
            intervention_or_method=["method beta"],
        )

        verifier.prime_evidence([paper], [record])
        verifier.assess_idea(idea, evidence=[record])

        self.assertEqual(extractor.calls, 0)
        self.assertGreaterEqual(
            verifier.metrics_snapshot()["verification_papers_already_cached"],
            1,
        )

    def test_prescreen_rejects_irrelevant_but_keeps_ambiguous_and_explicit(self):
        papers = [
            Paper(id="irrelevant", title="Ocean currents", abstract="marine tides"),
            Paper(id="ambiguous", title="Problem alpha study", abstract="a related evaluation"),
            Paper(
                id="explicit",
                title="Problem alpha with method beta",
                abstract="problem alpha is evaluated using method beta",
            ),
        ]

        screened = _pre_screen_papers(
            papers,
            [("problem alpha",), ("method beta",)],
            LexicalScorer(),
            "problem alpha method beta",
        )

        screened_ids = {paper.id for paper in screened}
        self.assertNotIn("irrelevant", screened_ids)
        self.assertIn("ambiguous", screened_ids)
        self.assertIn("explicit", screened_ids)

    def test_prescreen_keeps_explicit_support_for_all_structured_facets(self):
        paper = Paper(
            id="supported",
            title="Problem alpha and method beta",
            abstract="problem alpha is evaluated with method beta",
        )

        screened = _pre_screen_papers(
            [paper, Paper(id="other", title="Unrelated", abstract="unrelated")],
            [("problem alpha",), ("method beta",)],
            LexicalScorer(),
            "problem alpha method beta",
        )

        self.assertEqual([item.id for item in screened], ["supported"])


if __name__ == "__main__":
    unittest.main()
