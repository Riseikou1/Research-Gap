import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from src.analysis.clustering import LandscapeAnalyzer
from src.analysis.gap_candidates import GapCandidateGenerator
from src.analysis.verification import _idea_match_strength
from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.extraction.paper_extractor import (
    PaperExtractor,
    _BatchExtractionResult,
    _ExtractionResult,
)
from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import RetrievalRequest
from src.retrieval.multi_query import MultiQueryRetriever
from src.query.openai_decomposer import OpenAIDecomposer
from src.query.openai_generator import OpenAIQueryGenerator


def claim(value: str, evidence_text: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        evidence_text=evidence_text or value,
        source="abstract",
        confidence=0.9,
    )


class _Responses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        payload = kwargs["text_format"].model_validate(self.payload)
        return SimpleNamespace(output_parsed=payload, status="completed", output=[])


class _RetrievalClient:
    provider_name = "fake"

    def __init__(self):
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return [Paper(id="p1", title="Paper one")]


class _BatchResponses:
    def __init__(self, omitted: set[str] | None = None, *, omit_last_once: bool = False):
        self.calls = []
        self.omitted = omitted or set()
        self.omit_last_once = omit_last_once

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["text_format"] is _BatchExtractionResult:
            ids = [
                line.removeprefix("Paper ID: ")
                for line in kwargs["input"].splitlines()
                if line.startswith("Paper ID: ")
            ]
            omitted = set(self.omitted)
            if self.omit_last_once and ids:
                omitted.add(ids[-1])
                self.omit_last_once = False
            payload = {
                "papers": [
                    {
                        "paper_id": paper_id,
                        "evidence": {"extraction_confidence": 0.5},
                    }
                    for paper_id in ids
                    if paper_id not in omitted
                ]
            }
        else:
            payload = {"extraction_confidence": 0.5}
        return SimpleNamespace(
            output_parsed=kwargs["text_format"].model_validate(payload),
            status="completed",
            output=[],
        )


class OptimizationTest(unittest.TestCase):
    def test_planning_cache_reuses_decomposition_and_queries_across_instances(self):
        decomposition = {
            "original_text": "method A for problem B",
            "problem": ["problem B"],
            "population": [],
            "intervention_or_method": ["method A"],
            "data_or_modality": [],
            "comparison": [],
            "outcomes": [],
            "domain": [],
            "constraints": [],
            "keywords": ["method A", "problem B"],
            "synonyms": [],
            "canonical_facets": [
                {
                    "facet": "problem",
                    "values": [
                        {"value": "problem B", "canonical_value": "problem B"},
                    ],
                },
                {
                    "facet": "intervention_or_method",
                    "values": [
                        {"value": "method A", "canonical_value": "method A"},
                    ],
                },
            ],
        }
        query_payload = {
            "queries": [
                {
                    "text": "method A problem B evidence",
                    "strategy": "conceptual_reformulation",
                }
            ]
        }

        with TemporaryDirectory() as directory:
            first_decomposition = _Responses(decomposition)
            OpenAIDecomposer(
                client=SimpleNamespace(responses=first_decomposition),
                model="model-a",
                cache_path=f"{directory}/cache.sqlite3",
            ).decompose("  method A   for problem B  ")

            second_decomposition = _Responses(decomposition)
            cached_idea = OpenAIDecomposer(
                client=SimpleNamespace(responses=second_decomposition),
                model="model-a",
                cache_path=f"{directory}/cache.sqlite3",
            ).decompose("method A for problem B")
            self.assertEqual(second_decomposition.calls, 0)
            self.assertEqual(cached_idea.original_text, "method A for problem B")

            first_queries = _Responses(query_payload)
            OpenAIQueryGenerator(
                client=SimpleNamespace(responses=first_queries),
                model="model-a",
                cache_path=f"{directory}/cache.sqlite3",
            ).generate(cached_idea)
            second_queries = _Responses(query_payload)
            OpenAIQueryGenerator(
                client=SimpleNamespace(responses=second_queries),
                model="model-a",
                cache_path=f"{directory}/cache.sqlite3",
            ).generate(cached_idea)
            self.assertEqual(second_queries.calls, 0)

    def test_retrieval_cache_reuses_fresh_rows_and_refreshes_expired_rows(self):
        now = [0.0]
        client = _RetrievalClient()
        with TemporaryDirectory() as directory:
            retriever = MultiQueryRetriever(
                client,
                max_workers=1,
                cache_path=f"{directory}/cache.sqlite3",
                retrieval_cache_ttl_seconds=10,
                clock=lambda: now[0],
            )
            request = RetrievalRequest(
                query=SearchQuery(text=" Q ", strategy="test", source="deterministic"),
                mode=RetrievalMode.BROAD_LEXICAL,
                limit=2,
            )
            retriever._search_cached(request)
            retriever._search_cached(request)
            self.assertEqual(client.calls, 1)
            self.assertEqual(retriever.metrics_snapshot()["retrieval_cache_hits"], 1)

            now[0] = 11.0
            retriever._search_cached(request)
            self.assertEqual(client.calls, 2)

    def test_cross_field_explicit_text_supports_partial_facets_only(self):
        idea = ResearchIdea(
            original_text="method_a predicts problem_b under constraint_c in setting_d",
            problem=["problem_b"],
            intervention_or_method=["method_a"],
            constraints=["constraint_c"],
            population=["setting_d"],
        )
        record = PaperEvidence(
            paper_id="p1",
            title="Study p1",
            study_type="empirical",
            method_or_intervention=[
                claim(
                    "method_a",
                    "method_a predicts problem_b under constraint_c",
                )
            ],
            extraction_confidence=0.9,
        )
        complete, matched = _idea_match_strength(idea, record)
        self.assertFalse(complete)
        self.assertEqual(set(matched), {"problem", "method", "constraint"})

    def test_missing_comparison_requires_positive_structural_justification(self):
        records = [
            PaperEvidence(
                paper_id="p1",
                title="A",
                study_type="empirical",
                method_or_intervention=[claim("method_a")],
                comparison_or_baseline=[claim("baseline_c")],
                extraction_confidence=0.9,
            ),
            PaperEvidence(
                paper_id="p2",
                title="B",
                study_type="empirical",
                method_or_intervention=[claim("method_b")],
                comparison_or_baseline=[claim("baseline_c")],
                extraction_confidence=0.9,
            ),
        ]
        idea = ResearchIdea(
            original_text="method_a and method_b",
            intervention_or_method=["method_a", "method_b"],
        )
        candidates = GapCandidateGenerator(max_candidates=50).generate(
            idea,
            LandscapeAnalyzer().analyze(records),
            records,
        )
        self.assertFalse(any(item.pattern_type == "missing_comparison" for item in candidates))

    def test_batch_extraction_keeps_valid_siblings_and_retries_missing_member(self):
        responses = _BatchResponses(omit_last_once=True)
        extractor = PaperExtractor(
            client=SimpleNamespace(responses=responses),
            batch_size=3,
            max_workers=2,
        )
        papers = [Paper(id=f"p{index}", title=f"Paper {index}") for index in range(3)]
        result = extractor.extract_many(papers)
        self.assertEqual([item.paper_id for item in result], ["p0", "p1", "p2"])
        self.assertEqual(len(responses.calls), 2)
        self.assertEqual(extractor.metrics_snapshot()["new_evidence_extractions"], 3)


if __name__ == "__main__":
    unittest.main()
