from datetime import datetime, timezone
import unittest

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import RetrievalError
from src.retrieval.multi_query import HybridRetrievalPlanner, MultiQueryRetriever


def query(text, strategy="original"):
    return SearchQuery(text=text, strategy=strategy, source="deterministic")


class FakeRetriever:
    provider_name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, request):
        self.calls.append(request)
        if request.query.text == "bad query":
            raise RetrievalError("temporary failure")
        identifier = "W1" if "original" in request.query.strategy else request.query.text
        return [
            Paper(
                id=identifier,
                openalex_id=identifier,
                title=f"Paper {identifier}",
                publication_year=2025,
                provenance=[
                    RetrievalProvenance(
                        query=request.query,
                        provider="fake",
                        mode=request.mode,
                        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        provider_rank=1,
                    )
                ],
            )
        ]


class HybridRoutePlannerTest(unittest.TestCase):
    def test_original_gets_three_modes_and_generated_queries_one(self) -> None:
        requests = HybridRetrievalPlanner(per_route_limit=7).build(
            [query("original idea"), query("generated terms", "method_problem")]
        )
        self.assertEqual(len(requests), 4)
        self.assertEqual(
            [request.mode for request in requests[:3]], list(RetrievalMode)
        )
        self.assertEqual(requests[3].mode, RetrievalMode.BROAD_LEXICAL)
        self.assertTrue(all(request.limit == 7 for request in requests))


class HybridMultiQueryRetrieverTest(unittest.TestCase):
    def test_deduplicates_modes_and_preserves_provenance_in_plan_order(self) -> None:
        result = MultiQueryRetriever(
            FakeRetriever(), max_candidates=20, per_route_limit=2
        ).retrieve_hybrid(
            [query("original idea"), query("generated terms", "method_problem")]
        )
        self.assertEqual(len(result.papers), 2)
        shared = next(paper for paper in result.papers if paper.openalex_id == "W1")
        self.assertEqual(shared.retrieval_modes, list(mode.value for mode in RetrievalMode))
        self.assertEqual(result.requested_routes, 4)

    def test_duplicate_queries_and_failures_are_explicit(self) -> None:
        result = MultiQueryRetriever(FakeRetriever()).retrieve_hybrid(
            [query("original idea"), query("bad query", "keyword")],
            include_semantic=False,
        )
        self.assertTrue(result.partial)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].query, "bad query")
        self.assertEqual(result.failures[0].mode, "broad_lexical")


if __name__ == "__main__":
    unittest.main()
