from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import RetrievalRequest
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.openalex import OpenAlexError


TEST_TIME = datetime(
    2026,
    1,
    2,
    tzinfo=timezone.utc,
)


def query(
    text: str,
    *,
    strategy: str = "test",
    source: str = "deterministic",
) -> SearchQuery:
    return SearchQuery(
        text=text,
        strategy=strategy,
        source=source,
    )


def paper_for_request(
    request: RetrievalRequest,
    *,
    paper_id: str,
    title: str = "Example paper",
    rank: int = 1,
) -> Paper:
    return Paper(
        id=paper_id,
        openalex_id=paper_id,
        title=title,
        provenance=[
            RetrievalProvenance(
                query=request.query,
                provider="fake",
                mode=request.mode,
                retrieved_at=TEST_TIME,
                provider_rank=rank,
            )
        ],
    )


class SharedPaperClient:
    """Return the same paper from every successful retrieval route."""

    provider_name = "fake"

    def __init__(self) -> None:
        self.calls: list[RetrievalRequest] = []

    def search(
        self,
        request: RetrievalRequest,
    ) -> list[Paper]:
        self.calls.append(request)

        if request.query.text == "bad query":
            raise OpenAlexError("temporary failure")

        return [
            paper_for_request(
                request,
                paper_id="W1",
                title="Shared paper",
            )
        ]


class UniquePaperClient:
    """Return unique papers for each retrieval route."""

    provider_name = "fake"

    def __init__(self) -> None:
        self.calls: list[RetrievalRequest] = []

    def search(
        self,
        request: RetrievalRequest,
    ) -> list[Paper]:
        self.calls.append(request)

        papers: list[Paper] = []

        for rank in range(1, request.limit + 1):
            paper_id = (
                f"{request.query.text}:"
                f"{request.mode.value}:"
                f"{rank}"
            )

            papers.append(
                paper_for_request(
                    request,
                    paper_id=paper_id,
                    title=paper_id,
                    rank=rank,
                )
            )

        return papers


class MultiQueryRetrieverTest(unittest.TestCase):
    def test_builds_expected_hybrid_routes(self) -> None:
        client = UniquePaperClient()

        retriever = MultiQueryRetriever(
            client,
            max_candidates=20,
            per_route_limit=2,
        )

        queries = [
            query(
                "original query",
                strategy="original",
            ),
            query(
                "generated query",
                strategy="method_problem",
            ),
        ]

        result = retriever.retrieve_hybrid(
            queries
        )

        self.assertEqual(
            result.requested_routes,
            4,
        )

        routes = [
            (
                request.query.text,
                request.mode,
            )
            for request in client.calls
        ]

        self.assertIn(
            (
                "original query",
                RetrievalMode.BROAD_LEXICAL,
            ),
            routes,
        )

        self.assertIn(
            (
                "original query",
                RetrievalMode.TITLE_ABSTRACT,
            ),
            routes,
        )

        self.assertIn(
            (
                "original query",
                RetrievalMode.SEMANTIC,
            ),
            routes,
        )

        self.assertIn(
            (
                "generated query",
                RetrievalMode.BROAD_LEXICAL,
            ),
            routes,
        )

    def test_semantic_route_is_always_included_for_original_query(
        self,
    ) -> None:
        client = UniquePaperClient()

        retriever = MultiQueryRetriever(
            client,
            max_candidates=10,
            per_route_limit=1,
        )

        retriever.retrieve_hybrid(
            [
                query(
                    "original query",
                    strategy="original",
                )
            ]
        )

        modes = {
            request.mode
            for request in client.calls
        }

        self.assertEqual(
            modes,
            {
                RetrievalMode.BROAD_LEXICAL,
                RetrievalMode.TITLE_ABSTRACT,
                RetrievalMode.SEMANTIC,
            },
        )

    def test_failure_does_not_discard_successful_routes(
        self,
    ) -> None:
        client = SharedPaperClient()

        retriever = MultiQueryRetriever(
            client,
            max_candidates=20,
            per_route_limit=2,
        )

        result = retriever.retrieve_hybrid(
            [
                query(
                    "first query",
                    strategy="original",
                ),
                query(
                    "bad query",
                    strategy="method_problem",
                ),
                query(
                    "second query",
                    strategy="problem_context",
                ),
            ]
        )

        self.assertTrue(
            result.partial
        )

        self.assertEqual(
            len(result.failures),
            1,
        )

        self.assertEqual(
            result.failures[0].query,
            "bad query",
        )

        self.assertEqual(
            len(result.papers),
            1,
        )

        # The same paper was returned through several routes.
        # Deduplication should merge them while keeping query provenance.
        self.assertEqual(
            result.papers[0].matched_queries,
            [
                "first query",
                "second query",
            ],
        )

    def test_candidate_ceiling_is_respected(
        self,
    ) -> None:
        client = UniquePaperClient()

        retriever = MultiQueryRetriever(
            client,
            max_candidates=5,
            per_route_limit=3,
        )

        result = retriever.retrieve_hybrid(
            [
                query(
                    "original",
                    strategy="original",
                ),
                query(
                    "generated one",
                    strategy="method_problem",
                ),
                query(
                    "generated two",
                    strategy="problem_context",
                ),
            ]
        )

        self.assertEqual(
            len(result.papers),
            5,
        )

    def test_routes_are_interleaved_before_candidate_ceiling(
        self,
    ) -> None:
        client = UniquePaperClient()

        retriever = MultiQueryRetriever(
            client,
            max_candidates=5,
            per_route_limit=2,
        )

        result = retriever.retrieve_hybrid(
            [
                query(
                    "original",
                    strategy="original",
                ),
                query(
                    "generated one",
                    strategy="method_problem",
                ),
                query(
                    "generated two",
                    strategy="problem_context",
                ),
            ]
        )

        ids = {
            paper.id
            for paper in result.papers
        }

        # There are five routes:
        #
        # original/broad
        # original/title+abstract
        # original/semantic
        # generated one/broad
        # generated two/broad
        #
        # With fair interleaving and a ceiling of five, rank 1 from every
        # route should survive before rank 2 from any route.
        self.assertEqual(
            len(ids),
            5,
        )

        self.assertIn(
            "generated one:broad_lexical:1",
            ids,
        )

        self.assertIn(
            "generated two:broad_lexical:1",
            ids,
        )

        self.assertNotIn(
            "original:broad_lexical:2",
            ids,
        )

    def test_empty_query_plan_is_rejected(
        self,
    ) -> None:
        client = UniquePaperClient()

        retriever = MultiQueryRetriever(
            client
        )

        with self.assertRaises(
            ValueError
        ):
            retriever.retrieve_hybrid([])


if __name__ == "__main__":
    unittest.main()