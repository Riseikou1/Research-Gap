from datetime import datetime, timezone
from io import BytesIO
import json
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import RetrievalConfigurationError, RetrievalRequest
from src.retrieval.openalex import OpenAlexError, OpenAlexRetriever, normalize_work


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def response(payload):
    return FakeResponse(json.dumps(payload).encode())


def request(mode=RetrievalMode.BROAD_LEXICAL, limit=5):
    return RetrievalRequest(
        query=SearchQuery(
            text="RAG using LoRA", strategy="original", source="deterministic"
        ),
        mode=mode,
        limit=limit,
    )


class OpenAlexParsingTest(unittest.TestCase):
    def test_normalizes_all_supported_fields(self) -> None:
        value = normalize_work(
            {
                "id": "https://openalex.org/W1",
                "display_name": "A paper",
                "abstract_inverted_index": {"Useful": [0], "work": [1]},
                "authorships": [{"author": {"display_name": "Ada"}}],
                "publication_year": 2025,
                "publication_date": "2025-02-03",
                "doi": "https://doi.org/10.1/test",
                "primary_location": {
                    "landing_page_url": "https://example.test/paper",
                    "source": {"display_name": "Journal"},
                },
                "cited_by_count": 9,
            }
        )
        self.assertEqual(value["title"], "A paper")
        self.assertEqual(value["abstract"], "Useful work")
        self.assertEqual(value["authors"], ["Ada"])
        self.assertEqual(value["doi"], "https://doi.org/10.1/test")

    def test_malformed_optional_fields_are_treated_as_missing(self) -> None:
        value = normalize_work(
            {
                "display_name": "Sparse paper",
                "abstract_inverted_index": "bad",
                "authorships": {"bad": "shape"},
                "primary_location": "bad",
                "publication_year": "2025",
                "cited_by_count": "many",
            }
        )
        self.assertIsNone(value["abstract"])
        self.assertEqual(value["authors"], [])
        self.assertIsNone(value["year"])
        self.assertIsNone(value["doi"])
        self.assertEqual(value["citation_count"], 0)


class OpenAlexRetrieverTest(unittest.TestCase):
    def test_builds_each_search_mode_and_preserves_provider_features(self) -> None:
        urls = []

        def opener(req, timeout):
            urls.append(req.full_url)
            return response(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "display_name": "Paper",
                            "relevance_score": 12.5,
                        }
                    ]
                }
            )

        retriever = OpenAlexRetriever(
            api_key="test-key",
            opener=opener,
            clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        modes = list(RetrievalMode)
        papers = [retriever.search(request(mode, 1))[0] for mode in modes]
        params = [parse_qs(urlparse(url).query) for url in urls]
        self.assertIn("search", params[0])
        self.assertIn("filter", params[1])
        self.assertIn("search.semantic", params[2])
        self.assertEqual(
            [paper.provenance[0].mode for paper in papers], modes
        )
        self.assertEqual(papers[0].provenance[0].provider_score, 12.5)

    def test_semantic_search_without_key_fails_explicitly(self) -> None:
        retriever = OpenAlexRetriever(opener=lambda *_a, **_k: response({}))
        with self.assertRaisesRegex(RetrievalConfigurationError, "OPENALEX_API_KEY"):
            retriever.search(request(RetrievalMode.SEMANTIC))

    def test_retries_rate_limit_then_succeeds(self) -> None:
        calls = 0
        sleeps = []

        def opener(req, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise HTTPError(req.full_url, 429, "rate limited", {}, None)
            return response({"results": []})

        retriever = OpenAlexRetriever(
            opener=opener,
            sleeper=sleeps.append,
            max_retries=1,
            backoff_seconds=0.25,
        )
        self.assertEqual(retriever.search(request()), [])
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [0.25])

    def test_malformed_payload_raises_provider_error(self) -> None:
        retriever = OpenAlexRetriever(opener=lambda *_a, **_k: response([]))
        with self.assertRaisesRegex(OpenAlexError, "non-object"):
            retriever.search(request())


if __name__ == "__main__":
    unittest.main()
