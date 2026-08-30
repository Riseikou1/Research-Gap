from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import QueryOrigin, RetrievalMode, SearchQuery


class SearchQueryModelTest(unittest.TestCase):
    def test_normalizes_text_and_retains_all_origins(self) -> None:
        query = SearchQuery(
            text="  RAG   using LoRA ",
            strategy="original",
            source="deterministic",
            origins=[
                QueryOrigin(
                    strategy="conceptual_reformulation",
                    source="llm",
                    provider="openai",
                )
            ],
        )
        self.assertEqual(query.text, "RAG using LoRA")
        self.assertEqual(len(query.origins), 2)

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchQuery.model_validate(
                {
                    "text": "RAG LoRA",
                    "strategy": "original",
                    "source": "deterministic",
                    "unknown": True,
                }
            )


class PaperModelTest(unittest.TestCase):
    def test_computed_provenance_fields_are_deterministic(self) -> None:
        query = SearchQuery(
            text="RAG LoRA", strategy="original", source="deterministic"
        )
        paper = Paper(
            id="W1",
            title="Paper",
            provenance=[
                RetrievalProvenance(
                    query=query,
                    provider="openalex",
                    mode=RetrievalMode.BROAD_LEXICAL,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    provider_rank=1,
                ),
                RetrievalProvenance(
                    query=query,
                    provider="openalex",
                    mode=RetrievalMode.SEMANTIC,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    provider_rank=2,
                ),
            ],
        )
        self.assertEqual(paper.matched_queries, ["RAG LoRA"])
        self.assertEqual(
            paper.retrieval_modes, ["broad_lexical", "semantic"]
        )
        self.assertEqual(paper.retrieved_by, ["openalex"])


if __name__ == "__main__":
    unittest.main()
