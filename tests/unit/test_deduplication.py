from datetime import datetime, timezone
import unittest

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.deduplication import deduplicate_paper_models


def paper(**updates):
    values = {
        "id": "synthetic",
        "title": "Example paper",
        "publication_year": 2024,
    }
    values.update(updates)
    return Paper(**values)


def provenance(text, mode=RetrievalMode.BROAD_LEXICAL):
    return RetrievalProvenance(
        query=SearchQuery(
            text=text,
            strategy="test",
            source="deterministic",
        ),
        provider="openalex",
        mode=mode,
        retrieved_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        provider_rank=1,
    )


class TypedDeduplicationTest(unittest.TestCase):
    def test_openalex_id_merge_retains_richer_data_and_provenance(
        self,
    ) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    id="W1",
                    openalex_id="https://openalex.org/W1",
                    abstract="Short",
                    provenance=[provenance("one")],
                ),
                paper(
                    id="w1",
                    openalex_id="openalex:w1",
                    abstract="A substantially richer abstract",
                    citation_count=8,
                    provenance=[
                        provenance(
                            "two",
                            RetrievalMode.SEMANTIC,
                        )
                    ],
                ),
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].abstract,
            "A substantially richer abstract",
        )
        self.assertEqual(
            result[0].citation_count,
            8,
        )
        self.assertEqual(
            result[0].matched_queries,
            ["one", "two"],
        )

    def test_doi_merge_normalizes_url_and_prefix(self) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    doi="https://doi.org/10.123/ABC",
                    title="One",
                ),
                paper(
                    doi="doi:10.123/abc",
                    title="Two",
                ),
            ]
        )

        self.assertEqual(len(result), 1)

    def test_title_fallback_requires_compatible_year(self) -> None:
        same = deduplicate_paper_models(
            [
                paper(title="A Useful: Paper!"),
                paper(title="a useful paper"),
            ]
        )

        different = deduplicate_paper_models(
            [
                paper(
                    title="A Useful Paper",
                    publication_year=2024,
                ),
                paper(
                    title="a useful paper",
                    publication_year=2025,
                ),
            ]
        )

        self.assertEqual(len(same), 1)
        self.assertEqual(len(different), 2)

    def test_title_fallback_requires_publication_year(self) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    id="a",
                    title="Same Paper",
                    publication_year=None,
                ),
                paper(
                    id="b",
                    title="Same Paper",
                    publication_year=None,
                ),
            ]
        )

        self.assertEqual(len(result), 2)

    def test_similar_titles_and_conflicting_ids_stay_separate(
        self,
    ) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    id="W1",
                    openalex_id="W1",
                    title="Shared title",
                ),
                paper(
                    id="W2",
                    openalex_id="W2",
                    title="Shared title",
                ),
                paper(
                    id="W3",
                    title="Shared title extended",
                ),
            ]
        )

        self.assertEqual(len(result), 3)

    def test_title_fallback_does_not_bridge_conflicting_dois(
        self,
    ) -> None:
        result = deduplicate_paper_models(
            [
                Paper(
                    id="a",
                    title="Same Paper Title",
                    publication_year=2025,
                ),
                Paper(
                    id="b",
                    title="Same Paper Title",
                    publication_year=2025,
                    doi="10.1000/one",
                ),
                Paper(
                    id="c",
                    title="Same Paper Title",
                    publication_year=2025,
                    doi="10.1000/two",
                ),
            ]
        )

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()