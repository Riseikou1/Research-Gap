from datetime import datetime, timezone
import unittest

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.deduplication import canonicalize_against, deduplicate_paper_models


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

    def test_exact_titles_reconcile_openalex_aliases_but_similar_titles_stay_separate(
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

        self.assertEqual(len(result), 2)
        self.assertEqual(
            set(result[0].openalex_aliases),
            {"https://openalex.org/W1", "https://openalex.org/W2"},
        )

    def test_three_naacl_records_become_one_canonical_work(self) -> None:
        title = "Reducing hallucination in structured outputs via Retrieval-Augmented Generation"
        result = deduplicate_paper_models([
            paper(id=openalex_id, openalex_id=openalex_id, title=title, authors=["Edoardo Serra"], abstract=abstract)
            for openalex_id, abstract in (
                ("https://openalex.org/W4394838812", "Short abstract."),
                ("https://openalex.org/W6966460441", "A substantially richer abstract describing the enterprise workflow application."),
                ("https://openalex.org/W4401042735", None),
            )
        ])

        self.assertEqual(len(result), 1)
        self.assertEqual(
            set(result[0].openalex_aliases),
            {
                "https://openalex.org/W4394838812",
                "https://openalex.org/W6966460441",
                "https://openalex.org/W4401042735",
            },
        )
        self.assertIn("substantially richer", result[0].abstract)

    def test_hyphenated_review_variants_merge_with_year_and_author_support(self) -> None:
        result = deduplicate_paper_models([
            paper(
                id="W7117820692",
                openalex_id="W7117820692",
                title="Retrieval-Augmented Generation for Enterprise Applications: A Systematic Review",
                authors=["A. Researcher"],
            ),
            paper(
                id="W4417026990",
                openalex_id="W4417026990",
                title="Retrieval Augmented Generation for Enterprise Applications - A Systematic Review",
                authors=["A. Researcher"],
            ),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].openalex_aliases), 2)

    def test_verification_alias_maps_to_initial_canonical_id(self) -> None:
        title = "Reducing hallucination in structured outputs via Retrieval-Augmented Generation"
        initial = paper(
            id="https://openalex.org/W4394838812",
            openalex_id="https://openalex.org/W4394838812",
            title=title,
            authors=["Edoardo Serra"],
        )
        verification = paper(
            id="https://openalex.org/W6966460441",
            openalex_id="https://openalex.org/W6966460441",
            title=title,
            authors=["Edoardo Serra"],
            abstract="A richer abstract returned during verification.",
        )

        result = canonicalize_against([verification], [initial])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, initial.id)
        self.assertEqual(len(result[0].openalex_aliases), 2)
        self.assertIn("richer abstract", result[0].abstract)

    def test_different_similar_reviews_remain_separate(self) -> None:
        result = deduplicate_paper_models([
            paper(
                id="W1", openalex_id="W1", authors=["A. Researcher"],
                title="A Systematic Review of Retrieval Augmented Generation for Enterprise Applications",
            ),
            paper(
                id="W2", openalex_id="W2", authors=["A. Researcher"],
                title="A Systematic Review of Retrieval Augmented Generation for Healthcare Applications",
            ),
        ])
        self.assertEqual(len(result), 2)

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
