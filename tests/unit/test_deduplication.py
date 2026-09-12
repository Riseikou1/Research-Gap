from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.deduplication import canonicalize_against, deduplicate_paper_models
from src.retrieval.openalex import _parse_work


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
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = (
            Path(__file__).parents[1]
            / "fixtures"
            / "production_openalex_aliases.json"
        )
        cls.production_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

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

    def test_missing_doi_does_not_conflict_with_present_doi(self) -> None:
        result = deduplicate_paper_models([
            paper(id="missing", title="Same scholarly work", doi=None),
            paper(id="present", title="Same scholarly work", doi="10.1234/work"),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].doi, "10.1234/work")

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
        works = self.production_fixture["works"][:3]
        result = deduplicate_paper_models([_parse_work(work) for work in works])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "https://openalex.org/W4401042735")
        self.assertEqual(
            set(result[0].openalex_aliases),
            {
                "https://openalex.org/W4394838812",
                "https://openalex.org/W6966460441",
                "https://openalex.org/W4401042735",
            },
        )
        self.assertEqual(len(result[0].doi_aliases), 3)
        self.assertEqual(len(result[0].full_text_locations), 6)
        self.assertEqual(result[0].work_type, "conference-paper")
        self.assertIn("enterprise workflows", result[0].abstract)
        self.assertEqual(
            result[0].url,
            "https://doi.org/10.18653/v1/2024.naacl-industry.19",
        )
        self.assertTrue(any("Patrice" in author for author in result[0].authors))
        self.assertTrue(any("Orlando" in author for author in result[0].authors))

    def test_production_preprint_and_version_of_record_pairs_merge(self) -> None:
        works = self.production_fixture["works"]
        enterprise = deduplicate_paper_models([_parse_work(item) for item in works[3:5]])
        healthcare = deduplicate_paper_models([_parse_work(item) for item in works[5:7]])

        self.assertEqual(len(enterprise), 1)
        self.assertEqual(enterprise[0].id, "https://openalex.org/W7117820692")
        self.assertEqual(len(healthcare), 1)
        self.assertEqual(healthcare[0].id, "https://openalex.org/W4414128336")

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

    def test_production_verification_alias_uses_initial_canonical_identity(self) -> None:
        initial = _parse_work(self.production_fixture["works"][0])
        verification_alias = _parse_work(self.production_fixture["works"][2])

        result = canonicalize_against([verification_alias], [initial])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, initial.id)
        self.assertEqual(result[0].openalex_id, initial.openalex_id)
        self.assertEqual(len(result[0].openalex_aliases), 2)
        self.assertEqual(len(result[0].doi_aliases), 2)

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

    def test_different_dois_with_different_authors_remain_distinct(self) -> None:
        result = deduplicate_paper_models([
            paper(
                id="one", title="Shared research title", doi="10.1000/one",
                authors=["Ada Lovelace", "Grace Hopper"],
            ),
            paper(
                id="two", title="Shared research title", doi="10.1000/two",
                authors=["Alan Turing", "Claude Shannon"],
            ),
        ])
        self.assertEqual(len(result), 2)

    def test_exact_title_year_normalization_handles_unicode_and_spacing(self) -> None:
        result = deduplicate_paper_models([
            paper(id="one", title="  MÉTHODS—For   Retrieval: A Study!  "),
            paper(id="two", title="méthods for retrieval a study"),
        ])
        self.assertEqual(len(result), 1)

    def test_near_title_requires_year_similarity_and_strong_authorship(self) -> None:
        base = paper(
            id="base",
            title=(
                "A systematic review of retrieval augmented generation applications "
                "for clinical decision support and medical question answering in "
                "large healthcare systems"
            ),
            authors=["Fnu Neha", "Deepshikha Bhati", "Deepak Kumar Shukla"],
        )
        close = base.model_copy(update={
            "id": "close",
            "title": (
                "A systematic review of retrieval augmented generation application "
                "for clinical decision support and medical question answering in "
                "large healthcare systems"
            ),
        })
        wrong_authors = close.model_copy(update={
            "id": "wrong-authors",
            "authors": ["Different Author", "Another Scholar"],
        })
        wrong_year = close.model_copy(update={"id": "wrong-year", "publication_year": 2023})

        self.assertEqual(len(deduplicate_paper_models([base, close])), 1)
        self.assertEqual(len(deduplicate_paper_models([base, wrong_authors])), 2)
        self.assertEqual(len(deduplicate_paper_models([base, wrong_year])), 2)


if __name__ == "__main__":
    unittest.main()
