from __future__ import annotations

import unittest

from src.models.paper import Paper
from src.retrieval.deduplication import deduplicate_paper_models


def paper(
    *,
    paper_id: str,
    title: str = "Example paper",
    openalex_id: str | None = None,
    doi: str | None = None,
    year: int | None = None,
    abstract: str | None = None,
    citation_count: int = 0,
) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        openalex_id=openalex_id,
        doi=doi,
        publication_year=year,
        abstract=abstract,
        citation_count=citation_count,
    )


class DeduplicationTest(unittest.TestCase):
    def test_duplicate_openalex_ids_merge(self) -> None:
        first = paper(
            paper_id="first",
            openalex_id="https://openalex.org/W1",
            abstract="Useful abstract",
            citation_count=4,
        )

        second = paper(
            paper_id="second",
            openalex_id="https://openalex.org/w1/",
            citation_count=7,
        )

        result = deduplicate_paper_models(
            [first, second]
        )

        self.assertEqual(
            len(result),
            1,
        )

        self.assertEqual(
            result[0].abstract,
            "Useful abstract",
        )

        self.assertEqual(
            result[0].citation_count,
            7,
        )

    def test_doi_urls_are_normalized(self) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    paper_id="first",
                    title="First title",
                    doi="https://doi.org/10.123/ABC",
                ),
                paper(
                    paper_id="second",
                    title="Second title",
                    doi="doi:10.123/abc",
                ),
            ]
        )

        self.assertEqual(
            len(result),
            1,
        )

    def test_title_fallback_ignores_case_and_punctuation(
        self,
    ) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    paper_id="first",
                    title="A Useful: Paper!",
                    year=2025,
                ),
                paper(
                    paper_id="second",
                    title="a useful paper",
                    year=2025,
                ),
            ]
        )

        self.assertEqual(
            len(result),
            1,
        )

    def test_title_fallback_does_not_bridge_conflicting_dois(
        self,
    ) -> None:
        result = deduplicate_paper_models(
            [
                paper(
                    paper_id="unknown",
                    title="Same Paper",
                    year=2025,
                ),
                paper(
                    paper_id="doi-one",
                    title="Same Paper",
                    year=2025,
                    doi="10.1000/one",
                ),
                paper(
                    paper_id="doi-two",
                    title="Same Paper",
                    year=2025,
                    doi="10.1000/two",
                ),
            ]
        )

        self.assertEqual(
            len(result),
            2,
        )


if __name__ == "__main__":
    unittest.main()