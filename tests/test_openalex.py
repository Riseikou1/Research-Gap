import unittest

from src.retrieval.openalex import normalize_work, reconstruct_abstract


class OpenAlexHelpersTest(unittest.TestCase):
    def test_reconstruct_abstract_orders_positions(self) -> None:
        index = {"research": [1], "Useful": [0], "matters.": [2]}
        self.assertEqual(reconstruct_abstract(index), "Useful research matters.")

    def test_reconstruct_abstract_handles_missing_value(self) -> None:
        self.assertIsNone(reconstruct_abstract(None))
        self.assertIsNone(reconstruct_abstract({}))


    def test_normalize_work_handles_openalex_shape(self) -> None:
        work = {
            "id": "https://openalex.org/W1",
            "display_name": "A useful paper",
            "abstract_inverted_index": {"Hello": [0], "world": [1]},
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "publication_year": 2025,
            "doi": "https://doi.org/10.1/example",
            "primary_location": {"landing_page_url": "https://example.org/paper"},
            "cited_by_count": 7,
        }
        self.assertEqual(normalize_work(work), {
            "openalex_id": "https://openalex.org/W1",
            "title": "A useful paper",
            "abstract": "Hello world",
            "authors": ["Ada Lovelace"],
            "year": 2025,
            "doi": "https://doi.org/10.1/example",
            "url": "https://example.org/paper",
            "citation_count": 7,
        })


if __name__ == "__main__":
    unittest.main()
