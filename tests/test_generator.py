import unittest

from src.models.idea import ResearchIdea
from src.query.generator import generate_queries


class QueryGeneratorTest(unittest.TestCase):

    def test_original_query_is_preserved(self) -> None:
        idea = ResearchIdea(
            original_text="RAG for medical question answering"
        )

        queries = generate_queries(idea)

        self.assertEqual(
            queries[0],
            "RAG for medical question answering",
        )

    def test_query_count_is_bounded(self) -> None:
        idea = ResearchIdea(
            original_text="RAG for medical QA",
            problem=["medical question answering"],
            population=["patients"],
            intervention_or_method=[
                "retrieval augmented generation"
            ],
            outcomes=["improve accuracy"],
            domain=["healthcare"],
            keywords=[
                "RAG",
                "medical QA",
                "retrieval",
                "healthcare",
                "accuracy",
                "LLM",
            ],
            synonyms={
                "retrieval augmented generation": ["RAG"],
                "large language model": ["LLM"],
            },
        )

        queries = generate_queries(idea)

        self.assertLessEqual(
            len(queries),
            6,
        )

    def test_duplicate_queries_are_removed(self) -> None:
        idea = ResearchIdea(
            original_text="RAG medical QA",
            problem=["medical QA"],
            intervention_or_method=["RAG"],
            keywords=["RAG", "medical QA"],
        )

        queries = generate_queries(idea)

        normalized = [
            query.casefold()
            for query in queries
        ]

        self.assertEqual(
            len(normalized),
            len(set(normalized)),
        )

    def test_internal_duplicate_terms_are_removed(self) -> None:
        idea = ResearchIdea(
            original_text="RAG research",
            problem=[
                "medical QA",
                "Medical QA",
            ],
            intervention_or_method=[
                "RAG",
                "rag",
            ],
        )

        queries = generate_queries(idea)

        self.assertIn(
            "RAG medical QA",
            queries,
        )

    def test_synonyms_use_boolean_groups(self) -> None:
        idea = ResearchIdea(
            original_text="RAG with LoRA",
            synonyms={
                "retrieval augmented generation": [
                    "RAG"
                ],
                "low-rank adaptation": [
                    "LoRA"
                ],
            },
        )

        queries = generate_queries(idea)

        synonym_query = next(
            query
            for query in queries
            if " OR " in query
        )

        self.assertIn(
            '("retrieval augmented generation" OR RAG)',
            synonym_query,
        )

        self.assertIn(
            '("low-rank adaptation" OR LoRA)',
            synonym_query,
        )

    def test_duplicate_synonyms_are_removed(self) -> None:
        idea = ResearchIdea(
            original_text="RAG research",
            synonyms={
                "RAG": [
                    "rag",
                    "retrieval augmented generation",
                    "Retrieval Augmented Generation",
                ]
            },
        )

        queries = generate_queries(idea)

        synonym_query = next(
            query
            for query in queries
            if " OR " in query
        )

        self.assertEqual(
            synonym_query,
            '(RAG OR "retrieval augmented generation")',
        )

    def test_keyword_query_preserves_phrases(self) -> None:
        idea = ResearchIdea(
            original_text="AI research",
            keywords=[
                "reinforcement learning",
                "retrieval augmented generation",
            ],
        )

        queries = generate_queries(idea)

        self.assertIn(
            "reinforcement learning retrieval augmented generation",
            queries,
        )

    def test_invalid_max_queries_fails(self) -> None:
        idea = ResearchIdea(
            original_text="RAG research"
        )

        with self.assertRaises(ValueError):
            generate_queries(
                idea,
                max_queries=7,
            )


if __name__ == "__main__":
    unittest.main()