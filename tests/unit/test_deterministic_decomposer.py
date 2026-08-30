import unittest

from src.query.deterministic import (
    DeterministicDecomposer,
    clean_idea_text,
)


class TestDeterministicDecomposer(unittest.TestCase):

    def setUp(self):
        self.decomposer = DeterministicDecomposer()

    def test_rejects_empty_idea(self):
        with self.assertRaises(ValueError):
            self.decomposer.decompose("   ")

    def test_normalizes_whitespace(self):
        result = self.decomposer.decompose(
            "  RAG    for   medical question answering "
        )

        self.assertEqual(
            result.original_text,
            "RAG for medical question answering",
        )

    def test_for_task_is_problem_not_population(self):
        result = self.decomposer.decompose(
            "RAG for medical question answering"
        )

        self.assertIn(
            "medical question answering",
            result.problem,
        )

        self.assertEqual(result.population, [])

    def test_for_population(self):
        result = self.decomposer.decompose(
            "depression detection for adolescents"
        )

        self.assertIn(
            "adolescents",
            result.population,
        )

    def test_using_method_to_problem(self):
        result = self.decomposer.decompose(
            "Using transformers to detect pneumonia"
        )

        self.assertIn(
            "transformers",
            result.intervention_or_method,
        )

        self.assertIn(
            "detect pneumonia",
            result.problem,
        )

    def test_using_method_to_outcome(self):
        result = self.decomposer.decompose(
            "Using reinforcement learning to improve retrieval accuracy"
        )

        self.assertIn(
            "reinforcement learning",
            result.intervention_or_method,
        )

        self.assertIn(
            "improve retrieval accuracy",
            result.outcomes,
        )

    def test_multiple_facets(self):
        result = self.decomposer.decompose(
            "Detecting cancer using transformers among elderly patients "
            "in hospitals without labeled data"
        )

        self.assertIn(
            "Detecting cancer",
            result.problem,
        )

        self.assertIn(
            "transformers",
            result.intervention_or_method,
        )

        self.assertIn(
            "elderly patients",
            result.population,
        )

        self.assertIn(
            "hospitals",
            result.domain,
        )

        self.assertIn(
            "labeled data",
            result.constraints,
        )

    def test_preserves_short_acronyms_as_keywords(self):
        result = self.decomposer.decompose(
            "Using RL to improve RAG"
        )

        self.assertIn("RL", result.keywords)
        self.assertIn("RAG", result.keywords)

    def test_extracts_explicit_acronym_synonym(self):
        result = self.decomposer.decompose(
            "retrieval augmented generation (RAG)"
        )

        self.assertEqual(
            result.synonyms,
            {
                "retrieval augmented generation": ["RAG"]
            },
        )

    def test_keywords_are_bounded(self):
        result = self.decomposer.decompose(
            "machine learning neural network transformer retrieval "
            "generation medical healthcare evaluation prediction "
            "classification optimization representation embeddings "
            "language vision multimodal reasoning"
        )

        self.assertLessEqual(
            len(result.keywords),
            12,
        )

    def test_is_deterministic(self):
        text = (
            "Using reinforcement learning to improve retrieval accuracy "
            "in healthcare"
        )

        first = self.decomposer.decompose(text)
        second = self.decomposer.decompose(text)

        self.assertEqual(
            first.model_dump(),
            second.model_dump(),
        )


if __name__ == "__main__":
    unittest.main()