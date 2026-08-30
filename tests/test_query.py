import unittest

from src.models.idea import ResearchIdea
from src.query.deterministic import DeterministicDecomposer
from src.query.generator import generate_queries


class ResearchIdeaTest(unittest.TestCase):
    def test_defaults_and_text_normalization(self) -> None:
        idea = ResearchIdea(original_text="  retrieval   augmented generation  ")

        self.assertEqual(idea.original_text, "retrieval augmented generation")
        self.assertEqual(idea.problem, [])
        self.assertEqual(idea.synonyms, {})

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted"):
            ResearchIdea.model_validate({"original_text": "RAG", "novelty": True})


class DeterministicDecomposerTest(unittest.TestCase):
    def test_empty_idea_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            DeterministicDecomposer().decompose("  ")

    def test_decomposition_is_stable_and_uses_explicit_connectors(self) -> None:
        decomposer = DeterministicDecomposer()
        text = "  Feedback   using LLMs for language learners without teacher labels!!! "
        first = decomposer.decompose(text)
        second = decomposer.decompose(text)
        self.assertEqual(first, second)
        self.assertEqual(first.original_text, "Feedback using LLMs for language learners without teacher labels")
        self.assertEqual(first.problem, ["Feedback"])
        self.assertEqual(first.intervention_or_method, ["LLMs"])
        self.assertEqual(first.population, ["language learners"])
        self.assertEqual(first.constraints, ["teacher labels"])

    def test_missing_facets_stay_empty(self) -> None:
        result = DeterministicDecomposer().decompose("machine learning")
        self.assertEqual(result.problem, ["machine learning"])
        self.assertEqual(result.population, [])
        self.assertEqual(result.intervention_or_method, [])
        self.assertEqual(result.comparison, [])
        self.assertEqual(result.outcomes, [])
        self.assertEqual(result.domain, [])
        self.assertEqual(result.constraints, [])
        self.assertEqual(result.synonyms, {})


class QueryGeneratorTest(unittest.TestCase):
    def test_queries_are_bounded_and_casefolded_duplicates_are_removed(self) -> None:
        idea = ResearchIdea(
            original_text="RAG using LoRA",
            problem=["RAG"],
            population=["medical domain"],
            intervention_or_method=["LoRA"],
            outcomes=["accuracy"],
            keywords=["rag", "using", "lora"],
            synonyms={
                "RAG": ["retrieval augmented generation"],
                "LoRA": ["low-rank adaptation"],
            },
        )
        queries = generate_queries(idea)
        self.assertLessEqual(len(queries), 6)
        self.assertEqual(len(queries), len({query.casefold() for query in queries}))

    def test_synonyms_use_explicit_or_groups(self) -> None:
        idea = ResearchIdea(
            original_text="RAG with LoRA",
            synonyms={
                "RAG": ["retrieval augmented generation"],
                "LoRA": ["low-rank adaptation"],
            },
        )
        self.assertIn(
            '(RAG OR "retrieval augmented generation") AND (LoRA OR "low-rank adaptation")',
            generate_queries(idea),
        )

    def test_single_term_generated_queries_are_omitted(self) -> None:
        idea = ResearchIdea(
            original_text="AI",
            problem=["AI"],
            intervention_or_method=["ML"],
            outcomes=["speed"],
        )
        queries = generate_queries(idea)
        self.assertEqual(queries[0], "AI")
        self.assertNotIn("speed", queries)


if __name__ == "__main__":
    unittest.main()
