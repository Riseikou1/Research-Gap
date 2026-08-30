import unittest

from src.models.idea import ResearchIdea
from src.models.query import SearchQuery
from src.query.generator import DeterministicQueryGenerator
from src.query.planner import QueryPlanner


def query(text: str, strategy: str, source: str = "deterministic") -> SearchQuery:
    return SearchQuery(
        text=text,
        strategy=strategy,
        source=source,
        provider="openai" if source == "llm" else None,
    )


class QueryPlannerTest(unittest.TestCase):
    def test_keeps_original_and_bounds_combined_plan(self) -> None:
        idea = ResearchIdea(original_text="RAG using LoRA")
        deterministic = [
            query("RAG using LoRA", "original"),
            query("LoRA RAG", "method_problem"),
            query("retrieval augmented generation", "keywords"),
            query("low rank adaptation retrieval", "synonyms"),
        ]
        llm = [
            query("parameter efficient RAG", "terminology_expansion", "llm"),
            query("retrieval systems with adapters", "conceptual", "llm"),
            query("LoRA retrieval adaptation", "method", "llm"),
        ]
        result = QueryPlanner().plan(idea, deterministic, llm)
        self.assertEqual(result[0].text, "RAG using LoRA")
        self.assertLessEqual(len(result), 6)
        self.assertEqual(sum(item.source == "llm" for item in result), 3)

    def test_duplicate_deterministic_and_llm_queries_merge_origins(self) -> None:
        idea = ResearchIdea(original_text="RAG using LoRA")
        duplicate = "retrieval augmented generation LoRA"
        result = QueryPlanner().plan(
            idea,
            [query("RAG using LoRA", "original"), query(duplicate, "keywords")],
            [query(f" {duplicate.upper()} ", "conceptual", "llm")],
        )
        merged = next(item for item in result if item.text == duplicate)
        self.assertEqual(
            {origin.source for origin in merged.origins},
            {"deterministic", "llm"},
        )

    def test_empty_and_one_term_generated_queries_do_not_enter_plan(self) -> None:
        idea = ResearchIdea(original_text="AI")
        result = QueryPlanner().plan(
            idea,
            [query("AI", "original"), query("ML", "keyword")],
        )
        self.assertEqual([item.text for item in result], ["AI"])

    def test_deterministic_generator_returns_typed_stable_queries(self) -> None:
        idea = ResearchIdea(
            original_text="RAG using LoRA",
            problem=["RAG"],
            intervention_or_method=["LoRA"],
        )
        generator = DeterministicQueryGenerator()
        self.assertEqual(generator.generate(idea), generator.generate(idea))
        self.assertTrue(all(item.source == "deterministic" for item in generator.generate(idea)))


if __name__ == "__main__":
    unittest.main()
