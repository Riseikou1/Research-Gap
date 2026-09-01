import io
import os
import sys
import unittest
from unittest.mock import patch

import main
from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.models.query import SearchQuery
from src.analysis.models import IdeaAssessment
from src.pipeline import ResearchResult


class CliTest(unittest.TestCase):
    def test_openai_mode_without_api_key_has_clear_error(self) -> None:
        stderr = io.StringIO()
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(sys, "argv", ["main.py", "an idea", "--decomposer", "openai"]),
            patch("sys.stderr", stderr),
        ):
            status = main.main()
        self.assertEqual(status, 1)
        self.assertIn("OPENAI_API_KEY is required", stderr.getvalue())

    def test_hybrid_result_shows_queries_and_scores(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        result = ResearchResult(
            idea=ResearchIdea(original_text="RAG using LoRA"),
            queries=[
                SearchQuery(
                    text="RAG using LoRA",
                    strategy="original",
                    source="deterministic",
                )
            ],
            candidate_count=4,
            papers=[
                Paper(
                    id="W1",
                    title="Relevant paper",
                    lexical_score=0.7,
                    semantic_score=0.9,
                    final_score=0.82,
                    ranking_mode="hybrid",
                )
            ],
            ranking_mode="hybrid",
        )

        class FakePipeline:
            def run(self, idea, *, top_k):
                self.call = (idea, top_k)
                return result

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "RAG using LoRA",
                    "--show-queries",
                    "--show-scores",
                ],
            ),
            patch.object(main, "build_pipeline", return_value=FakePipeline()),
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
        ):
            status = main.main()

        self.assertEqual(status, 0)
        self.assertIn("Found 4 unique candidate papers", stdout.getvalue())
        self.assertIn("deterministic/original", stdout.getvalue())
        self.assertIn("Lexical: 0.700 | Semantic: 0.900", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_idea_assessment_prints_matched_facets_and_partial_matches(self) -> None:
        result = ResearchResult(
            idea=ResearchIdea(original_text="vision transformer plant disease few-shot field"),
            queries=[],
            candidate_count=2,
            papers=[],
            idea_assessment=IdeaAssessment(
                label="uncertain",
                rationale="Only partial matches were found.",
                counterexample_paper_ids=["https://openalex.org/W1"],
                partial_match_paper_ids=["https://openalex.org/W2"],
                matched_facets={
                    "https://openalex.org/W1": ["method", "problem", "constraint", "field setting"],
                    "https://openalex.org/W2": ["method", "problem"],
                },
            ),
        )
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            main.print_idea_assessment(result)
        output = stdout.getvalue()
        self.assertIn("matched: method, problem, constraint, field setting", output)
        self.assertIn("Partial/contextual support:", output)
        self.assertIn("https://openalex.org/W2", output)


if __name__ == "__main__":
    unittest.main()
