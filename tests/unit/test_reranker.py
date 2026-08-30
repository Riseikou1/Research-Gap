import unittest

from src.models.paper import Paper
from src.ranking.reranker import HybridReranker
from src.ranking.semantic import SemanticScoringError


class FixedLexical:
    def __init__(self, scores):
        self.scores = scores

    def score_many(self, query, papers):
        return list(self.scores)


class FixedSemantic:
    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error

    def score_many(self, query, papers):
        if self.error:
            raise self.error
        return list(self.scores)


class HybridRerankerTest(unittest.TestCase):
    def test_normalizes_fuses_and_sorts_with_configured_weights(self) -> None:
        papers = [Paper(id="lexical", title="Lexical"), Paper(id="semantic", title="Semantic")]
        reranker = HybridReranker(
            FixedLexical([10.0, 0.0]),
            FixedSemantic([0.0, 1.0]),
            lexical_weight=0.4,
            semantic_weight=0.6,
        )
        result = reranker.rerank("idea", papers)
        self.assertEqual(result.mode, "hybrid")
        self.assertEqual(result.papers[0].id, "semantic")
        self.assertAlmostEqual(result.papers[0].final_score, 0.6)
        self.assertAlmostEqual(result.papers[1].final_score, 0.4)

    def test_ties_use_deterministic_title_then_id_order(self) -> None:
        papers = [Paper(id="b", title="Zulu"), Paper(id="a", title="Alpha")]
        reranker = HybridReranker(
            FixedLexical([1.0, 1.0]),
            FixedSemantic([1.0, 1.0]),
        )
        result = reranker.rerank("idea", papers)
        self.assertEqual([paper.id for paper in result.papers], ["a", "b"])

    def test_semantic_failure_has_explicit_lexical_fallback(self) -> None:
        reranker = HybridReranker(
            FixedLexical([2.0, 1.0]),
            FixedSemantic(error=SemanticScoringError("provider down")),
            semantic_fallback="lexical",
        )
        result = reranker.rerank(
            "idea", [Paper(id="a", title="A"), Paper(id="b", title="B")]
        )
        self.assertEqual(result.mode, "lexical_only")
        self.assertIsNone(result.papers[0].semantic_score)
        self.assertIn("provider down", result.notice)

    def test_missing_semantic_component_can_be_strict(self) -> None:
        reranker = HybridReranker(
            FixedLexical([1.0]), None, semantic_fallback="error"
        )
        with self.assertRaisesRegex(SemanticScoringError, "unavailable"):
            reranker.rerank("idea", [Paper(id="a", title="A")])


if __name__ == "__main__":
    unittest.main()
