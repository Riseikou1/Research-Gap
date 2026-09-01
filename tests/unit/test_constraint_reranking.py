import unittest

from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.ranking.reranker import HybridReranker
from src.ranking.semantic import SemanticScoringError


class FakeScorer:
    def __init__(self, scores, failing_queries=()):
        self.scores = scores
        self.failing_queries = set(failing_queries)

    def score_many(self, query, papers):
        if query in self.failing_queries:
            raise SemanticScoringError("simulated provider failure")
        return list(self.scores[query])


class ConstraintRerankingTest(unittest.TestCase):
    papers = [Paper(id="A", title="Crop disease ViT"), Paper(id="B", title="Crop disease ViT")]

    def test_constraint_match_boosts_similarly_relevant_paper(self):
        lexical = FakeScorer({"topic": [1.0, 1.0], "limited labeled data": [0.0, 0.0]})
        semantic = FakeScorer({"topic": [1.0, 1.0], "limited labeled data": [0.0, 1.0]})
        reranker = HybridReranker(lexical, semantic, lexical_weight=.4, semantic_weight=.6)
        result = reranker.rerank(
            ResearchIdea(original_text="topic", constraints=["limited labeled data"]),
            self.papers,
        )
        self.assertEqual(result.papers[0].id, "B")
        self.assertGreater(result.papers[0].constraint_score, result.papers[1].constraint_score)

    def test_topic_relevance_remains_dominant(self):
        lexical = FakeScorer({"topic": [1.0, .2], "limited labeled data": [.4, 0.0]})
        semantic = FakeScorer({"topic": [1.0, .2], "limited labeled data": [.4, 1.0]})
        reranker = HybridReranker(lexical, semantic)
        result = reranker.rerank(
            ResearchIdea(original_text="topic", constraints=["limited labeled data"]),
            self.papers,
        )
        self.assertEqual(result.papers[0].id, "A")

    def test_no_constraints_preserve_base_scores(self):
        lexical = FakeScorer({"topic": [1.0, .2]})
        semantic = FakeScorer({"topic": [1.0, .2]})
        result = HybridReranker(lexical, semantic).rerank(
            ResearchIdea(original_text="topic"),
            self.papers,
        )
        self.assertIsNone(result.papers[0].constraint_score)
        self.assertEqual(result.papers[0].final_score, 1.0)
        self.assertEqual(result.papers[1].final_score, 0.0)

    def test_semantic_failure_uses_lexical_constraint_score(self):
        lexical = FakeScorer({"topic": [1.0, 1.0], "limited labeled data": [0.0, 1.0]})
        semantic = FakeScorer({}, failing_queries={"topic"})
        result = HybridReranker(lexical, semantic).rerank(
            ResearchIdea(original_text="topic", constraints=["limited labeled data"]),
            self.papers,
        )
        self.assertEqual(result.mode, "lexical_only")
        self.assertEqual(result.papers[0].id, "B")
        self.assertIsNotNone(result.papers[0].constraint_score)


if __name__ == "__main__":
    unittest.main()
