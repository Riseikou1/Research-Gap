import unittest

from src.models.paper import Paper
from src.ranking.lexical import LexicalScorer


class LexicalScorerTest(unittest.TestCase):
    def test_relevant_paper_scores_above_irrelevant_paper(self) -> None:
        scorer = LexicalScorer()
        relevant = Paper(
            id="relevant",
            title="Retrieval augmented generation with low rank adaptation",
            abstract="Parameter-efficient adaptation for RAG systems.",
        )
        irrelevant = Paper(
            id="irrelevant",
            title="Marine ecosystem temperature trends",
            abstract="A longitudinal field survey.",
        )
        scores = scorer.score_many("RAG using LoRA", [relevant, irrelevant])
        self.assertGreater(scores[0], scores[1])

    def test_is_deterministic_with_empty_abstract_and_punctuation(self) -> None:
        paper = Paper(id="one", title="RAG/LoRA: An Evaluation!", abstract=None)
        scorer = LexicalScorer()
        first = scorer.score("rag lora", paper)
        second = scorer.score("RAG, LoRA", paper)
        self.assertEqual(first, second)
        self.assertGreater(first, 0)


if __name__ == "__main__":
    unittest.main()
