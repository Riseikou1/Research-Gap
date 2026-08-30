import unittest

from src.evaluation import AblationVariant, evaluate_ranking


class EvaluationMetricsTest(unittest.TestCase):
    def test_binary_retrieval_metrics(self) -> None:
        metrics = evaluate_ranking(["x", "relevant-1", "y"], {"relevant-1", "relevant-2"})
        self.assertEqual(metrics.recall_at_10, 0.5)
        self.assertEqual(metrics.recall_at_50, 0.5)
        self.assertEqual(metrics.mrr, 0.5)
        self.assertGreater(metrics.ndcg_at_10, 0)

    def test_empty_judgment_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_ranking([], set())

    def test_all_six_ablation_variants_are_stable(self) -> None:
        self.assertEqual(len(AblationVariant), 6)
        self.assertIn(AblationVariant.HYBRID_RERANKED, AblationVariant)


if __name__ == "__main__":
    unittest.main()
