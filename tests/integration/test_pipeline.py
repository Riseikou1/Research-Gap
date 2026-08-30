from datetime import datetime, timezone
import unittest

from src.models.paper import Paper, RetrievalProvenance
from src.pipeline import ResearchPipeline
from src.query.deterministic import DeterministicDecomposer
from src.ranking.lexical import LexicalScorer
from src.ranking.reranker import HybridReranker
from src.ranking.semantic import SemanticScorer
from src.retrieval.multi_query import MultiQueryRetriever


class FakePaperRetriever:
    provider_name = "fake"

    def search(self, request):
        shared = Paper(
            id="relevant",
            openalex_id="relevant",
            title="Parameter-efficient retrieval augmented generation",
            abstract="Low-rank adaptation of retrieval systems.",
            publication_year=2025,
        )
        superficial = Paper(
            id="superficial",
            openalex_id="superficial",
            title="RAG LoRA acronym frequency survey",
            abstract="A lexical catalog unrelated to model adaptation.",
            publication_year=2024,
        )
        papers = [shared, superficial] if request.query.strategy == "original" else [shared]
        for rank, paper in enumerate(papers, start=1):
            paper.provenance = [
                RetrievalProvenance(
                    query=request.query,
                    provider="fake",
                    mode=request.mode,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    provider_rank=rank,
                )
            ]
        return papers


class MeaningEmbeddingProvider:
    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            if text == "RAG using LoRA" or text.startswith("Parameter-efficient"):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class PipelineIntegrationTest(unittest.TestCase):
    def test_relevant_semantic_paper_outranks_superficial_match(self) -> None:
        pipeline = ResearchPipeline(
            decomposer=DeterministicDecomposer(),
            retriever=MultiQueryRetriever(
                FakePaperRetriever(),
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            reranker=HybridReranker(
                LexicalScorer(),
                SemanticScorer(MeaningEmbeddingProvider()),
                lexical_weight=0.2,
                semantic_weight=0.8,
            ),
        )
        result = pipeline.run("RAG using LoRA", top_k=10)
        self.assertGreater(len(result.queries), 1)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.ranking_mode, "hybrid")
        self.assertEqual(result.papers[0].id, "relevant")
        self.assertGreater(len(result.papers[0].provenance), 1)


if __name__ == "__main__":
    unittest.main()
