import unittest

from src.models.paper import Paper
from src.ranking.semantic import (
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
    SemanticScorer,
)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        result = []
        for text in texts:
            if text == "RAG using LoRA" or "Relevant" in text:
                result.append([1.0, 0.0])
            else:
                result.append([0.0, 1.0])
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class SemanticScorerTest(unittest.TestCase):
    def test_similarity_is_real_cosine_and_documents_are_batched(self) -> None:
        provider = FakeProvider()
        papers = [
            Paper(id="r", title="Relevant methods", abstract=None),
            Paper(id="i", title="Ocean temperatures", abstract="Unrelated"),
        ]
        scores = SemanticScorer(provider).score_many("RAG using LoRA", papers)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.calls[0]), 3)


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append(list(input))
        data = [
            type("Item", (), {"index": index, "embedding": [float(index), 1.0]})()
            for index in range(len(input))
        ]
        return type("Response", (), {"data": data})()


class OpenAIEmbeddingProviderTest(unittest.TestCase):
    def test_batches_and_caches_duplicate_texts(self) -> None:
        embeddings = FakeEmbeddings()
        client = type("Client", (), {"embeddings": embeddings})()
        provider = OpenAIEmbeddingProvider(
            client=client, model="test", batch_size=2
        )
        first = provider.embed_documents(["one", "two", "three", "one"])
        second = provider.embed_documents(["one", "two"])
        self.assertEqual(len(embeddings.calls), 2)
        self.assertEqual([len(call) for call in embeddings.calls], [2, 1])
        self.assertEqual(first[0], first[3])
        self.assertEqual(second, first[:2])

    def test_malformed_embedding_count_is_explicit_error(self) -> None:
        embeddings = type(
            "Embeddings",
            (),
            {"create": lambda self, **kwargs: type("Response", (), {"data": []})()},
        )()
        client = type("Client", (), {"embeddings": embeddings})()
        with self.assertRaisesRegex(EmbeddingProviderError, "number"):
            OpenAIEmbeddingProvider(client=client).embed_documents(["one"])


if __name__ == "__main__":
    unittest.main()
