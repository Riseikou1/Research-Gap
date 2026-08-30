import unittest
from typing import Any

from src.models.idea import ResearchIdea
from src.query.openai_generator import (
    OpenAIQueryGenerationError,
    OpenAIQueryGenerator,
)


class FakeResponses:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.payload = payload
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        parsed = (
            kwargs["text_format"].model_validate(self.payload)
            if self.payload is not None
            else None
        )
        return type(
            "Response",
            (),
            {
                "output_parsed": parsed,
                "output": [],
                "status": "completed",
                "incomplete_details": None,
            },
        )()


class FakeClient:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.responses = FakeResponses(payload)


class OpenAIQueryGeneratorTest(unittest.TestCase):
    def test_validates_and_bounds_structured_queries(self) -> None:
        client = FakeClient(
            {
                "queries": [
                    {
                        "text": "retrieval augmented generation low rank adaptation",
                        "strategy": "terminology_expansion",
                    },
                    {
                        "text": "parameter efficient retrieval systems",
                        "strategy": "conceptual_reformulation",
                    },
                ]
            }
        )
        result = OpenAIQueryGenerator(client=client, model="test-model").generate(
            ResearchIdea(original_text="RAG using LoRA")
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(all(item.source == "llm" for item in result))
        self.assertTrue(all(item.provider == "openai" for item in result))
        self.assertEqual(client.responses.kwargs["reasoning"], {"effort": "low"})
        self.assertFalse(client.responses.kwargs["store"])

    def test_extra_fields_are_rejected(self) -> None:
        client = FakeClient(
            {
                "queries": [
                    {
                        "text": "retrieval augmented generation",
                        "strategy": "terminology_expansion",
                        "novelty": True,
                    }
                ]
            }
        )
        with self.assertRaisesRegex(OpenAIQueryGenerationError, "schema validation"):
            OpenAIQueryGenerator(client=client).generate(
                ResearchIdea(original_text="RAG using LoRA")
            )

    def test_more_than_three_queries_are_rejected(self) -> None:
        entries = [
            {
                "text": f"query terms {index}",
                "strategy": strategy,
            }
            for index, strategy in enumerate(
                [
                    "terminology_expansion",
                    "conceptual_reformulation",
                    "method_focused_reformulation",
                    "terminology_expansion",
                ]
            )
        ]
        with self.assertRaises(OpenAIQueryGenerationError):
            OpenAIQueryGenerator(client=FakeClient({"queries": entries})).generate(
                ResearchIdea(original_text="RAG using LoRA")
            )


if __name__ == "__main__":
    unittest.main()
