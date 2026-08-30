import unittest
from typing import Any

from pydantic import ValidationError

from src.query.openai_decomposer import (
    OpenAIDecomposer,
    OpenAIDecompositionError,
)


class FakeResponses:
    """Minimal fake of client.responses for Structured Outputs tests."""

    def __init__(
        self,
        payload: dict[str, Any] | None,
        *,
        status: str = "completed",
        output: list[Any] | None = None,
    ) -> None:
        self.payload = payload
        self.status = status
        self.output = output or []
        self.kwargs: dict[str, Any] | None = None

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs

        text_format = kwargs["text_format"]

        # Simulate what the OpenAI SDK does:
        # validate the structured response against the supplied Pydantic model.
        parsed = (
            text_format.model_validate(self.payload)
            if self.payload is not None
            else None
        )

        return type(
            "Response",
            (),
            {
                "output_parsed": parsed,
                "output": self.output,
                "status": self.status,
                "incomplete_details": None,
            },
        )()


class FakeClient:
    def __init__(
        self,
        payload: dict[str, Any] | None,
        *,
        status: str = "completed",
        output: list[Any] | None = None,
    ) -> None:
        self.responses = FakeResponses(
            payload,
            status=status,
            output=output,
        )


def valid_payload() -> dict[str, Any]:
    return {
        "original_text": "RAG using LoRA",
        "problem": ["RAG"],
        "population": [],
        "intervention_or_method": ["LoRA"],
        "comparison": [],
        "outcomes": [],
        "domain": [],
        "constraints": [],
        "keywords": ["RAG", "LoRA"],
        "synonyms": [
            {
                "canonical": "RAG",
                "alternatives": [
                    "retrieval augmented generation",
                ],
            }
        ],
    }


class OpenAIDecomposerTest(unittest.TestCase):

    def test_mocked_structured_output_is_validated(self) -> None:
        client = FakeClient(valid_payload())

        result = OpenAIDecomposer(
            client=client,
            model="test-model",
        ).decompose(
            "RAG using LoRA"
        )

        self.assertEqual(
            result.problem,
            ["RAG"],
        )

        self.assertEqual(
            result.intervention_or_method,
            ["LoRA"],
        )

        self.assertEqual(
            result.synonyms,
            {
                "RAG": [
                    "retrieval augmented generation"
                ]
            },
        )

        request = client.responses.kwargs

        self.assertIsNotNone(request)

        assert request is not None

        self.assertEqual(
            request["model"],
            "test-model",
        )

        self.assertEqual(
            request["reasoning"],
            {"effort": "low"},
        )

        self.assertFalse(
            request["store"],
        )

        self.assertEqual(
            request["input"],
            "RAG using LoRA",
        )

        self.assertIn(
            "instructions",
            request,
        )

        self.assertIn(
            "text_format",
            request,
        )

        self.assertEqual(
            request["max_output_tokens"],
            1600,
        )

    def test_invalid_or_extra_fields_are_rejected(self) -> None:
        payload = valid_payload()

        payload["novelty"] = "novel"

        client = FakeClient(payload)

        with self.assertRaisesRegex(
            OpenAIDecompositionError,
            "schema validation",
        ):
            OpenAIDecomposer(
                client=client,
                model="test-model",
            ).decompose(
                "RAG using LoRA"
            )

    def test_changed_original_text_is_rejected(self) -> None:
        payload = valid_payload()

        payload["original_text"] = "rag using lora"

        client = FakeClient(payload)

        with self.assertRaisesRegex(
            OpenAIDecompositionError,
            "changed original_text",
        ):
            OpenAIDecomposer(
                client=client,
                model="test-model",
            ).decompose(
                "RAG using LoRA"
            )

    def test_duplicate_keywords_are_normalized(self) -> None:
        payload = valid_payload()

        payload["keywords"] = [
            "RAG",
            "rag",
            " RAG ",
            "LoRA",
            "lora",
        ]

        client = FakeClient(payload)

        result = OpenAIDecomposer(
            client=client,
            model="test-model",
        ).decompose(
            "RAG using LoRA"
        )

        self.assertEqual(
            result.keywords,
            [
                "RAG",
                "LoRA",
            ],
        )

    def test_duplicate_synonym_groups_are_merged(self) -> None:
        payload = valid_payload()

        payload["synonyms"] = [
            {
                "canonical": "RAG",
                "alternatives": [
                    "retrieval augmented generation",
                ],
            },
            {
                "canonical": "rag",
                "alternatives": [
                    "Retrieval Augmented Generation",
                    "retrieval-augmented generation",
                ],
            },
        ]

        client = FakeClient(payload)

        result = OpenAIDecomposer(
            client=client,
            model="test-model",
        ).decompose(
            "RAG using LoRA"
        )

        self.assertEqual(
            result.synonyms,
            {
                "RAG": [
                    "retrieval augmented generation",
                    "retrieval-augmented generation",
                ]
            },
        )

    def test_missing_output_is_rejected(self) -> None:
        client = FakeClient(None)

        with self.assertRaisesRegex(
            OpenAIDecompositionError,
            "no parsed ResearchIdea payload",
        ):
            OpenAIDecomposer(
                client=client,
                model="test-model",
            ).decompose(
                "RAG using LoRA"
            )


if __name__ == "__main__":
    unittest.main()