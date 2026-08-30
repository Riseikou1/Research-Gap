import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.extraction.paper_extractor import (
    PaperExtractor,
    _Claim,
    _ExtractionResult,
    _Limitation,
)
from src.models.paper import Paper


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.inputs = []

    def parse(self, **kwargs):
        self.inputs.append(kwargs["input"])
        return SimpleNamespace(output_parsed=self.payload)


class EvidenceTest(unittest.TestCase):
    def test_model_tracks_missing_fields_and_validates_confidence(self):
        evidence = PaperEvidence(paper_id="p1", title="A paper", extraction_confidence=0.5)
        self.assertIsNone(evidence.sample_size)
        self.assertIn("sample_size", evidence.missing_fields)
        with self.assertRaises(ValidationError):
            EvidenceItem(value="claim", source="title", confidence=1.1)

    def test_extracts_supported_claims_and_filters_generic_criticism(self):
        payload = _ExtractionResult(
            method_or_intervention=[_Claim(value="LoRA", evidence_text="We use LoRA", source="abstract", confidence=.9)],
            limitations=[
                _Limitation(value="small cohort", evidence_text="A limitation is the small cohort", source="abstract", confidence=.8, author_stated=True),
                _Limitation(value="unsupported criticism", evidence_text="text", source="abstract", confidence=.2, author_stated=False),
            ],
            extraction_confidence=.8,
        )
        responses = FakeResponses(payload)
        result = PaperExtractor(client=SimpleNamespace(responses=responses)).extract(
            Paper(id="p1", title="A paper", abstract="We use LoRA.")
        )
        self.assertEqual(result.method_or_intervention[0].value, "LoRA")
        self.assertEqual(len(result.limitations), 1)
        self.assertIn("Title:", responses.inputs[0])

    def test_no_abstract_and_batch_limit_are_safe(self):
        payload = _ExtractionResult(extraction_confidence=.5)
        responses = FakeResponses(payload)
        extractor = PaperExtractor(client=SimpleNamespace(responses=responses), evidence_limit=1)
        papers = [Paper(id="a", title="A"), Paper(id="b", title="B")]
        results = extractor.extract_many(papers)
        self.assertEqual([item.paper_id for item in results], ["a"])
        self.assertEqual(len(responses.inputs), 1)


if __name__ == "__main__":
    unittest.main()
