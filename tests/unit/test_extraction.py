import unittest
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from unittest.mock import patch
from types import SimpleNamespace

from pydantic import ValidationError

from src.extraction.evidence import EvidenceItem, PaperEvidence, canonical_evidence_key
from src.extraction.paper_extractor import (
    EVIDENCE_SCHEMA_VERSION,
    PaperExtractor,
    _ExtractionResult,
    _LimitationClaim,
    _MethodClaim,
)
import src.extraction.paper_extractor as paper_extractor_module
from src.models.paper import Paper


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.inputs = []

    def parse(self, **kwargs):
        self.inputs.append(kwargs["input"])
        return SimpleNamespace(output_parsed=self.payload)


class ConcurrentResponses:
    def __init__(self, payload, *, barrier=None, failures=None):
        self.payload = payload
        self.barrier = barrier
        self.failures = set(failures or ())
        self.inputs = []
        self.max_active = 0
        self._active = 0
        self._lock = Lock()

    def parse(self, **kwargs):
        source = kwargs["input"]
        title = source.splitlines()[1]
        with self._lock:
            self.inputs.append(source)
            self._active += 1
            self.max_active = max(self.max_active, self._active)

        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            if title in self.failures:
                raise RuntimeError("boom")
            return SimpleNamespace(output_parsed=self.payload)
        finally:
            with self._lock:
                self._active -= 1


class EvidenceTest(unittest.TestCase):
    def test_model_tracks_missing_fields_and_validates_confidence(self):
        evidence = PaperEvidence(paper_id="p1", title="A paper", study_type="other", extraction_confidence=0.5)
        self.assertIsNone(evidence.sample_size)
        self.assertIn("sample_size", evidence.missing_fields)
        with self.assertRaises(ValidationError):
            EvidenceItem(value="claim", source="title", confidence=1.1)

    def test_extracts_supported_claims_and_filters_generic_criticism(self):
        payload = _ExtractionResult(
            method_or_intervention=[_MethodClaim(value="LoRA", evidence_text="We use LoRA", source="abstract", confidence=.9, role="primary")],
            limitations=[
                _LimitationClaim(value="small cohort", evidence_text="A limitation is the small cohort", source="abstract", confidence=.8, author_stated=True),
                _LimitationClaim(value="unsupported criticism", evidence_text="text", source="abstract", confidence=.2, author_stated=False),
            ],
            extraction_confidence=.8,
        )
        responses = FakeResponses(payload)
        result = PaperExtractor(client=SimpleNamespace(responses=responses)).extract(
            Paper(id="p1", title="A paper", abstract="We use LoRA. A limitation is the small cohort.")
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

    def test_extract_many_is_bounded_concurrent_and_preserves_input_order(self):
        responses = ConcurrentResponses(
            _ExtractionResult(extraction_confidence=.5),
            barrier=Barrier(3),
        )
        extractor = PaperExtractor(
            client=SimpleNamespace(responses=responses),
            max_workers=3,
        )
        papers = [
            Paper(id="a", title="A"),
            Paper(id="b", title="B"),
            Paper(id="c", title="C"),
        ]

        results = extractor.extract_many(papers)

        self.assertEqual([item.paper_id for item in results], ["a", "b", "c"])
        self.assertEqual(len(responses.inputs), 3)
        self.assertGreaterEqual(responses.max_active, 2)

    def test_duplicate_papers_share_one_in_flight_extraction(self):
        responses = FakeResponses(_ExtractionResult(extraction_confidence=.5))
        extractor = PaperExtractor(
            client=SimpleNamespace(responses=responses),
            max_workers=2,
        )
        duplicate = Paper(id="a", title="A")

        results = extractor.extract_many([duplicate, duplicate])

        self.assertEqual([item.paper_id for item in results], ["a", "a"])
        self.assertEqual(len(responses.inputs), 1)

    def test_cached_evidence_is_reused_but_changed_abstract_is_not(self):
        responses = FakeResponses(_ExtractionResult(extraction_confidence=.5))
        extractor = PaperExtractor(client=SimpleNamespace(responses=responses))
        paper = Paper(id="a", title="A", abstract="First abstract")

        first = extractor.extract_many([paper])[0]
        second = extractor.extract_many([paper])[0]
        paper.abstract = "Changed abstract"
        third = extractor.extract(paper)

        self.assertIs(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(len(responses.inputs), 2)

    def test_persistent_evidence_cache_reuses_across_instances_and_versions(self):
        paper = Paper(id="a", title="A", abstract="Stable abstract")

        with TemporaryDirectory() as directory:
            first_responses = FakeResponses(_ExtractionResult(extraction_confidence=.5))
            first = PaperExtractor(
                client=SimpleNamespace(responses=first_responses),
                cache_path=f"{directory}/evidence.sqlite3",
                model="model-a",
            )
            first.extract(paper)

            second_responses = FakeResponses(_ExtractionResult(extraction_confidence=.5))
            second = PaperExtractor(
                client=SimpleNamespace(responses=second_responses),
                cache_path=f"{directory}/evidence.sqlite3",
                model="model-a",
            )
            second.extract(paper)
            self.assertEqual(second_responses.inputs, [])
            self.assertEqual(second.metrics_snapshot()["persistent_cache_hits"], 1)

            changed = paper.model_copy(update={"abstract": "Changed abstract"})
            second.extract(changed)
            self.assertEqual(len(second_responses.inputs), 1)

            other_model = PaperExtractor(
                client=SimpleNamespace(responses=FakeResponses(_ExtractionResult(extraction_confidence=.5))),
                cache_path=f"{directory}/evidence.sqlite3",
                model="model-b",
            )
            other_model.extract(paper)
            self.assertEqual(other_model.metrics_snapshot()["persistent_cache_hits"], 0)

            with patch.object(
                paper_extractor_module,
                "EVIDENCE_SCHEMA_VERSION",
                EVIDENCE_SCHEMA_VERSION + 1,
            ):
                versioned_responses = FakeResponses(_ExtractionResult(extraction_confidence=.5))
                versioned = PaperExtractor(
                    client=SimpleNamespace(responses=versioned_responses),
                    cache_path=f"{directory}/evidence.sqlite3",
                    model="model-a",
                )
                versioned.extract(paper)
                self.assertEqual(len(versioned_responses.inputs), 1)

    def test_parallel_failures_are_recorded_without_dropping_successes(self):
        responses = ConcurrentResponses(
            _ExtractionResult(extraction_confidence=.5),
            failures={"B"},
        )
        extractor = PaperExtractor(
            client=SimpleNamespace(responses=responses),
            max_workers=3,
        )

        results = extractor.extract_many(
            [
                Paper(id="a", title="A"),
                Paper(id="b", title="B"),
                Paper(id="c", title="C"),
            ]
        )

        self.assertEqual([item.paper_id for item in results], ["a", "c"])
        self.assertEqual(len(extractor.failures), 1)
        self.assertIn("b: Evidence extraction failed: boom", str(extractor.failures[0]))

    def test_method_roles_separate_primary_and_comparison_models(self):
        claims = [
            _MethodClaim(value="DeiT", evidence_text="DeiT", source="abstract", confidence=.9, role="primary"),
            _MethodClaim(value="VGG19", evidence_text="VGG19", source="abstract", confidence=.9, role="comparison"),
            _MethodClaim(value="data augmentation", evidence_text="data augmentation", source="abstract", confidence=.9, role="supporting"),
        ]
        payload = _ExtractionResult(method_or_intervention=claims, study_type="empirical", extraction_confidence=.9)
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        result = extractor.extract(Paper(id="p1", title="Models", abstract="DeiT, VGG19, and data augmentation."))
        self.assertEqual([item.value for item in result.method_or_intervention], ["DeiT"])
        self.assertEqual([item.value for item in result.comparison_or_baseline], ["VGG19"])

    def test_generic_future_work_is_rejected_but_concrete_direction_survives(self):
        payload = _ExtractionResult(
            future_work=[
                EvidenceItem(value="future work is discussed", evidence_text="future work is discussed", source="abstract", confidence=.9),
                EvidenceItem(value="multilingual datasets", evidence_text="Future work will evaluate the model on multilingual datasets", source="abstract", confidence=.9),
            ],
            study_type="empirical",
            extraction_confidence=.9,
        )
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        result = extractor.extract(Paper(id="p1", title="Model", abstract="Future work is discussed. Future work will evaluate the model on multilingual datasets."))
        self.assertEqual([item.value for item in result.future_work], ["multilingual datasets"])

    def test_study_type_is_preserved(self):
        payload = _ExtractionResult(study_type="survey", extraction_confidence=.9)
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        result = extractor.extract(Paper(id="p1", title="A survey", abstract="We survey prior work."))
        self.assertEqual(result.study_type, "survey")

    def test_explicit_constraints_are_preserved_separately_from_limitations(self):
        payload = _ExtractionResult(
            constraints=[EvidenceItem(
                value="limited labeled data",
                evidence_text="The method requires limited labeled data",
                source="abstract",
                confidence=.9,
            )],
            limitations=[_LimitationClaim(
                value="poor field generalization",
                evidence_text="A limitation is poor field generalization",
                source="abstract",
                confidence=.9,
                author_stated=True,
            )],
            extraction_confidence=.9,
        )
        result = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload))).extract(
            Paper(id="p1", title="A paper", abstract="The method requires limited labeled data. A limitation is poor field generalization.")
        )
        self.assertEqual([item.value for item in result.constraints], ["limited labeled data"])
        self.assertEqual([item.value for item in result.limitations], ["poor field generalization"])

    def test_data_efficient_deit_name_is_method_not_constraint(self):
        payload = _ExtractionResult(
            method_or_intervention=[_MethodClaim(
                value="Data-efficient Image Transformers (DeiT)",
                evidence_text="Data-efficient Image Transformers (DeiT)",
                source="abstract",
                confidence=.9,
                role="primary",
            )],
            constraints=[EvidenceItem(
                value="Data-efficient Image Transformers (DeiT)",
                evidence_text="Data-efficient Image Transformers (DeiT)",
                source="abstract",
                confidence=.9,
            )],
            extraction_confidence=.9,
        )
        result = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload))).extract(
            Paper(id="p1", title="DeiT", abstract="Data-efficient Image Transformers (DeiT).")
        )
        self.assertEqual([item.value for item in result.method_or_intervention], ["Data-efficient Image Transformers (DeiT)"])
        self.assertEqual(result.constraints, [])

    def test_explicit_limited_label_experiment_keeps_method_and_constraint(self):
        payload = _ExtractionResult(
            method_or_intervention=[_MethodClaim(
                value="DeiT",
                evidence_text="DeiT achieved strong performance",
                source="abstract",
                confidence=.9,
                role="primary",
            )],
            constraints=[EvidenceItem(
                value="small labeled training dataset",
                evidence_text="Using a small labeled training dataset, DeiT achieved strong performance",
                source="abstract",
                confidence=.9,
            )],
            extraction_confidence=.9,
        )
        result = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload))).extract(
            Paper(
                id="p1",
                title="DeiT",
                abstract="Using a small labeled training dataset, DeiT achieved strong performance.",
            )
        )
        self.assertEqual([item.value for item in result.method_or_intervention], ["DeiT"])
        self.assertEqual([item.value for item in result.constraints], ["small labeled training dataset"])

    def test_background_methods_are_not_comparisons(self):
        payload = _ExtractionResult(
            comparison_or_baseline=[
                EvidenceItem(
                    value="visual inspection",
                    evidence_text="Traditional visual inspection is time-consuming.",
                    source="abstract",
                    confidence=.9,
                ),
                EvidenceItem(
                    value="CNN models",
                    evidence_text="GreenViT outperforms state-of-the-art CNN models.",
                    source="abstract",
                    confidence=.9,
                ),
            ],
            extraction_confidence=.9,
        )
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        result = extractor.extract(
            Paper(
                id="p1",
                title="GreenViT",
                abstract="Traditional visual inspection is time-consuming. GreenViT outperforms state-of-the-art CNN models.",
            )
        )
        self.assertEqual([item.value for item in result.comparison_or_baseline], ["CNN models"])

    def test_conservative_canonicalization_deduplicates_harmless_spelling(self):
        self.assertEqual(
            canonical_evidence_key("Method-Alpha"),
            canonical_evidence_key("method_alpha"),
        )

        payload = _ExtractionResult(
            comparison_or_baseline=[
                EvidenceItem(
                    value="Method-Alpha",
                    evidence_text="Method-Alpha",
                    source="abstract",
                    confidence=.9,
                ),
                EvidenceItem(
                    value="method_alpha",
                    evidence_text="method_alpha",
                    source="abstract",
                    confidence=.9,
                ),
            ],
            extraction_confidence=.9,
        )

        extractor = PaperExtractor(
            client=SimpleNamespace(
                responses=FakeResponses(payload)
            )
        )

        result = extractor.extract(
            Paper(
                id="p1",
                title="Methods",
                abstract="Method-Alpha and method_alpha.",
            )
        )

        self.assertEqual(
            len(result.comparison_or_baseline),
            1,
        )
        
    def test_equal_head_to_head_methods_remain_primary(self):
        values = ["Next-ViT", "ConvNeXt-ViT", "Swin Transformer"]
        payload = _ExtractionResult(
            method_or_intervention=[
                _MethodClaim(value=value, evidence_text=value, source="abstract", confidence=.9, role="primary")
                for value in values
            ],
            study_type="empirical",
            extraction_confidence=.9,
        )
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        result = extractor.extract(Paper(id="p1", title="Comparison", abstract=", ".join(values)))
        self.assertEqual([item.value for item in result.method_or_intervention], values)
        self.assertEqual(result.comparison_or_baseline, [])

    def test_named_multi_component_proposed_method_remains_primary(self):
        payload = _ExtractionResult(
            method_or_intervention=[
                _MethodClaim(
                    value="PMF+FA",
                    evidence_text="we call the overall method PMF+FA",
                    source="abstract",
                    confidence=.9,
                    role="primary",
                ),
                _MethodClaim(
                    value="feature attention module",
                    evidence_text="feature attention module",
                    source="abstract",
                    confidence=.9,
                    role="supporting",
                ),
            ],
            study_type="empirical",
            extraction_confidence=.9,
        )
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        result = extractor.extract(
            Paper(
                id="p1",
                title="PMF+FA",
                abstract="We call the overall method PMF+FA with a feature attention module.",
            )
        )
        self.assertEqual([item.value for item in result.method_or_intervention], ["PMF+FA"])

    def test_unsupported_evidence_is_rejected_with_debug_logging(self):
        payload = _ExtractionResult(
            research_objective=EvidenceItem(
                value="Unsupported objective",
                evidence_text="not present in source",
                source="abstract",
                confidence=.9,
            ),
            extraction_confidence=.5,
        )
        extractor = PaperExtractor(client=SimpleNamespace(responses=FakeResponses(payload)))
        with self.assertLogs("src.extraction.paper_extractor", level="DEBUG") as logs:
            result = extractor.extract(Paper(id="p1", title="A paper", abstract="No objective."))
        self.assertIsNone(result.research_objective)
        self.assertTrue(any("p1" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
