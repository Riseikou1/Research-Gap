import unittest
from datetime import datetime, timezone

from src.analysis.verification import GapVerifier, _idea_match_strength
from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.models.idea import ResearchIdea
from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode
from src.retrieval.multi_query import MultiQueryRetriever
from src.query.deterministic import DeterministicDecomposer


def claim(
    value: str,
    *,
    canonical_value: str | None = None,
    evidence_text: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        canonical_value=canonical_value,
        evidence_text=evidence_text or value,
        source="abstract",
        confidence=0.9,
    )


def record(
    paper_id: str,
    *,
    problem: str | None = None,
    method: str | None = None,
    population: str | None = None,
    modality: str | None = None,
    constraint: str | None = None,
) -> PaperEvidence:
    return PaperEvidence(
        paper_id=paper_id,
        title=f"Study {paper_id}",
        study_type="empirical",
        research_objective=claim(problem) if problem else None,
        population_or_setting=[claim(population)] if population else [],
        method_or_intervention=[claim(method)] if method else [],
        data_or_modality=[claim(modality)] if modality else [],
        constraints=[claim(constraint)] if constraint else [],
        extraction_confidence=0.9,
    )


class OnePaperRetriever:
    provider_name = "test"

    def __init__(self, paper: Paper) -> None:
        self.paper = paper

    def search(self, request):
        result = self.paper.model_copy(deep=True)
        result.provenance = [
            RetrievalProvenance(
                query=request.query,
                provider=self.provider_name,
                mode=request.mode,
                retrieved_at=datetime.now(timezone.utc),
                provider_rank=1,
            )
        ]
        return [result]


class EvidenceByIdExtractor:
    def __init__(self, evidence: list[PaperEvidence]) -> None:
        self.by_id = {item.paper_id: item for item in evidence}
        self.failures: list[str] = []

    def extract_many(self, papers, limit=None):
        result = [self.by_id[paper.id] for paper in papers if paper.id in self.by_id]
        return result[:limit] if limit is not None else result


class Milestone6CorrectnessTest(unittest.TestCase):
    def test_deterministic_decomposition_keeps_input_modality_out_of_method(self):
        idea = DeterministicDecomposer().decompose(
            "Detecting structural damage in bridges using few-shot learning "
            "with vibration sensor data under different climates"
        )

        self.assertEqual(idea.intervention_or_method, ["few-shot learning"])
        self.assertEqual(idea.data_or_modality, ["vibration sensor data"])
        self.assertNotIn("vibration sensor data", idea.intervention_or_method)
        self.assertNotIn("vibration sensor data", idea.domain)
        self.assertEqual(
            idea.model_dump()["data_or_modality"],
            ["vibration sensor data"],
        )

    def test_direct_matching_reports_partial_structured_facet_coverage(self):
        idea = ResearchIdea(
            original_text="bridge damage detection",
            problem=["structural damage detection"],
            population=["bridges"],
            intervention_or_method=["few-shot learning"],
            data_or_modality=["vibration sensor data"],
            constraints=["limited labeled failure examples"],
            synonyms={
                "structural damage detection": ["bridge damage detection"],
                "few-shot learning": ["few-shot fine-tuning"],
                "limited labeled failure examples": ["few labeled examples"],
            },
        )
        paper_a = record(
            "a",
            problem="bridge damage detection",
            method="few-shot learning",
            population="bridges",
            modality="image data",
            constraint="few labeled examples",
        )

        complete, matched = _idea_match_strength(idea, paper_a)

        self.assertFalse(complete)
        self.assertEqual(
            set(matched),
            {"problem", "method", "population_or_domain", "constraint"},
        )
        self.assertNotIn("data_or_modality", matched)

        paper_b = record(
            "b",
            problem="bridge damage detection",
            method="few-shot fine-tuning",
            population="bridges",
            modality="vibration sensor data",
            constraint="few labeled examples",
        )
        complete_b, matched_b = _idea_match_strength(idea, paper_b)

        self.assertTrue(complete_b)
        self.assertIn("data_or_modality", matched_b)

    def test_domain_is_search_context_not_a_required_direct_facet(self):
        idea = ResearchIdea(
            original_text="problem method in a broad domain",
            problem=["problem"],
            intervention_or_method=["method"],
            domain=["broad domain"],
        )
        paper = record("domain-context", problem="problem", method="method")

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertNotIn("broad domain", matched)

    def test_canonical_identity_matches_distinct_surface_forms(self):
        idea = ResearchIdea(
            original_text="detect damage with adaptation on vibration measurements",
            problem=["damage detection"],
            intervention_or_method=["few-shot adaptation"],
            data_or_modality=["vibration measurements"],
            canonical_facets={
                "problem": {"damage detection": "structural damage identification"},
                "intervention_or_method": {"few-shot adaptation": "few-shot learning"},
                "data_or_modality": {"vibration measurements": "vibration sensor data"},
            },
        )
        paper = PaperEvidence(
            paper_id="canonical",
            title="Canonical study",
            study_type="empirical",
            research_objective=claim(
                "structural anomaly identification",
                canonical_value="structural damage identification",
            ),
            method_or_intervention=[claim(
                "parameter-efficient adaptation",
                canonical_value="few-shot learning",
            )],
            data_or_modality=[claim(
                "accelerometer signals",
                canonical_value="vibration sensor data",
            )],
            extraction_confidence=0.9,
        )

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertEqual(
            set(matched),
            {"problem", "method", "data_or_modality"},
        )

    def test_generic_canonical_method_equivalence(self):
        idea = ResearchIdea(
            original_text="method surface a",
            intervention_or_method=["method_surface_a"],
            canonical_facets={
                "intervention_or_method": {
                    "method_surface_a": "concept_m",
                },
            },
        )
        paper = record("surface-b", method="method_surface_b")
        paper.method_or_intervention[0].canonical_value = "concept_m"

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertEqual(matched, ["method"])

    def test_conflicting_canonical_method_is_not_lexically_confirmed(self):
        idea = ResearchIdea(
            original_text="surface alpha",
            intervention_or_method=["surface alpha"],
            canonical_facets={
                "intervention_or_method": {"surface alpha": "concept_m1"},
            },
        )
        paper = record("surface-b", method="surface beta")
        paper.method_or_intervention[0].canonical_value = "concept_m2"

        complete, matched = _idea_match_strength(idea, paper)

        self.assertFalse(complete)
        self.assertNotIn("method", matched)

    def test_canonical_containment_matches(self):
        idea = ResearchIdea(
            original_text="method surface",
            intervention_or_method=["method surface"],
            canonical_facets={
                "intervention_or_method": {
                    "method surface": "target concept alpha",
                },
            },
        )
        paper = record("canonical-containment", method="different surface")
        paper.method_or_intervention[0].canonical_value = "concept alpha"

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertEqual(matched, ["method"])

    def test_canonical_mismatch_does_not_block_explicit_evidence(self):
        idea = ResearchIdea(
            original_text="method alpha",
            intervention_or_method=["method alpha"],
            canonical_facets={
                "intervention_or_method": {"method alpha": "canonical alpha"},
            },
        )
        paper = record("explicit-evidence", method="method beta")
        paper.method_or_intervention[0].canonical_value = "different canonical wording"
        paper.method_or_intervention[0].evidence_text = (
            "The study explicitly evaluates method alpha."
        )

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertEqual(matched, ["method"])

    def test_one_constraint_evidence_item_can_satisfy_two_groups(self):
        idea = ResearchIdea(
            original_text="two constraints",
            constraints=["constraint alpha", "constraint beta"],
        )
        paper = record(
            "two-constraints",
            constraint="constraint alpha and constraint beta",
        )

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertEqual(matched, ["constraint"])

    def test_missing_one_constraint_group_keeps_facet_incomplete(self):
        idea = ResearchIdea(
            original_text="two constraints",
            constraints=["constraint alpha", "constraint beta"],
        )
        paper = record("one-constraint", constraint="constraint alpha")

        complete, matched = _idea_match_strength(idea, paper)

        self.assertFalse(complete)
        self.assertNotIn("constraint", matched)

    def test_constraint_canonical_equivalence(self):
        idea = ResearchIdea(
            original_text="constraint surface a",
            constraints=["constraint_surface_a"],
            canonical_facets={
                "constraints": {
                    "constraint_surface_a": "concept_c",
                },
            },
        )
        paper = record("constraint-b", constraint="constraint_surface_b")
        paper.constraints[0].canonical_value = "concept_c"

        complete, matched = _idea_match_strength(idea, paper)

        self.assertTrue(complete)
        self.assertEqual(matched, ["constraint"])

    def test_required_canonical_facets_must_be_in_one_paper(self):
        idea = ResearchIdea(
            original_text="method and modality",
            intervention_or_method=["method_surface"],
            data_or_modality=["modality_surface"],
            canonical_facets={
                "intervention_or_method": {"method_surface": "concept_m"},
                "data_or_modality": {"modality_surface": "concept_d"},
            },
        )
        method_paper = record("method-only", method="different method")
        method_paper.method_or_intervention[0].canonical_value = "concept_m"
        modality_paper = record("modality-only", modality="different modality")
        modality_paper.data_or_modality[0].canonical_value = "concept_d"

        self.assertFalse(_idea_match_strength(idea, method_paper)[0])
        self.assertFalse(_idea_match_strength(idea, modality_paper)[0])

    def test_distinct_canonical_requirements_remain_and_requirements(self):
        idea = ResearchIdea(
            original_text="two methods",
            intervention_or_method=["method A", "method B"],
            canonical_facets={
                "intervention_or_method": {
                    "method A": "first method",
                    "method B": "second method",
                },
            },
        )
        paper = record("one-method", method="method A")

        complete, matched = _idea_match_strength(idea, paper)

        self.assertFalse(complete)
        self.assertNotIn("method", matched)

    def test_targeted_evidence_is_included_in_coverage(self):
        idea = ResearchIdea(
            original_text="problem method bridges vibration data constraint",
            problem=["problem"],
            population=["bridges"],
            intervention_or_method=["method"],
            data_or_modality=["vibration data"],
            constraints=["constraint"],
        )
        initial = record(
            "initial",
            problem="problem",
            method="method",
            population="bridges",
            constraint="constraint",
        )
        retrieved = record("retrieved", modality="vibration data")
        verifier = GapVerifier(
            MultiQueryRetriever(
                OnePaperRetriever(
                    Paper(
                        id="retrieved",
                        title="Study retrieved",
                        abstract="vibration data",
                    )
                ),
                max_candidates=10,
                per_route_limit=5,
                max_workers=1,
            ),
            EvidenceByIdExtractor([retrieved]),
        )

        assessment = verifier.assess_idea(idea, evidence=[initial])

        self.assertEqual(assessment.label, "uncertain")
        self.assertTrue(
            any("Analyzed 2 evidence-bearing papers" in note for note in assessment.coverage_notes)
        )
        self.assertFalse(
            any("data_or_modality" in note for note in assessment.coverage_notes)
        )


if __name__ == "__main__":
    unittest.main()
