import unittest
from datetime import datetime, timezone

from src.analysis.clustering import LandscapeAnalyzer
from src.analysis.gap_candidates import (
    GapCandidateGenerator,
    consolidate_candidates,
    is_concrete_entity,
    validate_evidence_semantics,
)
from src.analysis.models import (
    GapCandidate,
    GapEvidence,
    LandscapeBasis,
    VerificationQuery,
)
from src.analysis.verification import (
    GapVerifier,
    _constraint_matches,
    _idea_match_strength,
    build_idea_verification_queries,
    build_verification_queries,
)
from src.extraction.evidence import EvidenceItem, LimitationEvidence, PaperEvidence
from src.models.idea import ResearchIdea
from src.models.paper import Paper, RetrievalProvenance
from src.retrieval.base import RetrievalError
from src.retrieval.multi_query import MultiQueryRetriever


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def claim(
    value: str,
    text: str | None = None,
    *,
    source: str = "abstract",
) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        evidence_text=text or value,
        source=source,
        confidence=0.9,
    )


def evidence_record(
    paper_id: str,
    *,
    problem: str = "Task Omega",
    methods: list[str] | None = None,
    population: list[str] | None = None,
    constraints: list[str] | None = None,
    comparisons: list[str] | None = None,
    datasets: list[str] | None = None,
    metrics: list[str] | None = None,
    findings: list[str] | None = None,
    limitations: list[str] | None = None,
    future_work: list[str] | None = None,
    source: str = "abstract",
) -> PaperEvidence:
    return PaperEvidence(
        paper_id=paper_id,
        title=f"Study {paper_id}",
        study_type="empirical",
        research_objective=claim(problem, source=source) if problem else None,
        population_or_setting=[
            claim(value, source=source)
            for value in population or []
        ],
        method_or_intervention=[
            claim(value, source=source)
            for value in methods or []
        ],
        comparison_or_baseline=[
            claim(value, source=source)
            for value in comparisons or []
        ],
        datasets=[
            claim(value, source=source)
            for value in datasets or []
        ],
        evaluation_metrics=[
            claim(value, source=source)
            for value in metrics or []
        ],
        main_findings=[
            claim(value, source=source)
            for value in findings or []
        ],
        constraints=[
            claim(value, source=source)
            for value in constraints or []
        ],
        limitations=[
            LimitationEvidence(
                value=value,
                evidence_text=value,
                source=source,
                confidence=0.9,
                author_stated=True,
            )
            for value in limitations or []
        ],
        future_work=[
            claim(value, source=source)
            for value in future_work or []
        ],
        extraction_confidence=0.9,
    )


def paper(
    paper_id: str,
    *,
    title: str | None = None,
    abstract: str | None = "Structured abstract evidence.",
) -> Paper:
    return Paper(
        id=paper_id,
        title=title or f"Study {paper_id}",
        abstract=abstract,
        publication_year=2025,
    )


class FakeVerificationRetriever:
    provider_name = "fake"

    def __init__(
        self,
        papers=None,
        *,
        fail: bool = False,
    ):
        self.papers = list(papers or [])
        self.fail = fail
        self.queries = []

    def search(self, request):
        self.queries.append(request)

        if self.fail:
            raise RetrievalError(
                "fake verification failure"
            )

        result = []

        for rank, item in enumerate(
            self.papers,
            start=1,
        ):
            copied = item.model_copy(
                deep=True
            )

            copied.provenance = [
                RetrievalProvenance(
                    query=request.query,
                    provider=self.provider_name,
                    mode=request.mode,
                    retrieved_at=datetime(
                        2026,
                        1,
                        1,
                        tzinfo=timezone.utc,
                    ),
                    provider_rank=rank,
                )
            ]

            result.append(
                copied
            )

        return result


class FakeVerificationExtractor:
    def __init__(
        self,
        evidence,
        *,
        failures=None,
    ):
        self.evidence = list(evidence)
        self.failures = list(
            failures or []
        )

    def extract_many(
        self,
        papers,
        limit=None,
    ):
        by_id = {
            item.paper_id: item
            for item in self.evidence
        }

        result = [
            by_id[item.id]
            for item in papers
            if item.id in by_id
        ]

        if limit is not None:
            result = result[:limit]

        return result


def make_verifier(
    papers,
    records,
    *,
    fail: bool = False,
    extraction_failures=None,
) -> GapVerifier:
    backend = FakeVerificationRetriever(
        papers,
        fail=fail,
    )

    retriever = MultiQueryRetriever(
        backend,
        max_candidates=30,
        per_route_limit=10,
        max_workers=2,
    )

    extractor = FakeVerificationExtractor(
        records,
        failures=extraction_failures,
    )

    return GapVerifier(
        retriever,
        extractor,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class Milestone6Test(unittest.TestCase):
    def setUp(self):
        self.idea = ResearchIdea(
            original_text=(
                "Method Alpha for Task Omega "
                "in Population Delta under Resource Constraint"
            ),
            problem=[
                "Task Omega",
            ],
            population=[
                "Population Delta",
            ],
            intervention_or_method=[
                "Method Alpha",
            ],
            constraints=[
                "Resource Constraint",
            ],
        )

    # ------------------------------------------------------------------
    # Generic entity handling
    # ------------------------------------------------------------------

    def test_placeholder_entities_are_rejected(self):
        for value in (
            "other",
            "unknown",
            "unspecified",
            "other method",
            "unknown dataset",
        ):
            self.assertFalse(
                is_concrete_entity(value)
            )

        self.assertTrue(
            is_concrete_entity(
                "Method Alpha"
            )
        )
        self.assertTrue(
            is_concrete_entity(
                "Population Delta"
            )
        )

    # ------------------------------------------------------------------
    # Same-paper direct matching
    # ------------------------------------------------------------------

    def test_complete_same_paper_match_is_direct(self):
        record = evidence_record(
            "A",
            methods=["Method Alpha"],
            population=["Population Delta"],
            constraints=["Resource Constraint"],
        )

        direct, matched = _idea_match_strength(
            self.idea,
            record,
        )

        self.assertTrue(
            direct
        )

        self.assertEqual(
            set(matched),
            {
                "problem",
                "method",
                "population_or_domain",
                "constraint",
            },
        )

    def test_missing_method_is_partial(self):
        record = evidence_record(
            "A",
            methods=["Method Beta"],
            population=["Population Delta"],
            constraints=["Resource Constraint"],
        )

        direct, matched = _idea_match_strength(
            self.idea,
            record,
        )

        self.assertFalse(
            direct
        )
        self.assertNotIn(
            "method",
            matched,
        )

    def test_missing_constraint_is_partial(self):
        record = evidence_record(
            "A",
            methods=["Method Alpha"],
            population=["Population Delta"],
        )

        direct, matched = _idea_match_strength(
            self.idea,
            record,
        )

        self.assertFalse(
            direct
        )
        self.assertNotIn(
            "constraint",
            matched,
        )

    def test_split_facets_across_papers_do_not_form_direct_match(self):
        first = evidence_record(
            "A",
            methods=["Method Alpha"],
            population=["Population Delta"],
        )

        second = evidence_record(
            "B",
            methods=["Method Beta"],
            population=["Population Delta"],
            constraints=["Resource Constraint"],
        )

        verifier = make_verifier(
            [
                paper("A"),
                paper("B"),
            ],
            [
                first,
                second,
            ],
        )

        assessment = verifier.assess_idea(
            self.idea,
            LandscapeAnalyzer().analyze(
                [
                    first,
                    second,
                ]
            ),
            [
                first,
                second,
            ],
        )

        self.assertEqual(
            assessment.label,
            "uncertain",
        )
        self.assertEqual(
            assessment.counterexample_paper_ids,
            [],
        )
        self.assertEqual(
            set(
                assessment.partial_match_paper_ids
            ),
            {
                "A",
                "B",
            },
        )

    # ------------------------------------------------------------------
    # Multiple explicit requirements
    # ------------------------------------------------------------------

    def test_multiple_constraints_require_all_constraints(self):
        idea = ResearchIdea(
            original_text=(
                "Method Alpha for Task Omega "
                "under Constraint One and Constraint Two"
            ),
            problem=["Task Omega"],
            intervention_or_method=["Method Alpha"],
            constraints=[
                "Constraint One",
                "Constraint Two",
            ],
        )

        incomplete = evidence_record(
            "A",
            methods=["Method Alpha"],
            constraints=[
                "Constraint One",
            ],
        )

        complete = evidence_record(
            "B",
            methods=["Method Alpha"],
            constraints=[
                "Constraint One",
                "Constraint Two",
            ],
        )

        direct_a, matched_a = _idea_match_strength(
            idea,
            incomplete,
        )
        direct_b, matched_b = _idea_match_strength(
            idea,
            complete,
        )

        self.assertFalse(
            direct_a
        )
        self.assertNotIn(
            "constraint",
            matched_a,
        )

        self.assertTrue(
            direct_b
        )
        self.assertIn(
            "constraint",
            matched_b,
        )

    def test_multiple_methods_require_all_methods(self):
        idea = ResearchIdea(
            original_text=(
                "Method Alpha with Method Beta "
                "for Task Omega"
            ),
            problem=["Task Omega"],
            intervention_or_method=[
                "Method Alpha",
                "Method Beta",
            ],
        )

        one_method = evidence_record(
            "A",
            methods=[
                "Method Alpha",
            ],
        )

        both_methods = evidence_record(
            "B",
            methods=[
                "Method Alpha",
                "Method Beta",
            ],
        )

        direct_a, _ = _idea_match_strength(
            idea,
            one_method,
        )

        direct_b, _ = _idea_match_strength(
            idea,
            both_methods,
        )

        self.assertFalse(
            direct_a
        )
        self.assertTrue(
            direct_b
        )

    def test_explicit_synonyms_are_alternatives_not_separate_requirements(self):
        idea = ResearchIdea(
            original_text=(
                "Method Alpha for Task Omega"
            ),
            problem=["Task Omega"],
            intervention_or_method=[
                "Method Alpha",
            ],
            synonyms={
                "Method Alpha": [
                    "Alpha Technique",
                ],
            },
        )

        record = evidence_record(
            "A",
            methods=[
                "Alpha Technique",
            ],
        )

        direct, matched = _idea_match_strength(
            idea,
            record,
        )

        self.assertTrue(
            direct
        )
        self.assertIn(
            "method",
            matched,
        )

    # ------------------------------------------------------------------
    # Comparison requirements
    # ------------------------------------------------------------------

    def test_explicit_comparison_is_required_for_direct_idea_match(self):
        idea = ResearchIdea(
            original_text=(
                "Method Alpha versus Method Beta "
                "for Task Omega"
            ),
            problem=["Task Omega"],
            intervention_or_method=[
                "Method Alpha",
            ],
            comparison=[
                "Method Beta",
            ],
        )

        without_comparison = evidence_record(
            "A",
            methods=[
                "Method Alpha",
            ],
            comparisons=[
                "Method Gamma",
            ],
        )

        with_comparison = evidence_record(
            "B",
            methods=[
                "Method Alpha",
            ],
            comparisons=[
                "Method Beta",
            ],
        )

        direct_a, matched_a = _idea_match_strength(
            idea,
            without_comparison,
        )
        direct_b, matched_b = _idea_match_strength(
            idea,
            with_comparison,
        )

        self.assertFalse(
            direct_a
        )
        self.assertNotIn(
            "comparison",
            matched_a,
        )

        self.assertTrue(
            direct_b
        )
        self.assertIn(
            "comparison",
            matched_b,
        )

    # ------------------------------------------------------------------
    # Outcome requirements
    # ------------------------------------------------------------------

    def test_explicit_outcome_must_be_evaluated(self):
        idea = ResearchIdea(
            original_text=(
                "Method Alpha for Task Omega "
                "to reduce Outcome Sigma"
            ),
            problem=["Task Omega"],
            intervention_or_method=[
                "Method Alpha",
            ],
            outcomes=[
                "reduce Outcome Sigma",
            ],
        )

        generic_result = evidence_record(
            "A",
            methods=[
                "Method Alpha",
            ],
            metrics=[
                "accuracy",
            ],
            findings=[
                "Method Alpha achieved high accuracy",
            ],
        )

        explicit_result = evidence_record(
            "B",
            methods=[
                "Method Alpha",
            ],
            findings=[
                "Method Alpha reduced Outcome Sigma",
            ],
        )

        direct_a, matched_a = _idea_match_strength(
            idea,
            generic_result,
        )
        direct_b, matched_b = _idea_match_strength(
            idea,
            explicit_result,
        )

        self.assertFalse(
            direct_a
        )
        self.assertNotIn(
            "outcome",
            matched_a,
        )

        self.assertTrue(
            direct_b
        )
        self.assertIn(
            "outcome",
            matched_b,
        )

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def test_direct_idea_queries_preserve_explicit_facets(self):
        idea = ResearchIdea(
            original_text=(
                "Method Alpha versus Method Beta for Task Omega "
                "in Population Delta under Constraint One "
                "to reduce Outcome Sigma"
            ),
            problem=[
                "Task Omega",
            ],
            population=[
                "Population Delta",
            ],
            intervention_or_method=[
                "Method Alpha",
            ],
            comparison=[
                "Method Beta",
            ],
            constraints=[
                "Constraint One",
            ],
            outcomes=[
                "reduce Outcome Sigma",
            ],
        )

        queries = build_idea_verification_queries(
            idea
        )

        self.assertTrue(
            queries
        )
        self.assertLessEqual(
            len(queries),
            3,
        )

        joined = " ".join(
            item.query.casefold()
            for item in queries
        )

        for required in (
            "method alpha",
            "method beta",
            "task omega",
            "population delta",
            "constraint one",
            "outcome sigma",
        ):
            self.assertIn(
                required,
                joined,
            )

    def test_candidate_queries_preserve_every_defining_facet(self):
        candidate = GapCandidate(
            title=(
                "Method Alpha under Constraint One"
            ),
            description=(
                "The combination was not observed."
            ),
            category="combination",
            rationale=(
                "Both facets occurred separately."
            ),
            pattern_type="combination_gap",
            supporting_paper_ids=[
                "A",
                "B",
            ],
            landscape_basis=[
                LandscapeBasis(
                    dimension="method_family",
                    value="Method Alpha",
                    count=2,
                    total=4,
                    prevalence=0.5,
                    paper_ids=["A", "C"],
                ),
                LandscapeBasis(
                    dimension="constraint",
                    value="Constraint One",
                    count=2,
                    total=4,
                    prevalence=0.5,
                    paper_ids=["B", "D"],
                ),
            ],
        )

        queries = build_verification_queries(
            self.idea,
            candidate,
        )

        self.assertTrue(
            queries
        )

        for query in queries:
            normalized = query.query.casefold()

            self.assertIn(
                "method alpha",
                normalized,
            )
            self.assertIn(
                "constraint one",
                normalized,
            )

    def test_invalid_prebuilt_query_is_replaced(self):
        candidate = GapCandidate(
            title=(
                "Method Alpha under Constraint One"
            ),
            description=(
                "The combination was not observed."
            ),
            category="combination",
            rationale=(
                "Both facets occurred separately."
            ),
            pattern_type="combination_gap",
            supporting_paper_ids=[
                "A",
                "B",
            ],
            landscape_basis=[
                LandscapeBasis(
                    dimension="method_family",
                    value="Method Alpha",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["A"],
                ),
                LandscapeBasis(
                    dimension="constraint",
                    value="Constraint One",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["B"],
                ),
            ],
            verification_queries=[
                VerificationQuery(
                    candidate_id="wrong-id",
                    query="irrelevant search",
                    pattern_type="combination_gap",
                    strategy="verification_counterexample",
                )
            ],
        )

        backend = FakeVerificationRetriever()

        retriever = MultiQueryRetriever(
            backend,
            max_candidates=20,
            per_route_limit=10,
            max_workers=2,
        )

        verifier = GapVerifier(
            retriever,
            FakeVerificationExtractor([]),
        )

        verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertTrue(
            backend.queries
        )

        for request in backend.queries:
            normalized = request.query.text.casefold()

            self.assertIn(
                "method alpha",
                normalized,
            )
            self.assertIn(
                "constraint one",
                normalized,
            )

    # ------------------------------------------------------------------
    # Combination-gap verification
    # ------------------------------------------------------------------

    def _combination_candidate(self):
        return GapCandidate(
            title=(
                "Method Alpha under Constraint One"
            ),
            description=(
                "The combination remains unverified."
            ),
            category="combination",
            rationale=(
                "The two facets occur separately."
            ),
            pattern_type="combination_gap",
            supporting_paper_ids=[
                "A",
                "B",
            ],
            landscape_basis=[
                LandscapeBasis(
                    dimension="method_family",
                    value="Method Alpha",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["A"],
                ),
                LandscapeBasis(
                    dimension="constraint",
                    value="Constraint One",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["B"],
                ),
            ],
        )

    def test_combination_counterexample_requires_all_facets_same_paper(self):
        candidate = self._combination_candidate()

        record = evidence_record(
            "X",
            methods=[
                "Method Beta",
            ],
            constraints=[
                "Constraint One",
            ],
        )

        verifier = make_verifier(
            [
                paper(
                    "X",
                    title=(
                        "Method Beta under Constraint One "
                        "for Task Omega"
                    ),
                )
            ],
            [
                record,
            ],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertNotIn(
            "X",
            result.contradicting_paper_ids,
        )

    def test_combination_true_counterexample_is_confirmed(self):
        candidate = self._combination_candidate()

        record = evidence_record(
            "X",
            methods=[
                "Method Alpha",
            ],
            population=[
                "Population Delta",
            ],
            constraints=[
                "Constraint One",
            ],
        )

        verifier = make_verifier(
            [
                paper(
                    "X",
                    title=(
                        "Method Alpha under Constraint One "
                        "for Task Omega"
                    ),
                )
            ],
            [
                record,
            ],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertEqual(
            result.final_label,
            "well_studied",
        )
        self.assertIn(
            "X",
            result.contradicting_paper_ids,
        )

    # ------------------------------------------------------------------
    # Missing comparison verification
    # ------------------------------------------------------------------

    def _comparison_candidate(self):
        return GapCandidate(
            title=(
                "Matched comparison of Method Alpha "
                "and Method Beta"
            ),
            description=(
                "An explicit comparison was not observed."
            ),
            category="comparison",
            rationale=(
                "Both methods are repeatedly represented."
            ),
            pattern_type="missing_comparison",
            supporting_paper_ids=[
                "A",
                "B",
            ],
            landscape_basis=[
                LandscapeBasis(
                    dimension="method_family",
                    value="Method Alpha",
                    count=2,
                    total=4,
                    prevalence=0.5,
                    paper_ids=["A", "C"],
                ),
                LandscapeBasis(
                    dimension="method_family",
                    value="Method Beta",
                    count=2,
                    total=4,
                    prevalence=0.5,
                    paper_ids=["B", "D"],
                ),
            ],
        )

    def test_missing_comparison_false_positive_is_rejected(self):
        candidate = self._comparison_candidate()

        record = evidence_record(
            "X",
            methods=[
                "Method Alpha",
            ],
            comparisons=[
                "Method Gamma",
            ],
            findings=[
                (
                    "Background literature also discusses "
                    "Method Beta"
                )
            ],
        )

        verifier = make_verifier(
            [
                paper(
                    "X",
                    title=(
                        "Method Alpha compared with "
                        "Method Gamma for Task Omega"
                    ),
                )
            ],
            [
                record,
            ],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertNotIn(
            "X",
            result.contradicting_paper_ids,
        )

    def test_missing_comparison_true_positive_is_confirmed(self):
        candidate = self._comparison_candidate()

        record = evidence_record(
            "X",
            methods=[
                "Method Alpha",
            ],
            comparisons=[
                "Method Beta",
            ],
            findings=[
                (
                    "Method Alpha was directly compared "
                    "with Method Beta"
                )
            ],
        )

        verifier = make_verifier(
            [
                paper(
                    "X",
                    title=(
                        "Method Alpha versus Method Beta "
                        "for Task Omega"
                    ),
                )
            ],
            [
                record,
            ],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertEqual(
            result.final_label,
            "well_studied",
        )
        self.assertIn(
            "X",
            result.contradicting_paper_ids,
        )

    # ------------------------------------------------------------------
    # Title-only evidence
    # ------------------------------------------------------------------

    def test_title_only_complete_match_is_potential_not_direct(self):
        record = evidence_record(
            "T",
            methods=[
                "Method Alpha",
            ],
            population=[
                "Population Delta",
            ],
            constraints=[
                "Resource Constraint",
            ],
            source="title",
        )

        verifier = make_verifier(
            [
                paper(
                    "T",
                    title=(
                        "Method Alpha for Task Omega in "
                        "Population Delta under Resource Constraint"
                    ),
                    abstract=None,
                )
            ],
            [
                record,
            ],
        )

        assessment = verifier.assess_idea(
            self.idea,
            LandscapeAnalyzer().analyze(
                [record]
            ),
            [record],
        )

        self.assertEqual(
            assessment.counterexample_paper_ids,
            [],
        )
        self.assertIn(
            "T",
            assessment.potential_match_paper_ids,
        )
        self.assertEqual(
            assessment.label,
            "uncertain",
        )

    # ------------------------------------------------------------------
    # Provider and extraction failures
    # ------------------------------------------------------------------

    def test_verification_failure_forces_uncertain(self):
        candidate = self._combination_candidate()

        verifier = make_verifier(
            [],
            [],
            fail=True,
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertEqual(
            result.final_label,
            "uncertain",
        )
        self.assertTrue(
            result.verification.failures
        )

    def test_extraction_failure_forces_uncertain(self):
        candidate = self._combination_candidate()

        verifier = make_verifier(
            [
                paper("X"),
            ],
            [],
            extraction_failures=[
                RuntimeError(
                    "fake extraction failure"
                )
            ],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertEqual(
            result.final_label,
            "uncertain",
        )
        self.assertTrue(
            result.verification.failures
        )

    def test_zero_retrieved_papers_is_uncertain(self):
        candidate = self._combination_candidate()

        verifier = make_verifier(
            [],
            [],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertEqual(
            result.final_label,
            "uncertain",
        )

    # ------------------------------------------------------------------
    # Promising-gap semantics
    # ------------------------------------------------------------------

    def test_successful_search_without_counterexample_can_be_promising_gap(self):
        candidate = self._combination_candidate()

        unrelated = evidence_record(
            "Z",
            problem="Unrelated Task",
            methods=[
                "Method Zeta",
            ],
            constraints=[
                "Different Constraint",
            ],
        )

        verifier = make_verifier(
            [
                paper(
                    "Z",
                    title="Unrelated Study",
                )
            ],
            [
                unrelated,
            ],
        )

        result = verifier.verify(
            self.idea,
            candidate,
            [],
        )

        self.assertEqual(
            result.final_label,
            "promising_gap",
        )

        reason = result.verification.reason.casefold()

        self.assertNotIn(
            "globally novel",
            reason,
        )
        self.assertNotIn(
            "no previous research",
            reason,
        )
        self.assertNotIn(
            "first-ever",
            reason,
        )
        self.assertIn(
            "not a claim of global novelty",
            reason,
        )
    # ------------------------------------------------------------------
    # Duplicate targeted evidence precedence
    # ------------------------------------------------------------------

    def test_targeted_evidence_overrides_initial_duplicate_record(self):
        initial = evidence_record(
            "X",
            methods=[
                "Method Beta",
            ],
            population=[
                "Population Delta",
            ],
            constraints=[
                "Resource Constraint",
            ],
        )

        targeted = evidence_record(
            "X",
            methods=[
                "Method Alpha",
            ],
            population=[
                "Population Delta",
            ],
            constraints=[
                "Resource Constraint",
            ],
        )

        verifier = make_verifier(
            [
                paper("X"),
            ],
            [
                targeted,
            ],
        )

        assessment = verifier.assess_idea(
            self.idea,
            LandscapeAnalyzer().analyze(
                [initial]
            ),
            [
                initial,
            ],
        )

        self.assertEqual(
            assessment.label,
            "well_studied",
        )

        self.assertEqual(
            assessment.counterexample_paper_ids,
            [
                "X",
            ],
        )

        self.assertEqual(
            assessment.supporting_paper_ids.count(
                "X"
            ),
            1,
        )

    # ------------------------------------------------------------------
    # Landscape-grounded candidate generation
    # ------------------------------------------------------------------

    def test_combination_candidate_requires_landscape_grounding(self):
        evidence = [
            evidence_record(
                "A",
                methods=[
                    "Method Alpha",
                ],
            ),
            evidence_record(
                "B",
                methods=[
                    "Method Beta",
                ],
                constraints=[
                    "Resource Constraint",
                ],
            ),
        ]

        landscape = LandscapeAnalyzer().analyze(
            evidence
        )

        candidates = GapCandidateGenerator().generate(
            self.idea,
            landscape,
            evidence,
        )

        combinations = [
            item
            for item in candidates
            if item.pattern_type
            == "combination_gap"
        ]

        self.assertTrue(
            combinations
        )

        for candidate in combinations:
            self.assertTrue(
                candidate.landscape_basis
            )

    def test_empty_landscape_produces_no_candidates(self):
        generator = GapCandidateGenerator()

        candidates = generator.generate(
            self.idea,
            LandscapeAnalyzer().analyze(
                []
            ),
            [],
        )

        self.assertEqual(
            candidates,
            [],
        )

    # ------------------------------------------------------------------
    # Candidate consolidation
    # ------------------------------------------------------------------

    def test_near_duplicate_candidates_merge_provenance(self):
        def candidate(
            title: str,
            paper_id: str,
        ) -> GapCandidate:
            return GapCandidate(
                title=title,
                description=(
                    "Alternative validation remains limited."
                ),
                category="dataset",
                rationale=(
                    "Repeated landscape characteristic."
                ),
                pattern_type="narrow_dataset_setting",
                supporting_paper_ids=[
                    paper_id
                ],
                supporting_evidence=[
                    GapEvidence(
                        paper_id=paper_id,
                        evidence_type="dataset_type",
                        value="public benchmark",
                        evidence_text="public benchmark",
                        role="contextual_support",
                    )
                ],
                landscape_basis=[
                    LandscapeBasis(
                        dimension="dataset_type",
                        value="public benchmark",
                        count=2,
                        total=3,
                        prevalence=2 / 3,
                        paper_ids=[
                            "A",
                            "B",
                        ],
                    )
                ],
            )

        merged = consolidate_candidates(
            [
                candidate(
                    "Validation beyond public benchmark data",
                    "A",
                ),
                candidate(
                    "Evaluation beyond public benchmark data",
                    "B",
                ),
            ]
        )

        self.assertEqual(
            len(merged),
            1,
        )

        self.assertEqual(
            set(
                merged[0].supporting_paper_ids
            ),
            {
                "A",
                "B",
            },
        )

    # ------------------------------------------------------------------
    # Semantic guardrails
    # ------------------------------------------------------------------

    def test_limited_label_claim_requires_constraint_evidence(self):
        self.assertFalse(
            validate_evidence_semantics(
                claim_text=(
                    "limited labeled data"
                ),
                evidence_type="dataset",
                evidence=claim(
                    "annotated dataset"
                ),
            )
        )

        self.assertTrue(
            validate_evidence_semantics(
                claim_text=(
                    "limited labeled data"
                ),
                evidence_type="constraint",
                evidence=claim(
                    "limited labeled data"
                ),
            )
        )

    def test_accuracy_does_not_prove_generalization(self):
        self.assertFalse(
            validate_evidence_semantics(
                claim_text=(
                    "generalization"
                ),
                evidence_type="finding",
                evidence=claim(
                    "95 percent accuracy"
                ),
            )
        )

    def test_cost_limitation_does_not_prove_measured_efficiency(self):
        evidence = claim(
            "computational cost",
            (
                "Computational cost remains "
                "a limitation."
            ),
        )

        self.assertFalse(
            validate_evidence_semantics(
                claim_text=(
                    "computational efficiency"
                ),
                evidence_type="limitation",
                evidence=evidence,
            )
        )

    # ------------------------------------------------------------------
    # Constraint matching must respect structured fields
    # ------------------------------------------------------------------

    def test_constraint_in_main_findings_does_not_count_as_constraint(self):
        record = evidence_record(
            "A",
            methods=[
                "Method Alpha",
            ],
            findings=[
                "Resource Constraint affected the experiment"
            ],
        )

        self.assertFalse(
            _constraint_matches(
                self.idea,
                record,
            )
        )

    def test_explicit_structured_constraint_matches(self):
        record = evidence_record(
            "A",
            methods=[
                "Method Alpha",
            ],
            constraints=[
                "Resource Constraint",
            ],
        )

        self.assertTrue(
            _constraint_matches(
                self.idea,
                record,
            )
        )


if __name__ == "__main__":
    unittest.main()