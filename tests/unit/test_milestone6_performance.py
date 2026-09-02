import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from src.analysis.clustering import LandscapeAnalyzer
from src.analysis.gap_candidates import GapCandidateGenerator
from src.analysis.gap_candidates import prune_redundant_candidates
from src.analysis.models import GapCandidate, LandscapeBasis, VerificationQuery
from src.analysis.verification import (
    GapVerifier,
    _idea_match_strength,
    _pre_screen_papers,
    deduplicate_query_phrases,
)
from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.models.idea import ResearchIdea
from src.models.paper import Paper, RetrievalProvenance
from src.ranking.semantic import OpenAIEmbeddingProvider
from src.ranking.lexical import LexicalScorer
from src.retrieval.multi_query import MultiQueryRetriever


def claim(value: str, text: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        evidence_text=text or value,
        source="abstract",
        confidence=0.9,
    )


def evidence(
    paper_id: str,
    *,
    problem: str | None = None,
    method: str | None = None,
    setting: str | None = None,
    constraint: str | None = None,
) -> PaperEvidence:
    return PaperEvidence(
        paper_id=paper_id,
        title=f"Study {paper_id}",
        study_type="empirical",
        research_objective=claim(problem) if problem else None,
        method_or_intervention=[claim(method)] if method else [],
        population_or_setting=[claim(setting)] if setting else [],
        constraints=[claim(constraint)] if constraint else [],
        extraction_confidence=0.9,
    )


class EmptyRetriever:
    provider_name = "fake"

    def search(self, request):
        return []


class EmptyExtractor:
    failures = []

    def extract_many(self, papers, limit=None):
        return []


class AdaptiveRetriever:
    provider_name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, request):
        self.calls.append(request.query.text)
        if request.query.text == "first":
            return [Paper(id="p1", title="Paper one", abstract="one")]
        if request.query.text == "third":
            return [Paper(id="p2", title="Paper two", abstract="two")]
        return []


class RedundantRetriever:
    provider_name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, request):
        self.calls.append(request.query.text)
        return [Paper(id="p1", title="Paper one", abstract="one")]


class CountingEmptyRetriever:
    provider_name = "fake"

    def __init__(self):
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return []


class SinglePaperRetriever:
    provider_name = "fake"

    def __init__(self):
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return [
            Paper(
                id="counterexample",
                title="method_a problem_a constraint_a",
                abstract="method_a addresses problem_a under constraint_a",
            )
        ]


class SingleEvidenceExtractor:
    failures = []

    def extract_many(self, papers, limit=None):
        return [
            evidence(
                paper.id,
                problem="problem_a",
                method="method_a",
                constraint="constraint_a",
            )
            for paper in papers[:limit]
        ]


class PerformanceRegressionTest(unittest.TestCase):
    def test_same_paper_higher_order_combination_is_observed(self):
        records = [
            evidence(
                "p1",
                problem="problem_a",
                method="method_a",
                setting="setting_a",
                constraint="constraint_a",
            )
        ]
        landscape = LandscapeAnalyzer().analyze(records)

        self.assertTrue(
            any(
                item.dimensions == {
                    "problem": "problem a",
                    "method_family": "method a",
                    "population_or_setting": "setting a",
                }
                for item in landscape.combinations
            )
        )

        idea = ResearchIdea(
            original_text="method_a problem_a setting_a constraint_a",
            problem=["problem_a"],
            intervention_or_method=["method_a"],
            population=["setting_a"],
            constraints=["constraint_a"],
        )
        candidates = GapCandidateGenerator(max_candidates=50).generate(
            idea,
            landscape,
            records,
        )
        self.assertFalse(
            any(
                {
                    (item.dimension, item.value)
                    for item in candidate.landscape_basis
                }
                == {
                    ("method_family", "method a"),
                    ("population_or_setting", "setting a"),
                    ("constraint", "constraint a"),
                }
                for candidate in candidates
                if candidate.pattern_type == "combination_gap"
            )
        )
        self.assertFalse(
            any(
                candidate.pattern_type == "combination_gap"
                for candidate in GapCandidateGenerator(max_candidates=50).generate(
                    idea,
                    landscape,
                    records,
                )
            )
        )

    def test_separate_papers_do_not_create_observed_pair(self):
        records = [
            evidence("p1", method="method_a"),
            evidence("p2", constraint="constraint_a"),
        ]
        landscape = LandscapeAnalyzer().analyze(records)
        self.assertFalse(
            any(
                item.dimensions == {
                    "method_family": "method a",
                    "constraint": "constraint a",
                }
                for item in landscape.combinations
            )
        )

    def test_missing_three_component_combination_is_generated(self):
        records = [
            evidence("p1", problem="problem_a", method="method_a"),
            evidence("p2", method="method_a", setting="setting_a"),
            evidence("p3", constraint="constraint_a"),
            evidence("p4", problem="problem_a", setting="setting_a"),
        ]
        idea = ResearchIdea(
            original_text="method_a problem_a setting_a constraint_a",
            problem=["problem_a"],
            intervention_or_method=["method_a"],
            population=["setting_a"],
            constraints=["constraint_a"],
        )
        candidates = GapCandidateGenerator(max_candidates=50).generate(
            idea,
            LandscapeAnalyzer().analyze(records),
            records,
        )
        self.assertTrue(
            any(
                len(candidate.landscape_basis) == 3
                and {
                    (item.dimension, item.value)
                    for item in candidate.landscape_basis
                }
                == {
                    ("problem", "problem a"),
                    ("method_family", "method a"),
                    ("constraint", "constraint a"),
                }
                for candidate in candidates
                if candidate.pattern_type == "combination_gap"
            )
        )

    def test_one_structured_claim_can_support_multiple_explicit_facets(self):
        idea = ResearchIdea(
            original_text="method_a addresses problem_a under constraint_a",
            problem=["problem_a"],
            intervention_or_method=["method_a"],
            constraints=["constraint_a"],
        )
        record = PaperEvidence(
            paper_id="p1",
            title="Study p1",
            study_type="empirical",
            method_or_intervention=[
                claim(
                    "method_a addresses problem_a under constraint_a",
                )
            ],
            extraction_confidence=0.9,
        )
        complete, matched = _idea_match_strength(idea, record)
        self.assertTrue(complete)
        self.assertEqual(set(matched), {"problem", "method", "constraint"})

    def test_phrase_deduplication_preserves_distinct_phrases(self):
        self.assertEqual(
            deduplicate_query_phrases(["A", "B", "A", "C", "B"]),
            ["A", "B", "C"],
        )
        self.assertEqual(
            deduplicate_query_phrases(["A B", "A C", "A B"]),
            ["A B", "A C"],
        )

    def test_facet_aware_rationale_names_unconfirmed_components(self):
        idea = ResearchIdea(
            original_text="method_a problem_a setting_a constraint_a",
            problem=["problem_a"],
            intervention_or_method=["method_a"],
            population=["setting_a"],
            constraints=["constraint_a"],
        )
        record = evidence(
            "initial",
            problem="problem_a",
            method="method_a",
            setting="setting_a",
        )
        verifier = GapVerifier(
            MultiQueryRetriever(
                EmptyRetriever(),
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            EmptyExtractor(),
        )
        assessment = verifier.assess_idea(
            idea,
            LandscapeAnalyzer().analyze([record]),
            [record],
        )
        self.assertIn("constraint", assessment.rationale.casefold())
        self.assertNotIn("available evidence contains failures", assessment.rationale.casefold())

    def test_adaptive_verification_keeps_meaningful_query_after_empty_one(self):
        backend = AdaptiveRetriever()
        retriever = MultiQueryRetriever(
            backend,
            max_candidates=20,
            per_route_limit=5,
            max_workers=2,
        )
        from src.models.query import SearchQuery

        result = retriever.retrieve_verification(
            [
                SearchQuery(text="first", strategy="verification_counterexample", source="deterministic"),
                SearchQuery(text="second", strategy="verification_counterexample", source="deterministic"),
                SearchQuery(text="third", strategy="verification_counterexample", source="deterministic"),
            ],
            limit=10,
        )
        self.assertEqual(result.requested_routes, 3)
        self.assertEqual([paper.id for paper in result.papers], ["p1", "p2"])

    def test_adaptive_verification_stops_after_redundant_marginal_coverage(self):
        backend = RedundantRetriever()
        retriever = MultiQueryRetriever(
            backend,
            max_candidates=20,
            per_route_limit=5,
            max_workers=2,
        )
        from src.models.query import SearchQuery

        queries = [
            SearchQuery(text="same phrase", strategy="verification_counterexample", source="deterministic")
            for _ in range(4)
        ]
        result = retriever.retrieve_verification(queries, limit=10)
        self.assertLess(result.requested_routes, len(queries))
        self.assertGreaterEqual(result.requested_routes, 2)

    def test_embedding_cache_reuses_unchanged_text_across_instances(self):
        class Embeddings:
            def __init__(self):
                self.calls = []

            def create(self, *, model, input):
                self.calls.append((model, list(input)))
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
                        for index, _ in enumerate(input)
                    ]
                )

        with TemporaryDirectory() as directory:
            first_client = SimpleNamespace(embeddings=Embeddings())
            first = OpenAIEmbeddingProvider(
                client=first_client,
                model="embedding-a",
                cache_path=f"{directory}/cache.sqlite3",
            )
            first.embed_documents(["paper text"])
            self.assertEqual(len(first_client.embeddings.calls), 1)

            second_client = SimpleNamespace(embeddings=Embeddings())
            second = OpenAIEmbeddingProvider(
                client=second_client,
                model="embedding-a",
                cache_path=f"{directory}/cache.sqlite3",
            )
            second.embed_documents(["paper text"])
            self.assertEqual(len(second_client.embeddings.calls), 0)
            self.assertEqual(
                second.metrics_snapshot()["persistent_embedding_cache_hits"],
                1,
            )

    def test_candidate_dominance_keeps_strongest_nested_hypothesis(self):
        idea = ResearchIdea(
            original_text="problem_a method_a setting_a constraint_a",
            problem=["problem_a"],
            intervention_or_method=["method_a"],
            population=["setting_a"],
            constraints=["constraint_a"],
        )

        def candidate(values):
            basis = [
                LandscapeBasis(
                    dimension=dimension,
                    value=value,
                    count=1,
                    total=4,
                    prevalence=0.25,
                    paper_ids=["p1"],
                )
                for dimension, value in values
            ]
            return GapCandidate(
                title="synthetic combination",
                description="synthetic combination",
                category="combination",
                pattern_type="combination_gap",
                rationale="synthetic rationale",
                landscape_basis=basis,
                supporting_paper_ids=["p1", "p2"],
            )

        c1 = candidate([
            ("problem", "problem_a"),
            ("method_family", "method_a"),
            ("population_or_setting", "setting_a"),
            ("constraint", "constraint_a"),
        ])
        c2 = candidate([
            ("problem", "problem_a"),
            ("method_family", "method_a"),
            ("population_or_setting", "setting_a"),
        ])
        c3 = candidate([
            ("method_family", "method_a"),
            ("population_or_setting", "setting_a"),
        ])
        c4 = candidate([
            ("problem", "problem_a"),
            ("constraint", "constraint_a"),
            ("dataset", "dataset_a"),
        ])

        kept, removed = prune_redundant_candidates(
            [c1, c2, c3, c4],
            idea,
        )
        kept_sets = [
            {(item.dimension, item.value) for item in item.landscape_basis}
            for item in kept
        ]

        self.assertEqual(removed, 2)
        self.assertIn(
            {
                ("problem", "problem_a"),
                ("method_family", "method_a"),
                ("population_or_setting", "setting_a"),
                ("constraint", "constraint_a"),
            },
            kept_sets,
        )
        self.assertIn(
            {
                ("problem", "problem_a"),
                ("constraint", "constraint_a"),
                ("dataset", "dataset_a"),
            },
            kept_sets,
        )

    def test_same_candidate_outcome_reuses_verification_without_retrieval(self):
        backend = CountingEmptyRetriever()
        verifier = GapVerifier(
            MultiQueryRetriever(
                backend,
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            EmptyExtractor(),
        )
        candidate = GapCandidate(
            title="method_a with constraint_a",
            description="synthetic combination",
            category="combination",
            pattern_type="combination_gap",
            rationale="synthetic rationale",
            supporting_paper_ids=["p1", "p2"],
            landscape_basis=[
                LandscapeBasis(
                    dimension="method_family",
                    value="method_a",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["p1"],
                ),
                LandscapeBasis(
                    dimension="constraint",
                    value="constraint_a",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["p2"],
                ),
            ],
        )

        verifier.verify(ResearchIdea(original_text="method_a constraint_a"), candidate, [])
        calls_after_first = backend.calls
        verifier.verify(
            ResearchIdea(original_text="method_a constraint_a"),
            candidate.model_copy(deep=True),
            [],
        )

        self.assertGreater(calls_after_first, 0)
        self.assertEqual(backend.calls, calls_after_first)
        self.assertEqual(
            verifier.metrics_snapshot()["candidate_verification_results_reused_exact"],
            1,
        )
        self.assertEqual(
            verifier.metrics_snapshot()["candidate_verification_results_reused_nonidentical"],
            0,
        )

    def test_preverification_guard_rejects_observed_combination_without_retrieval(self):
        backend = CountingEmptyRetriever()
        verifier = GapVerifier(
            MultiQueryRetriever(
                backend,
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            EmptyExtractor(),
        )
        candidate = GapCandidate(
            id="observed-candidate",
            title="problem_a method_b constraint_c",
            description="synthetic combination",
            category="combination",
            pattern_type="combination_gap",
            rationale="synthetic rationale",
            supporting_paper_ids=["p1"],
            landscape_basis=[
                LandscapeBasis(
                    dimension="problem",
                    value="problem_a",
                    count=1,
                    total=1,
                    prevalence=1.0,
                    paper_ids=["p1"],
                ),
                LandscapeBasis(
                    dimension="method_family",
                    value="method_b",
                    count=1,
                    total=1,
                    prevalence=1.0,
                    paper_ids=["p1"],
                ),
                LandscapeBasis(
                    dimension="constraint",
                    value="constraint_c",
                    count=1,
                    total=1,
                    prevalence=1.0,
                    paper_ids=["p1"],
                ),
            ],
        )
        records = [
            evidence(
                "p1",
                problem="problem_a",
                method="method_b",
                constraint="constraint_c",
            )
        ]

        result = verifier.verify(
            ResearchIdea(original_text="problem_a method_b constraint_c"),
            candidate,
            records,
        )

        self.assertEqual(backend.calls, 0)
        self.assertEqual(result.final_label, "well_studied")
        self.assertEqual(
            verifier.metrics_snapshot()["observed_candidates_rejected_preverification"],
            1,
        )

    def test_pruned_candidates_are_not_returned_or_resurrected(self):
        backend = CountingEmptyRetriever()
        verifier = GapVerifier(
            MultiQueryRetriever(
                backend,
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            EmptyExtractor(),
        )
        idea = ResearchIdea(
            original_text="problem_a method_b setting_c constraint_d",
            problem=["problem_a"],
            intervention_or_method=["method_b"],
            population=["setting_c"],
            constraints=["constraint_d"],
        )

        def make_candidate(candidate_id, values):
            return GapCandidate(
                id=candidate_id,
                title=candidate_id,
                description="synthetic combination",
                category="combination",
                pattern_type="combination_gap",
                rationale="synthetic rationale",
                supporting_paper_ids=["p1", "p2"],
                landscape_basis=[
                    LandscapeBasis(
                        dimension=dimension,
                        value=value,
                        count=1,
                        total=4,
                        prevalence=0.25,
                        paper_ids=["p1"],
                    )
                    for dimension, value in values
                ],
            )

        strong = make_candidate(
            "strong",
            [
                ("problem", "problem_a"),
                ("method_family", "method_b"),
                ("population_or_setting", "setting_c"),
                ("constraint", "constraint_d"),
            ],
        )
        subset = make_candidate(
            "subset",
            [
                ("problem", "problem_a"),
                ("method_family", "method_b"),
            ],
        )

        result = verifier.verify_many(idea, [strong, subset], [])

        self.assertEqual([item.id for item in result], ["strong"])
        self.assertEqual(
            verifier.metrics_snapshot()["candidate_hypotheses_after_pruning"],
            1,
        )
        self.assertEqual(
            verifier.metrics_snapshot()["candidate_verification_results_reused_nonidentical"],
            0,
        )

    def test_nonidentical_candidate_does_not_reuse_final_assessment(self):
        backend = CountingEmptyRetriever()
        verifier = GapVerifier(
            MultiQueryRetriever(
                backend,
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            EmptyExtractor(),
        )

        def make_candidate(candidate_id, values):
            return GapCandidate(
                id=candidate_id,
                title=candidate_id,
                description="synthetic combination",
                category="combination",
                pattern_type="combination_gap",
                rationale="synthetic rationale",
                supporting_paper_ids=["p1", "p2"],
                landscape_basis=[
                    LandscapeBasis(
                        dimension=dimension,
                        value=value,
                        count=1,
                        total=4,
                        prevalence=0.25,
                        paper_ids=["p1"],
                    )
                    for dimension, value in values
                ],
            )

        first = make_candidate(
            "first",
            [("method_family", "method_b"), ("constraint", "constraint_c")],
        )
        second = make_candidate(
            "second",
            [
                ("method_family", "method_b"),
                ("constraint", "constraint_c"),
                ("population_or_setting", "setting_d"),
            ],
        )
        idea = ResearchIdea(original_text="method_b constraint_c setting_d")

        verifier.verify(idea, first, [])
        calls_after_first = backend.calls
        verifier.verify(idea, second, [])

        self.assertGreater(backend.calls, calls_after_first)
        self.assertEqual(
            verifier.metrics_snapshot()["candidate_verification_results_reused_nonidentical"],
            0,
        )

    def test_direct_facet_matching_reports_explicit_partial_cross_field_support(self):
        idea = ResearchIdea(
            original_text="problem_a method_b constraint_c setting_d",
            problem=["problem_a"],
            intervention_or_method=["method_b"],
            constraints=["constraint_c"],
            population=["setting_d"],
        )
        record = PaperEvidence(
            paper_id="p1",
            title="Study p1",
            study_type="empirical",
            method_or_intervention=[
                claim(
                    "method_b",
                    "method_b addresses problem_a under constraint_c",
                )
            ],
            extraction_confidence=0.9,
        )

        complete, matched = _idea_match_strength(idea, record)

        self.assertFalse(complete)
        self.assertEqual(
            set(matched),
            {"problem", "method", "constraint"},
        )

    def test_same_normalized_verification_query_uses_retrieval_cache(self):
        backend = CountingEmptyRetriever()
        verifier = GapVerifier(
            MultiQueryRetriever(
                backend,
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            EmptyExtractor(),
        )
        query = VerificationQuery(
            candidate_id="candidate",
            query="A  B",
            pattern_type="combination_gap",
            strategy="verification_counterexample",
        )
        verifier._retrieve_verification_queries([query])
        verifier._retrieve_verification_queries([
            query.model_copy(update={"query": "a b"})
        ])

        self.assertEqual(backend.calls, 1)
        self.assertEqual(
            verifier.metrics_snapshot()["verification_queries_cache_hits"],
            1,
        )

    def test_confirmed_counterexample_stops_remaining_candidate_queries(self):
        backend = SinglePaperRetriever()
        verifier = GapVerifier(
            MultiQueryRetriever(
                backend,
                max_candidates=20,
                per_route_limit=5,
                max_workers=2,
            ),
            SingleEvidenceExtractor(),
        )
        candidate = GapCandidate(
            title="method_a with constraint_a",
            description="synthetic combination",
            category="combination",
            pattern_type="combination_gap",
            rationale="synthetic rationale",
            supporting_paper_ids=["p1", "p2"],
            landscape_basis=[
                LandscapeBasis(
                    dimension="method_family",
                    value="method_a",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["p1"],
                ),
                LandscapeBasis(
                    dimension="constraint",
                    value="constraint_a",
                    count=1,
                    total=2,
                    prevalence=0.5,
                    paper_ids=["p2"],
                ),
            ],
        )
        idea = ResearchIdea(
            original_text="method_a problem_a constraint_a",
            problem=["problem_a"],
            intervention_or_method=["method_a"],
            constraints=["constraint_a"],
        )

        result = verifier.verify(idea, candidate, [])

        self.assertEqual(backend.calls, 1)
        self.assertEqual(result.final_label, "well_studied")
        self.assertEqual(
            verifier.metrics_snapshot()["verification_candidates_early_stopped"],
            1,
        )

    def test_prescreen_rejects_clear_nonmatching_paper_but_keeps_partial_match(self):
        papers = [
            Paper(id="irrelevant", title="Unrelated study", abstract="No matching components."),
            Paper(id="partial", title="method_a study", abstract="A partial evaluation of method_a."),
        ]
        screened = _pre_screen_papers(
            papers,
            ["method_a", "constraint_a", "setting_a"],
            LexicalScorer(),
            "method_a constraint_a setting_a",
        )
        self.assertEqual([paper.id for paper in screened], ["partial"])


if __name__ == "__main__":
    unittest.main()
