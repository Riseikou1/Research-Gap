import unittest
from datetime import datetime, timezone

from src.analysis.gap_candidates import GapCandidateGenerator
from src.analysis.verification import GapVerifier
from src.extraction.evidence import EvidenceItem, PaperEvidence
from src.models.paper import Paper, RetrievalProvenance
from src.pipeline import ResearchPipeline
from src.query.deterministic import DeterministicDecomposer
from src.ranking.lexical import LexicalScorer
from src.ranking.reranker import HybridReranker
from src.retrieval.multi_query import MultiQueryRetriever


def claim(value: str, text: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        value=value,
        evidence_text=text or value,
        source="abstract",
        confidence=0.9,
    )


def initial_evidence(paper_id: str) -> PaperEvidence:
    if paper_id == "A":
        return PaperEvidence(
            paper_id="A",
            title="Method Alpha for Task Omega",
            study_type="empirical",
            research_objective=claim("Task Omega"),
            method_or_intervention=[claim("Method Alpha")],
            extraction_confidence=0.9,
        )

    if paper_id == "B":
        return PaperEvidence(
            paper_id="B",
            title="Task Omega under Constraint One",
            study_type="empirical",
            research_objective=claim("Task Omega"),
            method_or_intervention=[claim("Method Beta")],
            constraints=[claim("Constraint One")],
            extraction_confidence=0.9,
        )

    raise ValueError(f"Unexpected paper id: {paper_id}")


class Milestone6Retriever:
    provider_name = "fake"

    def search(self, request):
        if request.query.strategy == "verification_counterexample":
            papers = [
                Paper(
                    id="X",
                    title="Method Alpha under Constraint One for Task Omega",
                    abstract=(
                        "We evaluate Method Alpha for Task Omega "
                        "under Constraint One."
                    ),
                    publication_year=2025,
                )
            ]
        else:
            papers = [
                Paper(
                    id="A",
                    title="Method Alpha for Task Omega",
                    abstract="We study Method Alpha for Task Omega.",
                    publication_year=2024,
                ),
                Paper(
                    id="B",
                    title="Task Omega under Constraint One",
                    abstract=(
                        "We study Method Beta for Task Omega "
                        "under Constraint One."
                    ),
                    publication_year=2023,
                ),
            ]

        for rank, paper in enumerate(papers, start=1):
            paper.provenance = [
                RetrievalProvenance(
                    query=request.query,
                    provider=self.provider_name,
                    mode=request.mode,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    provider_rank=rank,
                )
            ]

        return papers


class Milestone6Extractor:
    failures = []

    def extract_many(self, papers, limit=None):
        result = []

        for paper in papers[:limit]:
            if paper.id == "X":
                result.append(
                    PaperEvidence(
                        paper_id="X",
                        title=paper.title,
                        study_type="empirical",
                        research_objective=claim("Task Omega"),
                        method_or_intervention=[claim("Method Alpha")],
                        constraints=[claim("Constraint One")],
                        main_findings=[
                            claim(
                                "Method Alpha was evaluated under Constraint One"
                            )
                        ],
                        extraction_confidence=0.9,
                    )
                )
            else:
                result.append(
                    initial_evidence(paper.id)
                )

        return result


class Milestone6PipelineSmokeTest(unittest.TestCase):
    def test_full_pipeline_generates_and_verifies_combination_gap(self):
        retriever = MultiQueryRetriever(
            Milestone6Retriever(),
            max_candidates=20,
            per_route_limit=5,
            max_workers=2,
        )

        extractor = Milestone6Extractor()

        verifier = GapVerifier(
            retriever,
            extractor,
        )

        pipeline = ResearchPipeline(
            decomposer=DeterministicDecomposer(),
            retriever=retriever,
            reranker=HybridReranker(
                LexicalScorer(),
                None,
            ),
            extractor=extractor,
            gap_generator=GapCandidateGenerator(),
            gap_verifier=verifier,
        )

        result = pipeline.run(
        "Using Method Alpha for Task Omega under Constraint One",
            top_k=2,
        )

        self.assertTrue(
            result.evidence
        )

        self.assertIsNotNone(
            result.landscape
        )

        combination_gaps = [
            gap
            for gap in result.gaps
            if gap.pattern_type == "combination_gap"
        ]

        self.assertTrue(
            combination_gaps
        )

        gap = next(
            candidate
            for candidate in combination_gaps
            if {
                item.dimension
                for item in candidate.landscape_basis
            } == {
                "method_family",
                "constraint",
            }
        )

        self.assertEqual(
            gap.final_label,
            "well_studied",
        )

        self.assertIn(
            "X",
            gap.contradicting_paper_ids,
        )

        self.assertTrue(
            gap.verification
        )

        self.assertTrue(
            gap.verification.verification_queries
        )

        self.assertEqual(
            gap.verification.label,
            "well_studied",
        )

        serialized = result.to_dict()

        serialized_gap = next(
            item
            for item in serialized["gaps"]
            if item["id"] == gap.id
        )

        self.assertIn(
            "verification",
            serialized_gap,
        )

        self.assertEqual(
            serialized_gap["verification"]["label"],
            "well_studied",
        )


if __name__ == "__main__":
    unittest.main()