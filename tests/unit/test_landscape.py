import unittest

from src.analysis.clustering import LandscapeAnalyzer
from src.analysis.comparison import (
    normalize_constraint,
    normalize_dataset,
    normalize_method_family,
    normalize_metric,
    normalize_problem,
    to_paper_features,
)
from src.extraction.evidence import EvidenceItem, LimitationEvidence, PaperEvidence
from src.reporting.landscape import format_landscape


def claim(value: str, text: str | None = None) -> EvidenceItem:
    return EvidenceItem(value=value, evidence_text=text or value, source="abstract", confidence=0.9)


def paper(
    paper_id: str,
    *,
    method: str | None = None,
    dataset: str | None = None,
    objective: str | None = None,
    population: str | None = None,
    baseline: str | None = None,
    metric: str | None = None,
    finding: str | None = None,
    limitation: str | None = None,
    future_work: str | None = None,
    constraints: list[str] | None = None,
    study_type: str = "empirical",
) -> PaperEvidence:
    return PaperEvidence(
        paper_id=paper_id,
        title=f"Study {paper_id}",
        study_type=study_type,
        research_objective=claim(objective) if objective else None,
        population_or_setting=[claim(population)] if population else [],
        method_or_intervention=[claim(method)] if method else [],
        comparison_or_baseline=[claim(baseline)] if baseline else [],
        datasets=[claim(dataset)] if dataset else [],
        evaluation_metrics=[claim(metric)] if metric else [],
        main_findings=[claim(finding)] if finding else [],
        constraints=[claim(value) for value in constraints or []],
        limitations=[
            LimitationEvidence(
                value=limitation,
                evidence_text=limitation,
                source="abstract",
                confidence=0.9,
                author_stated=True,
            )
        ] if limitation else [],
        future_work=[claim(future_work)] if future_work else [],
        extraction_confidence=0.9,
    )


class LandscapeTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # Generic normalization
    # ------------------------------------------------------------------

    def test_unknown_problem_is_preserved_instead_of_forced_into_domain_bucket(self):
        value = normalize_problem("quantum phase estimation")
        self.assertTrue(value)
        self.assertNotEqual(value, "other")
        self.assertIn("quantum", value)
        self.assertIn("phase", value)

    def test_problem_normalization_is_deterministic(self):
        first = normalize_problem("Task Omega")
        second = normalize_problem("  task   omega  ")
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_unknown_methods_are_preserved_and_remain_distinct(self):
        alpha = normalize_method_family("Method Alpha")
        beta = normalize_method_family("Method Beta")

        self.assertTrue(alpha)
        self.assertTrue(beta)
        self.assertNotEqual(alpha, beta)
        self.assertNotEqual(alpha, "other")
        self.assertNotEqual(beta, "other")

    def test_method_normalization_handles_harmless_spelling_variants(self):
        first = normalize_method_family("Method Alpha")
        second = normalize_method_family("method-alpha")
        self.assertEqual(first, second)

    def test_unknown_named_datasets_are_preserved(self):
        gamma = normalize_dataset("Dataset Gamma")
        delta = normalize_dataset("Dataset Delta")

        self.assertTrue(gamma)
        self.assertTrue(delta)
        self.assertNotEqual(gamma, delta)
        self.assertNotEqual(gamma, "other")

    def test_constraint_normalization_does_not_invent_limited_label_experiment(self):
        self.assertEqual(
            normalize_constraint("limited labelled data"),
            normalize_constraint("limited labeled data"),
        )

        self.assertNotEqual(
            normalize_constraint("requires large labelled datasets"),
            "limited labeled data",
        )

        communication = normalize_constraint("communication cost")
        self.assertTrue(communication)
        self.assertNotEqual(communication, "other")

    def test_metric_normalization_keeps_known_and_unknown_metrics(self):
        self.assertEqual(normalize_metric("classification accuracy"), "accuracy")
        self.assertEqual(normalize_metric("inference time"), "inference time")

        unknown = normalize_metric("spectral distortion index")
        self.assertTrue(unknown)
        self.assertNotEqual(unknown, "other")

    # ------------------------------------------------------------------
    # Paper feature conversion
    # ------------------------------------------------------------------

    def test_feature_conversion_preserves_distinct_scientific_entities(self):
        records = [
            paper(
                "A",
                objective="Task Omega",
                method="Method Alpha",
                dataset="Dataset Gamma",
                population="Population Delta",
                constraints=["Constraint One"],
            ),
            paper(
                "B",
                objective="Task Omega",
                method="Method Beta",
                dataset="Dataset Zeta",
                population="Population Delta",
                constraints=["Constraint Two"],
            ),
        ]

        features = to_paper_features(records)

        self.assertEqual(len(features), 2)
        self.assertNotEqual(features[0].methods, features[1].methods)
        self.assertNotEqual(features[0].datasets, features[1].datasets)
        self.assertEqual(
            features[0].populations_or_settings,
            features[1].populations_or_settings,
        )

    def test_metrics_are_split_into_performance_and_efficiency(self):
        evidence = paper("A", method="Method Alpha")
        evidence.evaluation_metrics = [
            claim("classification accuracy"),
            claim("inference time"),
        ]

        features = to_paper_features([evidence])[0]

        self.assertIn("accuracy", features.performance_metrics)
        self.assertIn("inference time", features.efficiency_metrics)
        self.assertIn("accuracy", features.metrics)
        self.assertIn("inference time", features.metrics)

    def test_explicit_study_type_is_preserved(self):
        records = [
            paper("A", study_type="empirical"),
            paper("B", study_type="methodological"),
            paper("C", study_type="review"),
        ]

        features = to_paper_features(records)

        self.assertEqual(
            [item.study_type for item in features],
            ["empirical", "methodological", "review"],
        )

    def test_raw_evidence_is_not_mutated_by_feature_conversion(self):
        evidence = paper(
            "A",
            objective="Task Omega",
            method="Method Alpha",
            dataset="Dataset Gamma",
            constraints=["Constraint One"],
        )

        original = evidence.model_copy(deep=True).model_dump()

        to_paper_features([evidence])

        self.assertEqual(evidence.model_dump(), original)

    # ------------------------------------------------------------------
    # Frequencies
    # ------------------------------------------------------------------

    def test_frequency_counts_distinct_papers(self):
        records = [
            paper("A", method="Method Alpha"),
            paper("B", method="Method Alpha"),
            paper("C", method="Method Alpha"),
            paper("D", method="Method Beta"),
        ]

        features = to_paper_features(records)
        expected_method = features[0].methods[0]

        landscape = LandscapeAnalyzer().analyze(records)

        frequency = next(
            item
            for item in landscape.frequencies
            if item.dimension == "method" and item.value == expected_method
        )

        self.assertEqual(frequency.count, 3)
        self.assertEqual(frequency.prevalence, 0.75)
        self.assertEqual(frequency.paper_ids, ["A", "B", "C"])

    def test_duplicate_values_inside_one_paper_count_once(self):
        first = paper("A", method="Method Alpha")
        first.method_or_intervention.append(claim("Method Alpha"))

        second = paper("B", method="Method Alpha")

        features = to_paper_features([first, second])
        expected_family = features[0].method_families[0]

        landscape = LandscapeAnalyzer().analyze([first, second])

        frequency = next(
            item
            for item in landscape.frequencies
            if item.dimension == "method_family"
            and item.value == expected_family
        )

        self.assertEqual(frequency.count, 2)
        self.assertEqual(frequency.paper_ids, ["A", "B"])

    # ------------------------------------------------------------------
    # Combinations
    # ------------------------------------------------------------------

    def test_combination_counts_are_exact(self):
        records = [
            paper(
                "A",
                objective="Task Omega",
                method="Method Alpha",
                constraints=["Constraint One"],
            ),
            paper(
                "B",
                objective="Task Omega",
                method="Method Alpha",
                constraints=["Constraint One"],
            ),
            paper(
                "C",
                objective="Task Omega",
                method="Method Alpha",
                constraints=["Constraint Two"],
            ),
        ]

        features = to_paper_features(records)
        expected_method = features[0].method_families[0]
        expected_constraint = features[0].constraints[0]

        landscape = LandscapeAnalyzer().analyze(records)

        combination = next(
            item
            for item in landscape.combinations
            if item.dimensions == {
                "method_family": expected_method,
                "constraint": expected_constraint,
            }
        )

        self.assertEqual(combination.count, 2)
        self.assertEqual(combination.prevalence, 2 / 3)
        self.assertEqual(combination.paper_ids, ["A", "B"])

    def test_combinations_only_use_controlled_structured_dimensions(self):
        records = [
            paper(
                "A",
                objective="Task Omega",
                method="Method Alpha",
                dataset="Dataset Gamma",
                population="Population Delta",
                baseline="Method Beta",
                constraints=["Constraint One"],
                limitation="Limitation One",
                future_work="Future Direction One",
                finding="Method Alpha outperformed Method Beta",
            )
        ]

        landscape = LandscapeAnalyzer().analyze(records)

        allowed = {
            "problem",
            "population_or_setting",
            "method",
            "method_family",
            "dataset",
            "dataset_type",
            "baseline",
            "constraint",
        }

        for combination in landscape.combinations:
            self.assertTrue(set(combination.dimensions) <= allowed)
            self.assertNotIn("limitation", combination.dimensions)
            self.assertNotIn("future_work", combination.dimensions)
            self.assertNotIn("outcome", combination.dimensions)

    def test_sentinel_values_do_not_enter_combinations(self):
        records = [
            paper(
                "A",
                objective="Task Omega",
                method="other",
                constraints=["Constraint One"],
            ),
            paper(
                "B",
                objective="Task Omega",
                method="Method Alpha",
                constraints=["unknown"],
            ),
            paper(
                "C",
                objective="Task Omega",
                method="Method Alpha",
                constraints=["Constraint One"],
            ),
        ]

        landscape = LandscapeAnalyzer().analyze(records)

        for combination in landscape.combinations:
            values = {
                value.casefold()
                for value in combination.dimensions.values()
            }

            self.assertNotIn("other", values)
            self.assertNotIn("unknown", values)

        features = to_paper_features([records[2]])
        expected_method = features[0].method_families[0]
        expected_constraint = features[0].constraints[0]

        self.assertTrue(
            any(
                item.dimensions == {
                    "method_family": expected_method,
                    "constraint": expected_constraint,
                }
                for item in landscape.combinations
            )
        )

    def test_review_papers_do_not_create_empirical_combinations(self):
        review = paper(
            "R",
            objective="Task Omega",
            method="Method Alpha",
            dataset="Dataset Gamma",
            constraints=["Constraint One"],
            study_type="review",
        )

        landscape = LandscapeAnalyzer().analyze([review])

        self.assertEqual(landscape.combinations, [])

    # ------------------------------------------------------------------
    # Missing evidence
    # ------------------------------------------------------------------

    def test_missing_evidence_is_counted_without_dropping_papers(self):
        records = [
            paper("A", objective="Task Omega", dataset="Dataset Gamma"),
            paper("B", objective="Task Omega"),
            paper("C"),
        ]

        landscape = LandscapeAnalyzer().analyze(records)

        self.assertEqual(landscape.total_papers, 3)
        self.assertEqual(landscape.missing_field_counts["datasets"], 2)
        self.assertEqual(landscape.missing_field_counts["research_objective"], 1)
        self.assertEqual(landscape.missing_field_counts["evaluation_metrics"], 3)

    # ------------------------------------------------------------------
    # Conflicting findings
    # ------------------------------------------------------------------

    def test_conflict_requires_comparable_structured_context(self):
        comparable = [
            paper(
                "A",
                objective="Task Omega",
                method="Method Alpha",
                dataset="Dataset Gamma",
                metric="accuracy",
                finding="Method Alpha outperformed Method Beta",
            ),
            paper(
                "B",
                objective="Task Omega",
                method="Method Alpha",
                dataset="Dataset Gamma",
                metric="accuracy",
                finding="Method Alpha underperformed Method Beta",
            ),
        ]

        landscape = LandscapeAnalyzer().analyze(comparable)

        self.assertEqual(len(landscape.conflicts), 1)
        self.assertEqual(
            landscape.conflicts[0].status,
            "comparable_conflict",
        )
        self.assertEqual(
            landscape.conflicts[0].paper_ids,
            ["A", "B"],
        )

    def test_incomparable_findings_do_not_create_conflict(self):
        records = [
            paper(
                "A",
                objective="Task Omega",
                method="Method Alpha",
                dataset="Dataset Gamma",
                metric="accuracy",
                finding="Method Alpha outperformed Method Beta",
            ),
            paper(
                "B",
                objective="Different Task",
                method="Method Zeta",
                dataset="Dataset Delta",
                metric="latency",
                finding="Method Zeta underperformed Method Eta",
            ),
        ]

        landscape = LandscapeAnalyzer().analyze(records)

        self.assertEqual(landscape.conflicts, [])

    def test_generic_higher_and_lower_words_do_not_create_fake_conflict(self):
        records = [
            paper(
                "A",
                objective="Task Omega",
                method="Method Alpha",
                metric="accuracy",
                finding="The experiment used a higher threshold",
            ),
            paper(
                "B",
                objective="Task Omega",
                method="Method Alpha",
                metric="accuracy",
                finding="The experiment used a lower threshold",
            ),
        ]

        landscape = LandscapeAnalyzer().analyze(records)

        self.assertEqual(landscape.conflicts, [])

    # ------------------------------------------------------------------
    # Determinism / reporting
    # ------------------------------------------------------------------

    def test_ordering_is_deterministic(self):
        records = [
            paper("B", objective="Task Omega", method="Method Beta"),
            paper("A", objective="Task Omega", method="Method Alpha"),
        ]

        first = LandscapeAnalyzer().analyze(records)
        second = LandscapeAnalyzer().analyze(records)

        self.assertEqual(
            first.model_dump(),
            second.model_dump(),
        )

    def test_empty_input_is_safe_and_reportable(self):
        landscape = LandscapeAnalyzer().analyze([])

        self.assertEqual(landscape.total_papers, 0)
        self.assertEqual(landscape.frequencies, [])
        self.assertEqual(landscape.combinations, [])
        self.assertEqual(landscape.conflicts, [])
        self.assertTrue(
            all(
                value == 0
                for value in landscape.missing_field_counts.values()
            )
        )

        report = format_landscape(landscape)

        self.assertIn(
            "Evidence papers analyzed: 0",
            report,
        )


if __name__ == "__main__":
    unittest.main()