import os
import unittest
from unittest.mock import patch

from src.config import ConfigurationError, Settings


class ConfigurationTest(unittest.TestCase):
    def test_defaults_are_typed_and_sensible(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.openalex.max_candidates, 100)
        self.assertEqual(settings.ranking.lexical_weight, 0.4)
        self.assertEqual(settings.ranking.semantic_weight, 0.6)
        self.assertEqual(settings.ranking.semantic_fallback, "lexical")
        self.assertEqual(settings.extraction_workers, 4)
        self.assertEqual(settings.cache_directory.name, "cache")
        self.assertEqual(settings.analysis_database_path.name, "research_gap.sqlite3")
        self.assertEqual(settings.max_analysis_workers, 2)

    def test_environment_overrides_are_validated(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENALEX_CANDIDATE_LIMIT": "35",
                "RESEARCH_GAP_LEXICAL_WEIGHT": "0.25",
                "RESEARCH_GAP_SEMANTIC_WEIGHT": "0.75",
                "RESEARCH_GAP_EXTRACTION_WORKERS": "3",
                "RESEARCH_GAP_DATABASE_PATH": "/tmp/research-gap-test.sqlite3",
                "RESEARCH_GAP_MAX_ANALYSIS_WORKERS": "3",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.openalex.per_route_limit, 35)
        self.assertEqual(settings.ranking.semantic_weight, 0.75)
        self.assertEqual(settings.extraction_workers, 3)
        self.assertEqual(settings.analysis_database_path.name, "research-gap-test.sqlite3")
        self.assertEqual(settings.max_analysis_workers, 3)

    def test_invalid_values_fail_clearly(self) -> None:
        with patch.dict(
            os.environ, {"OPENALEX_MAX_RETRIES": "many"}, clear=True
        ):
            with self.assertRaisesRegex(ConfigurationError, "integer"):
                Settings.from_env()

    def test_weights_cannot_both_be_zero(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RESEARCH_GAP_LEXICAL_WEIGHT": "0",
                "RESEARCH_GAP_SEMANTIC_WEIGHT": "0",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "both be zero"):
                Settings.from_env()

    def test_extraction_workers_must_be_positive(self) -> None:
        with patch.dict(
            os.environ,
            {"RESEARCH_GAP_EXTRACTION_WORKERS": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "positive"):
                Settings.from_env()

    def test_runtime_bounds_and_fallback_are_validated(self) -> None:
        invalid = {
            "OPENALEX_CANDIDATE_LIMIT": "101",
            "RESEARCH_GAP_RETRIEVAL_WORKERS": "17",
            "OPENALEX_TIMEOUT_SECONDS": "inf",
            "RESEARCH_GAP_LEXICAL_WEIGHT": "-1",
            "RESEARCH_GAP_SEMANTIC_FALLBACK": "silent",
            "RESEARCH_GAP_MAX_ANALYSIS_WORKERS": "9",
        }
        for name, value in invalid.items():
            with self.subTest(name=name), patch.dict(os.environ, {name: value}, clear=True):
                with self.assertRaises(ConfigurationError):
                    Settings.from_env()


if __name__ == "__main__":
    unittest.main()
