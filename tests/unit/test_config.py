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

    def test_environment_overrides_are_validated(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENALEX_CANDIDATE_LIMIT": "35",
                "RESEARCH_GAP_LEXICAL_WEIGHT": "0.25",
                "RESEARCH_GAP_SEMANTIC_WEIGHT": "0.75",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.openalex.per_route_limit, 35)
        self.assertEqual(settings.ranking.semantic_weight, 0.75)

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


if __name__ == "__main__":
    unittest.main()
