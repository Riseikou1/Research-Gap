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
        self.assertFalse(settings.web.billing_enabled)
        self.assertIsNone(settings.web.stripe_secret_key)

    def test_environment_overrides_are_validated(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENALEX_CANDIDATE_LIMIT": "35",
                "RESEARCH_GAP_LEXICAL_WEIGHT": "0.25",
                "RESEARCH_GAP_SEMANTIC_WEIGHT": "0.75",
                "RESEARCH_GAP_EXTRACTION_WORKERS": "3",
                "RESEARCH_GAP_DATABASE_PATH": "/tmp/research-gap-test.sqlite3",
                "DATABASE_URL": "postgresql://example.invalid/research_gap",
                "RESEARCH_GAP_MAX_ANALYSIS_WORKERS": "3",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.openalex.per_route_limit, 35)
        self.assertEqual(settings.ranking.semantic_weight, 0.75)
        self.assertEqual(settings.extraction_workers, 3)
        self.assertEqual(settings.analysis_database_path.name, "research-gap-test.sqlite3")
        self.assertEqual(settings.database_url, "postgresql://example.invalid/research_gap")
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

    def test_enabled_provider_budget_requires_explicit_pricing(self) -> None:
        with patch.dict(
            os.environ,
            {"RESEARCH_GAP_DAILY_PROVIDER_BUDGET_USD": "5"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "explicit positive"):
                Settings.from_env()

    def test_billing_enabled_requires_every_stripe_setting(self) -> None:
        stripe_values = {
            "STRIPE_SECRET_KEY": "sk_test_configured",
            "STRIPE_WEBHOOK_SECRET": "whsec_configured",
            "STRIPE_PRICE_ID": "price_configured",
        }
        for missing_name in stripe_values:
            environment = {"RESEARCH_GAP_BILLING_ENABLED": "true", **stripe_values}
            del environment[missing_name]
            with self.subTest(missing_name=missing_name), patch.dict(
                os.environ, environment, clear=True,
            ):
                with self.assertRaisesRegex(ConfigurationError, missing_name):
                    Settings.from_env()

        with patch.dict(
            os.environ,
            {"RESEARCH_GAP_BILLING_ENABLED": "true", **stripe_values},
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertTrue(settings.web.billing_enabled)

    def test_production_configuration_does_not_require_stripe_when_billing_is_disabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RESEARCH_GAP_APP_URL": "https://research-gap.example",
                "RESEARCH_GAP_ALLOWED_ORIGINS": "https://research-gap.example",
                "RESEARCH_GAP_SECURE_COOKIES": "true",
                "RESEARCH_GAP_GUEST_COOKIE_SECRET": "g" * 32,
                "RESEARCH_GAP_LIFETIME_CREDIT_HMAC_SECRET": "h" * 32,
                "DATABASE_URL": "postgresql://example.invalid/research_gap",
                "OPENAI_API_KEY": "configured",
                "SUPABASE_URL": "https://supabase.example",
                "SUPABASE_ANON_KEY": "configured",
                "SUPABASE_SERVICE_ROLE_KEY": "configured",
                "SUPABASE_JWT_ISSUER": "https://supabase.example/auth/v1",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertFalse(settings.web.billing_enabled)


if __name__ == "__main__":
    unittest.main()
