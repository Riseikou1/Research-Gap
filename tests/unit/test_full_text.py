from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.analysis.landscape import LandscapeAnalyzer
from src.config import ConfigurationError, Settings
from src.extraction.document import (
    FIELD_SECTION_PREFERENCES,
    PaperDocument,
    PaperSection,
    build_extraction_context,
    normalize_heading,
)
from src.extraction.evidence import EvidenceItem, ExtractionCoverage, PaperEvidence
from src.extraction.full_text import (
    FullTextClient,
    FullTextStore,
    SafeURLValidator,
    _SafeRedirectHandler,
    parse_document,
)
from src.extraction.paper_extractor import PaperExtractor, _ExtractionResult, _to_evidence
from src.models.paper import FullTextLocation, Paper
from src.reporting.landscape import format_landscape
from src.retrieval.openalex import _parse_work
from main import build_parser


def public_resolver(host, port, type=None):
    address = "93.184.216.34" if host != "internal.test" else "127.0.0.1"
    return [(2, 1, 6, "", (address, port))]


class Headers(Message):
    pass


class Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str, *, url="https://example.org/a"):
        super().__init__(body)
        self.headers = Headers()
        self.headers["Content-Type"] = content_type
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Responses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return type("Parsed", (), {"output_parsed": self.payload})()


class OpenAI:
    def __init__(self, payload):
        self.responses = Responses(payload)


class Loader:
    def __init__(self, documents):
        self.documents = documents
        self.paper_ids = []

    def load(self, paper):
        self.paper_ids.append(paper.id)
        return self.documents[paper.id]


class SequenceLoader:
    def __init__(self, documents):
        self.documents = iter(documents)

    def load(self, paper):
        return next(self.documents)


def article_html() -> bytes:
    return (
        "<html><main><h2>Methods</h2><p>We enrolled 120 participants and used Model A "
        "for controlled evaluation in two clinical centers.</p><h2>Results and Discussion</h2>"
        "<p>Model A outperformed the baseline on accuracy while performance remained lower "
        "for the external cohort.</p></main></html>"
    ).encode()


def text_pdf() -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    fonts = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): fonts})
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 11 Tf 50 740 Td (Methods and results from a controlled study with one hundred twenty participants and reliable accuracy measurements.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class FullTextTest(unittest.TestCase):
    def test_full_text_configuration_is_bounded(self):
        with patch.dict(os.environ, {"RESEARCH_GAP_FULL_TEXT_MAX_BYTES": "25000001"}, clear=False):
            with self.assertRaises(ConfigurationError):
                Settings.from_env()

    def test_full_text_cli_is_explicitly_opt_in(self):
        parser = build_parser()
        self.assertFalse(parser.parse_args(["idea"]).full_text)
        self.assertTrue(parser.parse_args(["idea", "--full-text"]).full_text)

    def test_openalex_locations_prefer_best_oa_and_deduplicate(self):
        work = {
            "id": "https://openalex.org/W1", "display_name": "Paper",
            "best_oa_location": {"is_oa": True, "pdf_url": "https://oa.test/a.pdf", "landing_page_url": "https://oa.test/a"},
            "primary_location": {"is_oa": True, "pdf_url": "https://oa.test/a.pdf", "landing_page_url": "https://publisher.test/a"},
            "locations": [
                {"is_oa": False, "pdf_url": "https://closed.test/a.pdf"},
                {"is_oa": True, "landing_page_url": "https://repository.test/a.xml"},
            ],
        }
        urls = [item.url for item in _parse_work(work).full_text_locations]
        self.assertEqual(urls, ["https://oa.test/a.pdf", "https://oa.test/a", "https://publisher.test/a", "https://repository.test/a.xml"])

    def test_safe_url_validation_rejects_private_and_non_http(self):
        validator = SafeURLValidator(public_resolver)
        validator.validate("https://example.org/a.pdf")
        with self.assertRaises(ValueError):
            validator.validate("file:///tmp/a.pdf")
        with self.assertRaises(ValueError):
            validator.validate("http://internal.test/a")

    def test_redirect_count_is_bounded_before_following(self):
        from urllib.request import Request

        handler = _SafeRedirectHandler(SafeURLValidator(public_resolver), 1)
        request = Request("https://example.org/a")
        request._research_gap_redirects = 1
        with self.assertRaises(ValueError):
            handler.redirect_request(request, None, 302, "Found", Headers(), "https://example.org/b")

    def test_html_xml_and_pdf_are_parsed(self):
        html = parse_document("p", "https://example.org/a", "html", article_html(),
                              max_document_chars=5000, max_section_chars=2000, max_chunk_chars=500)
        self.assertEqual(html.status, "usable")
        self.assertIn("methods", html.sections[0].section_types)

        xml = b"<article><abstract><p>A sufficiently detailed abstract for testing structured XML article parsing behavior.</p></abstract><body><sec><title>Limitations and Future Work</title><p>Future studies should validate the method in larger external cohorts with additional measurements.</p></sec></body></article>"
        jats = parse_document("p", "https://example.org/a.xml", "xml", xml,
                              max_document_chars=5000, max_section_chars=2000, max_chunk_chars=500)
        self.assertEqual(jats.sections[1].section_types, ["limitations", "future_work"])

        pdf = parse_document("p", "https://example.org/a.pdf", "pdf", text_pdf(),
                             max_document_chars=5000, max_section_chars=2000, max_chunk_chars=60)
        self.assertEqual(pdf.status, "usable")
        self.assertFalse(pdf.structure_available)
        self.assertGreater(len(pdf.sections), 1)

    def test_heading_mapping_and_context_are_bounded_and_deterministic(self):
        self.assertEqual(normalize_heading("Results and Discussion"), ["results", "discussion"])
        self.assertEqual(normalize_heading("Materials and Methods"), ["methods", "materials"])
        self.assertEqual(FIELD_SECTION_PREFERENCES["limitations"][:2], ("limitations", "discussion"))
        document = PaperDocument(
            paper_id="p", source_url="https://example.org/a", source_format="html", status="usable",
            structure_available=True,
            sections=[
                PaperSection(id="s1", heading="Methods", section_types=["methods"], text="M" * 90),
                PaperSection(id="s2", heading="Results", section_types=["results"], text="R" * 90),
            ],
        )
        first = build_extraction_context(document, max_chars=130)
        second = build_extraction_context(document, max_chars=130)
        self.assertEqual(first, second)
        self.assertTrue(first[2])
        self.assertLessEqual(len(first[0]), 130)

    def test_fetch_failures_bad_types_oversize_and_unsafe_redirect_fall_back(self):
        validator = SafeURLValidator(public_resolver)
        paper = Paper(id="p", title="Paper", abstract="Abstract evidence.", full_text_locations=[FullTextLocation(url="https://example.org/a")])
        cases = [
            (lambda req, timeout: (_ for _ in ()).throw(TimeoutError("timed out")), "fetch_failed"),
            (lambda req, timeout: Response(b"landing", "application/json"), "unavailable"),
            (lambda req, timeout: Response(b"x" * 30, "text/html"), "fetch_failed"),
            (lambda req, timeout: Response(article_html(), "text/html", url="http://internal.test/a"), "fetch_failed"),
        ]
        for opener, expected in cases:
            with self.subTest(expected=expected):
                client = FullTextClient(validator=validator, opener=opener, max_bytes=20)
                self.assertEqual(client.load(paper).status, expected)

    def test_generic_landing_and_malformed_documents_fail_safely(self):
        validator = SafeURLValidator(public_resolver)
        for body, content_type in [(b"<html><body>download</body></html>", "text/html"), (b"not-pdf", "application/pdf")]:
            client = FullTextClient(validator=validator, opener=lambda req, timeout, b=body, c=content_type: Response(b, c))
            paper = Paper(id="p", title="Paper", full_text_locations=[FullTextLocation(url="https://example.org/a")])
            self.assertEqual(client.load(paper).status, "parse_failed")

    def test_full_text_provenance_is_required_and_validated_against_section(self):
        with self.assertRaises(ValidationError):
            EvidenceItem(value="x", evidence_text="x", source="full_text", confidence=0.8)

        section = PaperSection(id="section-0001", heading="Results", section_types=["results"], text="The model improved accuracy by five points.")
        document = PaperDocument(paper_id="p", source_url="https://example.org/a", source_format="html",
                                 sections=[section], status="usable", structure_available=True)
        claim = EvidenceItem(value="accuracy improvement", evidence_text="improved accuracy by five points",
                             source="full_text", confidence=0.9, section_type="results",
                             section_heading="Results", section_id="section-0001")
        payload = _ExtractionResult(main_findings=[claim], extraction_confidence=0.9)
        loader = Loader({"p": document})
        extractor = PaperExtractor(client=OpenAI(payload), full_text_client=loader, evidence_limit=1)
        result = extractor.extract_many([Paper(id="p", title="Paper")])[0]
        self.assertEqual(result.main_findings, [claim])
        self.assertEqual(result.coverage.source_level, "full_text")

        bad = claim.model_copy(update={"section_id": "wrong"})
        rejected = _to_evidence(
            Paper(id="p", title="Paper"),
            _ExtractionResult(main_findings=[bad], extraction_confidence=0.9),
            document=document,
            inspected_sections=[section],
        )
        self.assertEqual(rejected.main_findings, [])

    def test_detailed_fields_require_controlled_full_text_passages(self):
        fixture = json.loads(
            (Path(__file__).parents[1] / "fixtures" / "controlled_full_text_evidence.json")
            .read_text(encoding="utf-8")
        )
        paper = Paper.model_validate(fixture["paper"])
        document = PaperDocument.model_validate(fixture["document"])

        class FixtureResponses:
            def __init__(self):
                self.calls = 0

            def parse(self, **kwargs):
                self.calls += 1
                return type("Parsed", (), {
                    "output_parsed": kwargs["text_format"].model_validate(fixture["extraction"]),
                })()

        full_text_responses = FixtureResponses()
        full_text = PaperExtractor(
            client=type("Client", (), {"responses": full_text_responses})(),
            full_text_client=Loader({paper.id: document}),
        ).extract(paper)

        self.assertEqual([item.value for item in full_text.datasets], ["EnterpriseFlow-500"])
        self.assertEqual(full_text.sample_size.value, "500 examples")
        self.assertEqual(
            [item.value for item in full_text.comparison_or_baseline],
            ["GPT-4 zero-shot baseline"],
        )
        self.assertEqual(
            [item.value for item in full_text.evaluation_metrics],
            ["Exact Match Accuracy", "Hallucination Rate"],
        )
        self.assertTrue(all(
            item.source == "full_text"
            for item in [
                *full_text.datasets,
                full_text.sample_size,
                *full_text.comparison_or_baseline,
                *full_text.evaluation_metrics,
            ]
        ))

        abstract_only_responses = FixtureResponses()
        abstract_only = PaperExtractor(
            client=type("Client", (), {"responses": abstract_only_responses})(),
        ).extract(paper)
        self.assertEqual(abstract_only.datasets, [])
        self.assertIsNone(abstract_only.sample_size)
        self.assertEqual(abstract_only.comparison_or_baseline, [])
        self.assertEqual(abstract_only.evaluation_metrics, [])

    def test_unavailable_fetch_and_parse_statuses_use_abstract_fallback(self):
        abstract = "This study evaluates Model A on a clinical cohort."
        claim = EvidenceItem(value="Model A", evidence_text="Model A", source="abstract", confidence=0.9)
        for status in ("unavailable", "fetch_failed", "parse_failed"):
            loader = Loader({"p": PaperDocument(paper_id="p", status=status, notices=[status])})
            payload = _ExtractionResult(method_or_intervention=[{
                **claim.model_dump(), "role": "primary"
            }], extraction_confidence=0.8)
            result = PaperExtractor(client=OpenAI(payload), full_text_client=loader).extract(
                Paper(id="p", title="Paper", abstract=abstract)
            )
            self.assertEqual(result.coverage.source_level, "abstract")
            self.assertEqual(result.coverage.full_text_status, status)
            self.assertEqual(result.method_or_intervention[0].value, "Model A")

    def test_changed_acquisition_outcome_invalidates_fallback_evidence_cache(self):
        section = PaperSection(id="s1", heading="Methods", section_types=["methods"], text="A sufficiently detailed methods section describes Model A and its controlled evaluation protocol.")
        loader = SequenceLoader([
            PaperDocument(paper_id="p", status="unavailable"),
            PaperDocument(paper_id="p", source_url="https://example.org/a", source_format="html",
                          status="usable", structure_available=True, sections=[section]),
        ])
        payload = _ExtractionResult(extraction_confidence=0.8)
        client = OpenAI(payload)
        extractor = PaperExtractor(client=client, full_text_client=loader)
        paper = Paper(id="p", title="Paper", abstract="An abstract is available.")
        self.assertEqual(extractor.extract(paper).coverage.full_text_status, "unavailable")
        self.assertEqual(extractor.extract(paper).coverage.full_text_status, "usable")
        self.assertEqual(len(client.responses.calls), 2)

    def test_enrichment_only_loads_existing_evidence_limit_subset(self):
        papers = [Paper(id=str(i), title=f"Paper {i}") for i in range(5)]
        docs = {paper.id: PaperDocument(paper_id=paper.id, status="unavailable") for paper in papers}
        loader = Loader(docs)
        extractor = PaperExtractor(client=OpenAI(_ExtractionResult(extraction_confidence=0.5)),
                                   full_text_client=loader, evidence_limit=2)
        self.assertEqual(len(extractor.extract_many(papers)), 2)
        self.assertEqual(loader.paper_ids, ["0", "1"])

    def test_coverage_reporting_distinguishes_source_levels(self):
        records = []
        for index, (level, status, truncated) in enumerate([
            ("full_text", "usable", True), ("abstract", "unavailable", False),
            ("metadata_only", "not_attempted", False),
        ]):
            records.append(PaperEvidence(
                paper_id=str(index), title="Paper", study_type="other", extraction_confidence=0.5,
                coverage=ExtractionCoverage(source_level=level, full_text_status=status, truncated=truncated),
            ))
        landscape = LandscapeAnalyzer().analyze(records)
        report = format_landscape(landscape)
        self.assertIn("1 full-text, 1 abstract-only, 1 metadata-only", report)
        self.assertIn("Truncated full-text documents: 1", report)

    def test_full_text_cache_uses_parser_version(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FullTextStore(Path(directory) / "cache.sqlite3")
            document = PaperDocument(paper_id="p", source_url="https://example.org/a", source_format="html",
                                     status="usable", sections=[PaperSection(id="s", heading="Methods", section_types=["methods"], text="usable text")])
            store.put(document)
            self.assertIsNotNone(store.get("p", document.source_url, negative_ttl=10))
            with patch("src.extraction.full_text.PARSER_VERSION", "full-text-v-next"):
                self.assertIsNone(store.get("p", document.source_url, negative_ttl=10))

    def test_full_text_client_reuses_cached_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def opener(request, timeout):
                calls.append(request.full_url)
                return Response(article_html(), "text/html")

            client = FullTextClient(
                validator=SafeURLValidator(public_resolver), opener=opener,
                cache_path=Path(directory) / "cache.sqlite3",
            )
            paper = Paper(id="p", title="Paper", full_text_locations=[FullTextLocation(url="https://example.org/a")])
            self.assertEqual(client.load(paper).status, "usable")
            self.assertEqual(client.load(paper).status, "usable")
            self.assertEqual(calls, ["https://example.org/a"])


if __name__ == "__main__":
    unittest.main()
