"""Safe bounded acquisition, parsing, and caching of open-access full text."""

from __future__ import annotations

import io
import ipaddress
import json
import logging
import re
import socket
import sqlite3
import time
from threading import RLock
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from defusedxml import ElementTree

from bs4 import BeautifulSoup

from src.models.paper import FullTextLocation, Paper

from .document import PaperDocument, PaperSection, SourceFormat, normalize_heading


LOGGER = logging.getLogger(__name__)
PARSER_VERSION = "full-text-v1"
ACCEPTED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/jats+xml": "xml",
    "text/html": "html",
    "application/xhtml+xml": "html",
}
_REFERENCE_HEADING = re.compile(r"^(?:references|bibliography|works cited)$", re.I)


class FullTextSafetyError(ValueError):
    """Raised before requesting a URL that is not a safe public web target."""


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validator: "SafeURLValidator", max_redirects: int) -> None:
        self.validator = validator
        self.max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        count = int(getattr(req, "_research_gap_redirects", 0)) + 1
        if count > self.max_redirects:
            raise FullTextSafetyError("too many redirects")
        target = urljoin(req.full_url, newurl)
        self.validator.validate(target)
        redirected = super().redirect_request(req, fp, code, msg, headers, target)
        if redirected is not None:
            redirected._research_gap_redirects = count
        return redirected


class SafeURLValidator:
    """Reject non-web and non-public full-text targets before each request."""

    def __init__(self, resolver: Callable[..., Iterable[tuple]] = socket.getaddrinfo) -> None:
        self.resolver = resolver

    def validate(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            raise FullTextSafetyError("full-text URL must use HTTP or HTTPS")
        if not parsed.hostname or parsed.username or parsed.password:
            raise FullTextSafetyError("full-text URL has an invalid host or credentials")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addresses = {
                item[4][0].split("%", 1)[0]
                for item in self.resolver(parsed.hostname, port, type=socket.SOCK_STREAM)
            }
        except (OSError, ValueError) as exc:
            raise FullTextSafetyError(f"full-text host could not be resolved: {exc}") from exc
        if not addresses:
            raise FullTextSafetyError("full-text host resolved to no addresses")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise FullTextSafetyError("full-text host resolves to a non-public address")


class FullTextStore:
    def __init__(self, path: str | Path | None) -> None:
        self.connection: sqlite3.Connection | None = None
        self.lock = RLock()
        if path is None:
            return
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(cache_path, timeout=30, check_same_thread=False)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS full_text_cache (
            paper_id TEXT NOT NULL, source_url TEXT NOT NULL, parser_version TEXT NOT NULL,
            stored_at REAL NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (paper_id, source_url, parser_version))"""
        )
        self.connection.commit()

    def get(self, paper_id: str, source_url: str, *, negative_ttl: float) -> PaperDocument | None:
        if self.connection is None:
            return None
        with self.lock:
            row = self.connection.execute(
                "SELECT stored_at, payload FROM full_text_cache WHERE paper_id=? AND source_url=? AND parser_version=?",
                (paper_id, source_url, PARSER_VERSION),
            ).fetchone()
        if row is None:
            return None
        try:
            document = PaperDocument.model_validate(json.loads(row[1]))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        if document.status != "usable" and time.time() - float(row[0]) > negative_ttl:
            return None
        return document

    def put(self, document: PaperDocument) -> None:
        if self.connection is None or not document.source_url:
            return
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO full_text_cache VALUES (?, ?, ?, ?, ?)",
                (document.paper_id, document.source_url, PARSER_VERSION, time.time(),
                 json.dumps(document.model_dump(mode="json"), ensure_ascii=False)),
            )
            self.connection.commit()


class FullTextClient:
    def __init__(
        self, *, timeout: float = 12.0, max_bytes: int = 8_000_000,
        max_document_chars: int = 120_000, max_section_chars: int = 20_000,
        max_chunk_chars: int = 8_000, max_redirects: int = 3,
        validator: SafeURLValidator | None = None, opener=None,
        cache_path: str | Path | None = None, negative_ttl_seconds: float = 3600,
    ) -> None:
        if min(
            timeout, max_bytes, max_document_chars, max_section_chars,
            max_chunk_chars, negative_ttl_seconds,
        ) <= 0:
            raise ValueError("full-text bounds must be positive")
        if max_redirects < 0 or max_redirects > 10:
            raise ValueError("max_redirects must be between 0 and 10")
        self.timeout, self.max_bytes = timeout, max_bytes
        self.max_document_chars, self.max_section_chars = max_document_chars, max_section_chars
        self.max_chunk_chars, self.negative_ttl_seconds = max_chunk_chars, negative_ttl_seconds
        self.validator = validator or SafeURLValidator()
        self.opener = opener or build_opener(_SafeRedirectHandler(self.validator, max_redirects)).open
        self.store = FullTextStore(cache_path)

    def load(self, paper: Paper) -> PaperDocument:
        if not paper.full_text_locations:
            return PaperDocument(paper_id=paper.id, status="unavailable", notices=["no open-access full-text location"])
        failures: list[PaperDocument] = []
        for location in paper.full_text_locations:
            try:
                cached = self.store.get(paper.id, location.url, negative_ttl=self.negative_ttl_seconds)
            except Exception as exc:
                LOGGER.info("full-text cache read failed paper=%s error=%s", paper.id, exc)
                cached = None
            if cached is not None:
                if cached.status == "usable":
                    return cached
                failures.append(cached)
                continue
            document = self._load_location(paper.id, location)
            try:
                self.store.put(document)
            except Exception as exc:
                LOGGER.info("full-text cache write failed paper=%s error=%s", paper.id, exc)
            if document.status == "usable":
                return document
            failures.append(document)
        priority = {"parse_failed": 3, "fetch_failed": 2, "unavailable": 1}
        return max(failures, key=lambda item: priority.get(item.status, 0)) if failures else PaperDocument(paper_id=paper.id, status="unavailable")

    def _load_location(self, paper_id: str, location: FullTextLocation) -> PaperDocument:
        try:
            self.validator.validate(location.url)
            request = Request(location.url, headers={
                "User-Agent": "research-gap/0.9",
                "Accept": "application/pdf, application/xml, text/xml, text/html",
            })
            with self.opener(request, timeout=self.timeout) as response:
                final_url = getattr(response, "geturl", lambda: location.url)()
                self.validator.validate(final_url)
                content_type = response.headers.get_content_type().casefold()
                source_format = ACCEPTED_CONTENT_TYPES.get(content_type)
                if (
                    source_format is None
                    and content_type == "application/octet-stream"
                    and location.source_format == "pdf"
                ):
                    source_format = "pdf"
                if source_format is None:
                    return self._failure(paper_id, final_url, "unavailable", location.source_format,
                                         f"unsupported content type: {content_type}")
                data = self._read_bounded(response)
            document = parse_document(
                paper_id, final_url, source_format, data,
                max_document_chars=self.max_document_chars,
                max_section_chars=self.max_section_chars,
                max_chunk_chars=self.max_chunk_chars,
            )
            return document
        except (FullTextSafetyError, HTTPError, URLError, TimeoutError, OSError) as exc:
            LOGGER.info("full-text fetch failed paper=%s url=%s error=%s", paper_id, location.url, exc)
            return self._failure(paper_id, location.url, "fetch_failed", location.source_format, str(exc))
        except Exception as exc:
            LOGGER.info("full-text parse failed paper=%s url=%s error=%s", paper_id, location.url, exc)
            return self._failure(paper_id, location.url, "parse_failed", location.source_format, str(exc))

    def _read_bounded(self, response: BinaryIO) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared:
            try:
                if int(declared) > self.max_bytes:
                    raise OSError("full-text response exceeds maximum bytes")
            except ValueError as exc:
                raise OSError("full-text response has an invalid content length") from exc
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(65536, self.max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > self.max_bytes:
                raise OSError("full-text response exceeds maximum bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _failure(paper_id: str, url: str, status: str, source_format: str, notice: str) -> PaperDocument:
        return PaperDocument(paper_id=paper_id, source_url=url, source_format=source_format,
                             status=status, notices=[notice[:300]])


def parse_document(
    paper_id: str, source_url: str, source_format: SourceFormat, data: bytes, *,
    max_document_chars: int, max_section_chars: int, max_chunk_chars: int,
) -> PaperDocument:
    if source_format == "pdf":
        raw_sections = _parse_pdf(data)
    elif source_format == "xml":
        raw_sections = _parse_xml(data)
    elif source_format == "html":
        raw_sections = _parse_html(data)
    else:
        raise ValueError("unsupported full-text format")
    return _normalize_document(
        paper_id, source_url, source_format, raw_sections,
        max_document_chars=max_document_chars, max_section_chars=max_section_chars,
        max_chunk_chars=max_chunk_chars,
    )


def _parse_pdf(data: bytes) -> list[tuple[str, str]]:
    if not data.startswith(b"%PDF"):
        raise ValueError("malformed PDF")
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data), strict=False)
    pages = [(f"Page {index}", page.extract_text() or "") for index, page in enumerate(reader.pages, 1)]
    text = "\n".join(value for _, value in pages).strip()
    if len(text) < 80 or len(re.sub(r"\W", "", text)) < 40:
        raise ValueError("PDF has no usable text layer")
    return pages


def _parse_xml(data: bytes) -> list[tuple[str, str]]:
    root = ElementTree.fromstring(data)
    sections: list[tuple[str, str]] = []
    abstract = next((node for node in root.iter() if _local_name(node.tag) == "abstract"), None)
    if abstract is not None:
        sections.append(("Abstract", " ".join(abstract.itertext())))
    for section in (node for node in root.iter() if _local_name(node.tag) == "sec"):
        title = next((node for node in section if _local_name(node.tag) == "title"), None)
        heading = " ".join(title.itertext()) if title is not None else "Section"
        if _REFERENCE_HEADING.fullmatch(" ".join(heading.split())):
            continue
        paragraphs = [
            " ".join(node.itertext())
            for node in section
            if _local_name(node.tag) == "p"
        ]
        if paragraphs:
            sections.append((heading, "\n".join(paragraphs)))
    if not sections:
        article = next(
            (node for node in root.iter() if _local_name(node.tag) in {"article-body", "body"}),
            None,
        )
        if article is not None:
            sections.append(("Full text", " ".join(article.itertext())))
    return sections


def _local_name(tag: object) -> str:
    return tag.rsplit("}", 1)[-1].casefold() if isinstance(tag, str) else ""


def _parse_html(data: bytes) -> list[tuple[str, str]]:
    soup = BeautifulSoup(data, "html.parser")
    for node in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        node.decompose()
    container = soup.find("article") or soup.find("main")
    if container is None:
        raise ValueError("HTML is a generic landing page, not a structured article")
    sections: list[tuple[str, str]] = []
    heading = "Full text"
    paragraphs: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if node.name.startswith("h"):
            if paragraphs:
                if not _REFERENCE_HEADING.fullmatch(heading):
                    sections.append((heading, "\n".join(paragraphs)))
            heading, paragraphs = text, []
        else:
            paragraphs.append(text)
    if paragraphs:
        if not _REFERENCE_HEADING.fullmatch(heading):
            sections.append((heading, "\n".join(paragraphs)))
    return sections


def _normalize_document(
    paper_id: str, source_url: str, source_format: SourceFormat,
    raw_sections: list[tuple[str, str]], *, max_document_chars: int,
    max_section_chars: int, max_chunk_chars: int,
) -> PaperDocument:
    cleaned = [(" ".join(h.split()) or "Section", re.sub(r"\s+", " ", t).strip())
               for h, t in raw_sections if re.sub(r"\s+", " ", t).strip()]
    if not cleaned:
        raise ValueError("document contains no article text")
    structure = any(not heading.casefold().startswith(("page ", "full text", "section")) for heading, _ in cleaned)
    sections: list[PaperSection] = []
    total = 0
    truncated = False
    for heading, original_text in cleaned:
        if total >= max_document_chars:
            truncated = True
            break
        text = original_text[: min(max_section_chars, max_document_chars - total)]
        section_truncated = len(text) < len(original_text)
        if not structure:
            for offset in range(0, len(text), max_chunk_chars):
                chunk = text[offset:offset + max_chunk_chars]
                sections.append(PaperSection(id=f"chunk-{len(sections)+1:04d}", heading=heading,
                                             section_types=["other"], text=chunk,
                                             truncated=offset + max_chunk_chars < len(text) or section_truncated))
        else:
            sections.append(PaperSection(id=f"section-{len(sections)+1:04d}", heading=heading,
                                         section_types=normalize_heading(heading), text=text,
                                         truncated=section_truncated))
        total += len(text)
        truncated = truncated or section_truncated
    all_text = "".join(section.text for section in sections)
    if len(all_text) < 80 or len(re.sub(r"\W", "", all_text)) < 40:
        raise ValueError("document text is empty or garbled")
    notices = ["normalized full text was truncated"] if truncated else []
    if not structure:
        notices.append("structural headings unavailable; deterministic chunks used")
    return PaperDocument(paper_id=paper_id, source_url=source_url, source_format=source_format,
                         sections=sections, status="usable", structure_available=structure,
                         truncated=truncated, notices=notices)


__all__ = ["FullTextClient", "FullTextSafetyError", "PARSER_VERSION", "SafeURLValidator", "parse_document"]
