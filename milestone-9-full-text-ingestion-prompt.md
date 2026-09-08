# Milestone 9 — Full-text ingestion and section-aware evidence extraction

Implement the next Research GAP milestone in the current repository. The system currently extracts `PaperEvidence` mainly from a paper's title and abstract. Add an optional, bounded full-text enrichment path for selected papers while keeping the existing abstract workflow as a reliable fallback.

Do not stop after proposing an architecture. Inspect the repository, implement the feature, run the tests, and report what actually works.

## Repository constraints

Before editing, inspect at least:

- `IMPORTANT.md`
- `src/models/paper.py`
- `src/retrieval/openalex.py`
- `src/extraction/evidence.py`
- `src/extraction/paper_extractor.py`
- `src/extraction/store.py`
- `src/analysis/landscape.py`
- `src/models/landscape.py`
- `src/reporting/landscape.py`
- `src/pipeline.py`
- `src/config.py`
- relevant extraction, landscape, pipeline, configuration, and CLI tests

Follow the repository's current naming, Pydantic, configuration, cache, error-handling, and test conventions. Extend existing abstractions instead of duplicating them.

Do not redesign retrieval, reranking, gap generation, verification, evaluation, the API, or persistence except for the smallest integration changes this milestone requires.

## Required pipeline

The completed flow must be:

```text
retrieve and rank papers cheaply
→ choose the existing evidence-limit subset
→ discover legitimate open-access full-text locations
→ fetch and parse usable full text when available
→ build bounded section-aware extraction context
→ extract evidence with source provenance
→ fall back to the existing title/abstract path per paper
→ build the existing landscape and gap analysis
```

Never download full text for every retrieval candidate. Enrich only papers selected for evidence extraction.

## 1. Open-access location discovery

Preserve the OpenAlex location metadata needed for later full-text resolution. Inspect the actual OpenAlex response shape used by this repository; do not guess field names. Consider the primary location, best open-access location, and other locations when available, deduplicate candidate URLs, and prefer clearly identified open-access full-text URLs.

The retrieval layer may expose provider-independent full-text location candidates on `Paper` or an equivalent small model. Do not perform downloads during initial retrieval or ranking.

Only use legitimate URLs exposed by paper metadata. Do not bypass paywalls, scrape credentials, automate logins, or circumvent publisher access controls.

## 2. Acquisition and parsing boundary

Keep these responsibilities separate:

```text
location resolution → bounded HTTP acquisition → format parser → normalized paper document
```

The evidence extractor must consume a normalized document, not PDF-specific objects.

Add the smallest models needed to represent:

- paper ID
- source URL and source format
- ordered sections
- original section heading
- normalized semantic section type or types
- section text
- acquisition/parse status
- truncation state and concise notices

Combined headings such as `Results and Discussion` and `Limitations and Future Work` must be usable by both relevant extraction routes. Do not force every section into exactly one semantic role if that loses information.

For the first version, support text-based content that can be parsed reliably with small maintained dependencies:

- direct PDFs identified by content type or a trustworthy direct-PDF URL
- structured XML/JATS or full-article HTML when the response clearly contains article text

Skip generic landing pages, scanned PDFs requiring OCR, garbled/empty documents, and unsupported content safely. Do not implement OCR, figure interpretation, table reconstruction, equation interpretation, references, or supplementary-file processing.

Use the existing HTTP stack where practical. Add only necessary dependencies and pin them consistently with `requirements.txt`.

## 3. Network and document safety

Full-text fetching is untrusted network input. Implement and test:

- HTTP/HTTPS only
- rejection of loopback, link-local, private-network, and otherwise unsafe targets, including redirects
- bounded redirects
- request timeout
- maximum response bytes, including streamed responses
- accepted content types
- maximum normalized document characters
- maximum section and chunk characters
- deterministic ordering and truncation
- useful developer logging without flooding normal output

Expose truncation in document/extraction coverage metadata. Never silently treat a truncated document as fully inspected.

## 4. Generic section normalization

Normalize common headings with a small deterministic layer. Support semantic roles including:

```text
abstract, introduction, related_work, methods, materials, dataset,
experimental_setup, results, discussion, limitations, conclusion,
future_work, appendix, other
```

Handle ordinary variants such as `Methodology`, `Materials and Methods`, `Experiments`, `Results and Discussion`, and `Limitations and Future Work`. Keep this generic across scientific domains; do not add paper-specific or domain-specific heading lists. Do not use an LLM merely to classify obvious headings.

When structural headings are unavailable, use deterministic bounded chunks as a fallback and mark that structure was unavailable.

## 5. Section-aware extraction

Do not send an entire paper blindly to the model. Build bounded extraction contexts from the most relevant sections. Use this as guidance rather than an exclusion rule:

| Evidence fields | Preferred sections |
| --- | --- |
| research objective | abstract, introduction |
| population/setting, data/modality | methods, materials, dataset |
| methods/interventions, constraints | methods, experimental setup |
| datasets, sample size | dataset, methods, materials, experimental setup |
| baselines, evaluation metrics | experimental setup, results |
| main findings | results, discussion, abstract |
| limitations | limitations, discussion, conclusion |
| future work | future work, conclusion, discussion |

Use a small bounded number of extraction requests per paper or batch. Group related fields if needed; do not make one request per field. Preserve the current extraction batching and worker limits where possible.

Relevant evidence from a non-preferred section may still be accepted when directly supported. Do not invent, extrapolate, or force values merely because full text was available.

## 6. Evidence provenance and validation

The current `EvidenceSource` only allows `title` and `abstract`. Extend it compatibly for full-text evidence.

Every `EvidenceItem` must retain:

- a concise normalized value
- verbatim `evidence_text`
- source level (`title`, `abstract`, or `full_text`)
- confidence
- optional lightweight full-text provenance such as normalized section type, original heading, and stable section/chunk identifier

Existing callers and fixtures that use title/abstract evidence must continue working without artificial full-text fields.

Update evidence cleaning/validation so a full-text quote is checked against its referenced normalized section or chunk. Do not validate full-text evidence against only the abstract, and do not accept a quote that cannot be found in the supplied source text after the repository's existing harmless normalization rules.

Do not copy whole documents into every evidence item.

## 7. Extraction coverage semantics

An empty extracted field means “the system did not extract evidence,” not automatically “the paper does not report it.” Preserve this distinction explicitly.

Add one compact paper-level coverage object, or the closest clean equivalent, that can represent:

- source level actually used: metadata/title only, abstract, or full text
- full-text status: not attempted, unavailable, fetch failed, parse failed, or usable
- inspected section types
- whether structural parsing was available
- whether content/context was truncated
- concise non-fatal notices

Keep this optional or provide compatibility-safe defaults so old `PaperEvidence` fixtures do not require large rewrites. The new production path must populate it accurately.

Update landscape/reporting semantics so the CLI does not imply that an unextracted field was absent from the paper. Preserve existing `missing_field_counts` if changing it would be disruptive, but label it as extraction missingness and add a compact source-coverage summary such as:

```text
Evidence sources: 4 full-text, 5 abstract-only, 1 metadata-only
Full-text outcomes: 4 usable, 3 unavailable, 1 fetch failure, 2 not attempted
Truncated full-text documents: 1
```

The implementation must let downstream code distinguish abstract-only missing evidence from missing evidence after usable full-text inspection.

## 8. Fallback and failure behavior

Handle each paper independently:

```text
usable full text                 → section-aware full-text extraction
no legitimate full-text URL     → current abstract extraction
fetch or parse failure           → abstract extraction plus recorded notice
no abstract either               → preserve the existing safe metadata-only behavior
```

One bad document must not fail the analysis. Log actionable technical details, but keep routine CLI output concise.

## 9. Configuration and cache

Make full-text enrichment explicitly opt-in for this first version so existing commands retain their present performance and network behavior. Add a clear CLI option such as `--full-text` and thread it through the normal application path. If the API needs a matching request option, add only the smallest backward-compatible field.

Add validated, bounded configuration only where needed, for example fetch timeout, maximum bytes, maximum document characters, and maximum extraction-context characters. Choose conservative defaults and document all new environment variables.

Reuse the existing cache directory/store design. Cache reusable acquisition/parsing results when sensible, including negative results only for a reasonable bounded period. Cache keys must include the paper/source identity and an explicit resolver/parser version so incompatible parsing changes invalidate old entries. Do not create an unrelated database system or commit downloaded papers/cache artifacts.

## 10. Tests

All unit tests must be offline and deterministic. Mock HTTP responses; do not depend on a live publisher or OpenAlex in the unit suite.

Add focused tests for:

- OpenAlex location parsing and URL deduplication
- safe URL validation, redirects, timeouts, oversize streams, bad content types, and malformed content
- usable PDF and structured-document parsing
- empty/garbled/scanned or otherwise unsupported content falling back safely
- heading normalization, including combined headings mapping to multiple roles
- deterministic structural and fallback chunking with visible truncation
- field-to-section context routing
- full-text quote/provenance validation
- abstract fallback after unavailable, fetch-failed, and parse-failed full text
- coverage/reporting differences between full-text, abstract-only, and metadata-only papers
- cache hits and parser-version invalidation
- `--full-text` orchestration only enriching the selected evidence subset
- backward compatibility for current `Paper`, `EvidenceItem`, and `PaperEvidence` fixtures

Run the repository's canonical full test suite after the focused tests. Do not weaken or delete existing tests to make the feature pass.

An optional live smoke test may be added behind an explicit environment flag, but it must not run by default. Never claim a live full-text test passed unless it was actually executed successfully.

## Non-goals

Do not add:

- paywall bypassing or authenticated publisher scraping
- browser automation
- OCR
- publisher-specific parser classes unless absolutely unavoidable
- domain-specific method, metric, or dataset dictionaries
- LLM-based heading classification
- figure/table/equation interpretation
- a second retrieval, evidence, cache, or verification subsystem
- unbounded downloads, context, retries, concurrency, or extraction calls

## Acceptance criteria

The milestone is complete only when all of the following are true:

1. Existing commands work unchanged when full text is not enabled.
2. A `--full-text` run enriches only the selected evidence papers.
3. Each selected paper independently uses usable open-access full text or falls back safely.
4. Full-text evidence includes a verifiable quote and section provenance.
5. Abstract-only missing evidence is distinguishable from missing evidence after full-text inspection.
6. Truncation and parse/fetch failures are visible in coverage metadata.
7. The landscape/gap pipeline consumes the richer `PaperEvidence` without a parallel code path.
8. Focused tests and the complete existing test suite pass.
9. No placeholder, TODO-only, fake-parser, or mocked production implementation remains.

Full text should improve evidence coverage, but it does not guarantee that every paper contains sample size, datasets, metrics, limitations, or future work. Never fabricate those fields to satisfy a demonstration.

## Final response

After implementation, report concisely:

- architecture and fallback behavior
- files created and modified
- new dependencies, CLI flags, and environment variables
- focused and full-suite test commands with actual results
- one exact manual command for a real query
- one real output excerpt if a live run was actually completed; otherwise state why it was not run
- deliberately deferred limitations

