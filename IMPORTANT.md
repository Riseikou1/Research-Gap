# Research GAP — Implementation Contract and Roadmap

> This file is the source of truth for the project. Before changing code, read this file and

> `README.md`. Implement only the requested milestone, preserve existing behavior, and run the

> milestone's acceptance checks before declaring it complete.

## 1. Product definition

Research GAP accepts a proposed research idea and produces an evidence-backed analysis of the

surrounding literature. It must help a researcher answer:

1. What existing work is most relevant to this idea?

2. How has the problem already been studied?

3. Which populations, methods, datasets, domains, and evaluation settings were used?

4. What limitations, conflicts, and untested combinations remain?

5. Is the proposed idea plausibly novel, partially novel, or already well studied?

6. Which papers support every conclusion?

The system must never claim that a research gap exists merely because no paper appeared in the

first search results. A gap claim must be traceable to retrieved papers and qualified by the

coverage and limitations of the search.

## 2. Non-goals for the first local version

- Do not build a website yet.

- Do not train or fine-tune a language model.

- Do not add PostgreSQL, pgvector, Neo4j, Docker, or cloud deployment prematurely.

- Do not let an LLM invent paper metadata, citations, findings, or limitations.

- Do not use an LLM as a replacement for retrieval.

- Do not call the output a systematic review unless the workflow actually follows a systematic

  review protocol.

## 3. Target pipeline

```text
research idea
  -> validate input
  -> decompose idea into structured research facets
  -> generate deterministic baseline queries
  -> optionally generate LLM terminology/concept queries
  -> normalize, deduplicate, validate, and bound the query plan
  -> retrieve candidates from lexical and semantic search strategies
  -> normalize and deduplicate papers while preserving retrieval provenance
  -> rerank papers against the original idea
  -> extract structured evidence from papers
  -> cluster and compare the literature
  -> generate candidate gap hypotheses
  -> verify every hypothesis against evidence
  -> produce a qualified, cited research-gap report
```

The LLM is allowed to improve query formulation, but it is not the retrieval engine and it is never the sole source of search coverage. The original idea and deterministic query baseline must remain available so retrieval behavior is inspectable and evaluable.

## 4. Engineering rules

1. Keep provider-specific code behind interfaces.
2. Every milestone needs unit tests and at least one end-to-end smoke test.
3. Store normalized internal objects; do not leak raw provider response shapes throughout the app.
4. Prefer deterministic code for validation, normalization, merging, deduplication, scoring, and citation handling.
5. Use LLMs only where language understanding materially helps.
6. Require structured output from LLM calls and validate it before use.
7. LLM-generated search queries must supplement, not replace, deterministic and semantic retrieval strategies.
8. Preserve query-generation provenance and retrieval provenance so every paper can be traced to the strategy and query that found it.
9. Bound external calls, generated query counts, candidate counts, and output sizes explicitly.
10. Cache external responses during development when practical.
11. Log the query, query generator, provider, search mode, retrieval timestamp, result count, latency, and errors.
12. Never overwrite a user's existing environment variables or secrets.
13. Never commit API keys, `.env` files, caches, model weights, or generated databases.
14. A failed optional provider must produce a clear error or fall back explicitly; it must not silently alter the scientific meaning of the workflow.
15. Complete milestones in order unless this file is intentionally revised.

## 5. Planned project structure

```text
research-gap/
├── IMPORTANT.md
├── README.md
├── main.py
├── requirements.txt
├── .env.example
├── data/
│   ├── cache/                 # ignored by Git
│   └── evaluations/
├── src/
│   ├── config.py
│   ├── models/
│   │   ├── idea.py
│   │   ├── query.py           # SearchQuery and query provenance
│   │   ├── paper.py
│   │   └── report.py
│   ├── query/
│   │   ├── base.py            # QueryDecomposer / QueryGenerator protocols
│   │   ├── deterministic.py
│   │   ├── openai_decomposer.py
│   │   ├── generator.py       # deterministic query generation
│   │   ├── openai_generator.py
│   │   └── planner.py         # merge, normalize, deduplicate, bound query plan
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── openalex.py
│   │   └── multi_query.py
│   ├── ranking/
│   │   ├── lexical.py
│   │   ├── semantic.py
│   │   └── reranker.py
│   ├── extraction/
│   │   ├── paper_extractor.py
│   │   └── evidence.py
│   ├── analysis/
│   │   ├── comparison.py
│   │   ├── clustering.py
│   │   ├── gap_candidates.py
│   │   └── verification.py
│   ├── reporting/
│   │   ├── markdown.py
│   │   └── citations.py
│   └── storage/
│       ├── cache.py
│       └── repository.py
└── tests/
    ├── fixtures/
    ├── unit/
    └── integration/
```

Create files and folders only when their milestone begins. Do not add empty architecture merely to match this tree.

## 6. Milestone status

| Milestone | Status | Deliverable |
|---|---|---|
| 1. Basic OpenAlex retrieval | Complete | CLI retrieves and normalizes papers |
| 2. Query decomposition and deterministic multi-query retrieval | Complete | Structured facets, deterministic queries, merged results |
| 3. Hybrid query expansion, retrieval and reranking | Complete | Typed hybrid retrieval with lexical + semantic reranking |
| 4. Structured evidence extraction | Complete | Methods, datasets, populations, findings, limitations |
| 5. Literature comparison and clustering | Complete | Research landscape and comparable paper groups |
| 6. Gap candidate generation and verification | Implemented; live smoke blocked | Evidence-backed, qualified gap hypotheses |
| 7. Evaluation harness | Planned | Retrieval and report-quality measurements |
| 8. Local API and persistence | Planned | FastAPI plus PostgreSQL/pgvector if justified |
| 9. Web interface | Planned | Interactive application and evidence views |
| 10. Citation graph and deployment | Planned | Graph exploration, packaging, monitoring |

## 8. Milestone 7 — evaluation harness

Milestone 7 is an offline-first measurement layer over the frozen Milestones 1–6 pipeline. It does
not tune retrieval, ranking, extraction, landscape analysis, candidate generation, verification, or
caching. Evaluation data lives separately from prompts, examples, normalization rules, and manual
tuning decisions. Use `src.evaluation` for strict JSONL datasets and saved predictions, and run the
CLI with `python -m src.evaluation.runner --dataset ... --evaluation-type ... --predictions ...`.

The harness reports retrieval, pairwise deduplication, field-level extraction, evidence attribution,
direct/candidate verification, counterexample discovery, performance, request/token metadata, and
cache hit rates. It can export candidate contexts for later expert 1–5 ratings; it never invents
human usefulness/correctness scores. Provider failures are recorded by case and stage. Token usage
and cost remain unavailable when the provider does not expose reliable usage metadata, and pricing
is supplied through `ModelPricing` rather than fetched at runtime. All reports carry a dataset
version and schema version. Scores describe retrieved and verified evidence only and do not prove
global novelty.

## 7. Milestone 1 — basic retrieval (complete)

### Current behavior

- `main.py` accepts a research idea.

- `OpenAlexClient` sends a regular OpenAlex `search` request.

- Results are normalized to title, reconstructed abstract, authors, year, DOI/URL, citation count,

  and OpenAlex ID.

- Output supports human-readable text and JSON.

### Known limitation

Regular OpenAlex `search` covers title, abstract, and full text. It can therefore retrieve broad or

noisy papers in which query terms appear somewhere in the full text. The OpenAlex website search

observed during development used a narrower title-and-abstract search, so its result set differed.

Neither result set should be treated as ground truth.

### Existing acceptance command

```bash

python3 -m unittest discover -s tests -v

python3 main.py "retrieval augmented generation for medical question answering" --limit 5

```

## 8. Milestone 2 — query decomposition and multi-query retrieval (complete)

### Goal

Convert one unstructured idea into research facets and several complementary queries. Search each

query, merge results, and remove duplicates. This improves recall and makes the search process

inspectable.

### Required structured output

Create a `ResearchIdea` model with these fields:

```json

{

  "original_text": "string",

  "problem": ["string"],

  "population": ["string"],

  "intervention_or_method": ["string"],

  "data_or_modality": ["string"],

  "comparison": ["string"],

  "outcomes": ["string"],

  "domain": ["string"],

  "constraints": ["string"],

  "keywords": ["string"],

  "synonyms": {"canonical term": ["alternative term"]}

}

```

All list fields may be empty. Unknown information must remain empty rather than being invented.

### Decomposer design

Implement one protocol/interface:

```python

class QueryDecomposer(Protocol):

    def decompose(self, idea: str) -> ResearchIdea: ...

```

Implement two backends:

1. `DeterministicDecomposer`

   - Always available and free.

   - Normalizes whitespace and punctuation.

   - Extracts useful noun phrases or keywords with transparent rules.

   - Recognizes explicit connectors such as `using`, `for`, `in`, `among`, `compared with`,

     `while`, and `without` where possible.

   - Does not pretend to fully understand ambiguous ideas.

   - Supplies a stable baseline for tests and evaluation.

2. `OpenAIDecomposer`

   - Optional; activated only when configured.

   - Uses the OpenAI Responses API and Structured Outputs.

   - Produces exactly the validated `ResearchIdea` schema.

   - Uses a cost-sensitive model configurable through `OPENAI_MODEL`; the initial documented

     default should be `gpt-5.6-luna`, not hard-coded throughout the codebase.

   - Uses low reasoning effort because decomposition is a narrow extraction task.

   - Sends only the research idea and a short extraction instruction.

   - Sets API response storage off when appropriate for this stateless operation.

   - Never generates claims about whether the idea is novel.

### Why OpenAI first instead of a local open-weight Llama

The current development laptop has no NVIDIA GPU. A local model would therefore be slower, consume

substantial RAM, complicate installation, and likely produce less reliable schema-following output.

For this small, infrequent extraction call, a hosted model is the pragmatic first LLM backend.

This is not vendor lock-in because callers depend on `QueryDecomposer`, not the OpenAI SDK. A later

`LocalDecomposer` may use Ollama, llama.cpp, or another inference server without changing retrieval.

Do not download model weights during milestone 2.

### Query generation rules

`generator.py` must deterministically create a small, bounded set of queries from the facets.

Milestone 2 intentionally keeps query generation deterministic so it provides a stable, free baseline. LLM-generated query expansion is added in Milestone 3 and must be evaluated against this baseline rather than silently replacing it.

The deterministic rules are:

1. One cleaned version of the original idea.

2. One method + problem query.

3. One problem + population/domain query when available.

4. One method + outcome query when available.

5. One synonym-expanded query using explicit `OR` groups.

6. Never exceed six generated lexical queries in this milestone.

7. Remove duplicate queries after case-folding and whitespace normalization.

8. Do not generate a query with fewer than two meaningful terms unless the original idea itself is

   that short.

Example input:

```text

Using reinforcement learning to build a RAG system with LoRA adaptation

```

Illustrative facets—not hard-coded expected text:

```json

{

  "problem": ["RAG system optimization"],

  "intervention_or_method": ["reinforcement learning", "LoRA adaptation"],

  "domain": ["retrieval-augmented generation"],

  "keywords": ["reinforcement learning", "RAG", "LoRA", "adaptation"]

}

```

Illustrative queries:

```text

reinforcement learning retrieval augmented generation

reinforcement learning RAG optimization

LoRA adaptation retrieval augmented generation

("retrieval augmented generation" OR RAG) AND (LoRA OR "low-rank adaptation")

```

### Multi-query retrieval rules

1. Search each query through the existing OpenAlex client.

2. Record which query and search strategy retrieved each paper.

3. Deduplicate primarily by OpenAlex ID, then DOI, then a normalized title fallback.

4. Preserve all provenance when duplicate records are merged.

5. Do not sum citation counts or overwrite a non-empty abstract with an empty one.

6. Put a configurable ceiling on total candidates, initially 100.

7. Respect provider errors: report partial retrieval clearly if some queries fail.

### CLI behavior

Add options similar to:

```bash

python3 main.py "research idea" --decomposer deterministic

python3 main.py "research idea" --decomposer openai

python3 main.py "research idea" --show-queries --limit 20

```

The default must work without an API key. If `--decomposer openai` is selected without

`OPENAI_API_KEY`, exit with a clear configuration message.

### Milestone 2 tests

- Empty ideas are rejected.

- Deterministic decomposition is stable.

- Missing facets remain empty.

- Query count is bounded and duplicates are removed.

- Synonyms are grouped correctly.

- Duplicate works merge by OpenAlex ID.

- DOI and normalized-title fallbacks work.

- Provenance from multiple queries is preserved.

- One failed query does not discard successful results.

- OpenAI output validation uses mocked responses; unit tests never spend API credits.

### Milestone 2 definition of done

- All old tests still pass.

- New unit tests pass without network access or an API key.

- A live smoke test prints the facets, generated queries, and deduplicated papers.

- The same idea retrieves candidates from more than one query.

- Every paper records its retrieval provenance.

- README documents both decomposer modes and environment configuration.

## 9. Milestone 3 — hybrid query expansion, retrieval and reranking

### Goal

Improve both recall and precision by combining multiple independent retrieval signals:

- the original research idea,
- deterministic facet-based queries,
- optional LLM-generated terminology/concept queries,
- OpenAlex lexical retrieval,
- OpenAlex semantic retrieval,
- and local reranking.

Do not assume the first OpenAlex results are the best papers. Do not assume LLM-generated queries are better merely because they sound more natural. Every retrieval strategy must be measurable against a judged evaluation set.

### 9.1 Query-generator abstraction

Introduce a provider-neutral query-generation interface:

```python
class QueryGenerator(Protocol):
    def generate(self, idea: ResearchIdea) -> list[SearchQuery]: ...
```

Use a normalized internal query object rather than passing raw strings everywhere. It should preserve at least:

```json
{
  "text": "string",
  "strategy": "string",
  "source": "deterministic | llm",
  "provider": "optional provider name"
}
```

The exact Pydantic field names may change during implementation, but callers must not depend on raw OpenAI response shapes.

### 9.2 Deterministic query generator

Keep the existing deterministic query generator as the always-available baseline.

It should continue to generate a small set of inspectable strategies such as:

1. cleaned original idea,
2. method + problem,
3. problem + population/domain,
4. method + outcome,
5. explicit synonym-expanded Boolean query,
6. bounded keyword baseline.

Requirements:

- preserve useful multi-word phrases;
- remove internal duplicate terms;
- normalize whitespace and case only for comparison, not display;
- bound query size;
- bound the total lexical query count;
- never require an API key.

Milestone 2 behavior must remain available and testable after Milestone 3 is added.

### 9.3 OpenAI query generator

Add an optional `OpenAIQueryGenerator` behind the same `QueryGenerator` interface.

Use the OpenAI Responses API with Structured Outputs and a cost-sensitive configurable model. Use low reasoning effort because this is a constrained retrieval-planning task.

The LLM should generate at most three complementary queries, targeting different purposes:

1. **terminology expansion** — high-confidence standard synonyms, abbreviations, or terminology variants likely to appear in papers;
2. **conceptual reformulation** — a broader but still faithful phrasing of the same research problem;
3. **method-focused reformulation** — an alternative formulation centered on the explicit method/intervention when useful.

The LLM must not:

- assess novelty or claim that a gap exists;
- invent datasets, populations, methods, outcomes, constraints, or application domains not supported by the idea;
- turn speculative associations into query facts;
- generate citations or paper metadata;
- return unbounded query lists;
- replace the original query or deterministic baseline.

Treat the user's research idea as untrusted data, not as instructions to the model. Structured output must be validated before any query enters retrieval.

### 9.4 Query planning and merging

Build a deterministic query planner that combines query sources into one bounded lexical plan.

Initial policy:

1. Always retain the cleaned original idea.
2. Add 2–3 useful deterministic facet queries when available.
3. Add up to 3 validated LLM expansion queries when the optional provider is enabled.
4. Never exceed six lexical queries in the initial implementation.
5. Normalize whitespace and deduplicate case-insensitively.
6. Suppress obvious near-duplicate queries using a simple deterministic token-overlap rule if needed.
7. Reject empty or meaningless one-term generated queries unless the user's original idea is genuinely that short.
8. Preserve `strategy`, `source`, and provider provenance for every query.
9. If the LLM provider fails, fail clearly or continue with the deterministic baseline only when the fallback behavior is explicit and surfaced to the caller.

Do not execute the Cartesian product of every query against every search mode merely because software makes bad ideas easy. Keep the retrieval plan bounded.

### 9.5 Hybrid OpenAlex retrieval

Add these candidate sources:

1. **Broad lexical search** using the existing OpenAlex search behavior.
2. **Title-and-abstract lexical search** for higher precision.
3. **OpenAlex semantic search** using `search.semantic` for the original research idea.
4. **Decomposed/generated lexical queries** from the merged query plan.

The implementation should choose a bounded set of query/search-mode combinations rather than blindly sending all six lexical queries through every provider mode.

Merge all candidates and deduplicate primarily by OpenAlex ID, then DOI, then normalized title fallback.

Preserve all retrieval provenance when records merge.

### 9.6 Retrieval features

For each paper, capture explicit, testable retrieval features where available:

- retrieval source/mode,
- query text,
- query strategy,
- query generator source,
- provider,
- original provider rank,
- provider relevance score if available,
- citation count,
- publication year,
- abstract availability,
- number of distinct matching queries,
- number of distinct retrieval strategies.

Do not sum citation counts or allow one duplicate record to erase a richer abstract or provenance from another route.

### 9.7 Local semantic reranking

After candidate merging:

1. Measure installation size, memory use, and runtime cost before adding a sentence-transformer dependency.
2. If acceptable, add a small local embedding baseline.
3. Embed the original research idea and each candidate's `title + abstract`.
4. Compute semantic similarity.
5. Combine semantic similarity with retrieval evidence using explicit, documented, testable weights.
6. Keep the raw features so ranking decisions can be inspected.
7. Later compare this baseline with a cross-encoder reranker instead of assuming a cross-encoder is automatically better.

### 9.8 Evaluation and ablations

Create at least 20 research ideas with manually judged relevant papers.

Evaluate these systems separately:

- **A — Original only:** original OpenAlex lexical search.
- **B — Deterministic expansion:** original + deterministic generated queries.
- **C — LLM expansion:** original + LLM-generated queries.
- **D — Combined query expansion:** original + deterministic + LLM queries.
- **E — Hybrid retrieval:** D + title/abstract + OpenAlex semantic retrieval.
- **F — Hybrid + reranking:** E + local semantic reranker.

Track at minimum:

- Recall@10,
- Recall@50,
- MRR,
- NDCG,
- unique relevant-paper yield by strategy,
- duplicate-query rate,
- candidate overlap between strategies,
- latency,
- OpenAI token/API cost when LLM query generation is enabled.

The LLM query generator is justified only if it improves retrieval quality or useful coverage enough to warrant its latency and cost. Do not select query-generation or reranking strategies based on one attractive example.

### 9.9 Milestone 3 tests

Add unit tests for:

- `SearchQuery` validation and provenance;
- deterministic query generation remains stable;
- OpenAI query output validation uses mocked responses and spends no API credits;
- LLM query count is bounded;
- LLM output cannot add unknown schema fields;
- query planner keeps the original query;
- deterministic and LLM duplicate queries merge correctly;
- near-duplicate suppression is deterministic if implemented;
- provider failure/fallback behavior is explicit;
- semantic, lexical, and title/abstract retrieval provenance is preserved;
- candidate deduplication still works across retrieval modes;
- reranking is deterministic for fixed features and weights.

Add at least one integration smoke test showing that one research idea can produce candidates from more than one query-generation/retrieval strategy.

### 9.10 Milestone 3 definition of done

- Milestones 1 and 2 behavior remains intact.
- All unit tests pass without network access or API credits.
- The CLI works without `OPENAI_API_KEY` using deterministic query generation.
- When OpenAI query generation is enabled, generated queries are shown with their provenance.
- A live smoke test demonstrates lexical and semantic candidate retrieval.
- Every retrieved paper retains enough provenance to explain how it was found.
- Evaluation scripts can compare original-only, deterministic, LLM, combined, hybrid, and reranked variants.
- README documents the query-generation modes, fallback behavior, costs, and evaluation commands.

### 9.11 Implemented Milestone 3 architecture

Milestone 3 was completed as one typed pipeline rather than a second retrieval implementation:

```text
ResearchIdea
  -> deterministic queries + optional OpenAI expansions
  -> QueryPlanner (original retained, exact duplicates merged, maximum six)
  -> original broad + original title/abstract + original semantic routes
  -> generated broad lexical routes
  -> normalized Paper models and provenance-preserving deduplication
  -> lexical and embedding similarity against the original idea
  -> normalized weighted fusion and deterministic top-paper ordering
```

Implemented modules and decisions:

- `src/models/query.py` defines strict `SearchQuery`, query origins, and retrieval modes.
- `src/models/paper.py` is the provider-independent paper boundary. It retains publication
  metadata, retrieval provenance, raw component scores, normalized component scores, final score,
  and ranking mode.
- `src/query/generator.py` retains the Milestone 2 string adapter while the production pipeline
  uses `DeterministicQueryGenerator`. `src/query/openai_generator.py` adds at most three strict,
  mocked-testable OpenAI expansions. `src/query/planner.py` merges their provenance and bounds the
  plan.
- `src/retrieval/openalex.py` owns all OpenAlex request construction, response parsing, abstract
  reconstruction, pagination, timeout/retry/backoff behavior, and a per-process response cache.
  Provider JSON does not leave this boundary. The title/abstract route uses OpenAlex's legacy
  field-scoped filter because no current non-deprecated field-scoped equivalent exists.
- `src/retrieval/multi_query.py` executes a bounded route plan concurrently but merges responses in
  plan order. It never applies every query to every mode. Provider/configuration failures remain
  explicit and successful partial retrieval is preserved.
- `src/retrieval/deduplication.py` resolves identity by normalized OpenAlex ID, DOI, then normalized
  title plus publication year. Conflicting strong identifiers prevent a title-only merge. Merging
  retains all distinct provenance, the richer abstract, author names, and maximum citation count.
- `src/ranking/lexical.py` supplies deterministic title-boosted BM25-style evidence.
  `src/ranking/semantic.py` supplies real cosine similarity through a swappable embedding protocol;
  the OpenAI implementation batches and caches title-plus-abstract embeddings. No lexical overlap
  is mislabeled as semantic relevance.
- `src/ranking/reranker.py` exclusively owns min-max normalization, configurable weighted fusion,
  fallback policy, and deterministic tie-breaking. Citation count is not a relevance feature.
- `src/pipeline.py` is the testable application service consumed by `main.py`. CLI output exposes
  candidate count, ranking mode, optional scores, query origins, and failed routes.
- `src/evaluation/` provides Recall@10, Recall@50, MRR, NDCG@10, and stable identifiers for all six
  planned ablations.

Semantic backend and fallback policy:

- OpenAlex semantic retrieval uses only the original idea and requires `OPENALEX_API_KEY`.
- Semantic reranking uses `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`) and a batched
  OpenAI provider behind `EmbeddingProvider`.
- When embeddings are not configured and fallback is `lexical`, the result is visibly marked
  `lexical_only`, a notice is logged/printed, and `semantic_score` remains absent. Setting fallback
  to `error` makes semantic scoring mandatory.

Milestone 4 must consume the ordered `Paper` models from `ResearchResult.papers`; it must not parse
OpenAlex JSON or recompute retrieval scores. The paper title, abstract, identifiers, and provenance
are the evidence-extraction inputs, while missing abstracts remain explicit missing data.

Remaining Milestone 3 evaluation limitation: metric code and ablation identifiers exist, but no
20-idea manually judged relevance set is committed. Such judgments require real human review and
must not be fabricated. Live provider quality, latency, and cost therefore still need evaluation
outside the mocked test suite.
Current hybrid ranking measures overall topical relevance effectively, but does not yet explicitly enforce facet-level or constraint-level matching. A paper may rank highly because it strongly matches the problem and method while failing an important constraint such as limited labeled data. Later evidence-analysis stages should perform facet-level matching rather than relying solely on aggregate retrieval relevance.


## 10. Milestone 4 — structured evidence extraction

### Goal

Turn each relevant paper into comparable evidence, not a free-form summary.

### Paper evidence schema

- Research question or objective

- Population or setting

- Method/model/intervention

- Baseline or comparison

- Dataset and sample size

- Evaluation metrics

- Main findings

- Limitations explicitly stated by authors

- Future-work statements explicitly stated by authors

- Evidence source: title, abstract, full-text section, and exact location when available

- Confidence and missing-data flags

Start with title and abstract. Full-text extraction is a later enhancement and must respect access

and licensing. Never label an inferred limitation as author-stated.

## 11. Milestone 5 — comparison, clustering, and landscape

1. Create comparable feature tables from extracted evidence.

2. Cluster papers by problem, method, population, dataset, and outcome.

3. Identify dominant combinations and sparsely studied combinations.

4. Detect conflicting results only when outcomes and experimental settings are comparable.

5. Produce a machine-readable landscape plus a concise Markdown view.

6. Add citation-graph exploration only after the text-based comparison is useful.

## 12. Milestone 6 — gap candidates and verification

Generate candidate gaps from explicit patterns:

- Method applied in one domain but not another relevant domain

- Population repeatedly excluded or underrepresented

- Dataset or evaluation setting too narrow

- Conflicting findings under comparable conditions

- Repeated author-stated limitation or future-work direction

- Missing comparison between established approaches

- Lack of replication, longitudinal evaluation, external validation, or real-world testing

Every candidate gap must include supporting and potentially contradicting papers. The verifier must

search specifically for counterexamples before accepting a gap. Final labels should be qualified,

using the user-facing decisions `Well studied`, `Uncertain`, or `Promising gap`—not a fake numeric
novelty percentage.

### Implemented Milestone 6 architecture

Milestones 4 and 5 remain the source of structured evidence and the deterministic
`LiteratureLandscape`. Milestone 6 adds a deterministic `GapCandidateGenerator` that consumes the
research idea, landscape, and `PaperEvidence`; candidates retain their triggering pattern,
landscape observations, evidence roles, and supporting paper IDs. Known invalid evidence
equivalences are rejected by centralized semantic guardrails, and near-duplicate candidates are
consolidated before verification.

`GapVerifier` creates at most three candidate-specific counterexample queries, executes them
through the existing bounded OpenAlex retrieval/normalization stack, ranks results against the
candidate statement, and inspects structured title/abstract evidence. It preserves potential and
confirmed contradictions, retrieval/extraction failures, query provenance, coverage notes, and
one of the three qualified categorical assessments. Verification failures are never interpreted as
evidence that no counterexample exists. The CLI `--show-gaps` runs generation and verification;
`--show-landscape` remains the existing Milestone-5 view. The required live smoke test remains
blocked when the configured provider cannot establish a connection.

### Milestone 6 known limitation

Direct idea verification intentionally favors precision over recall.

Semantically related expressions may remain unmatched when the available
structured canonical values, explicit synonyms, and conservative lexical
containment do not establish equivalence.

Examples include differences such as environmental wording, task
paraphrases, or closely related scientific outcome terminology.

This can produce false-negative or `Uncertain` facet assessments.

Do not solve this by adding domain-specific synonym tables, scientific
regex ontologies, embedding thresholds, or additional LLM matching stages
inside Milestone 6.

The system must prefer `Uncertain` over unsupported confirmation.

Milestone 6 is considered frozen unless a future regression violates a core
invariant, crashes the pipeline, produces unsupported positive claims, or
breaks same-paper verification.

## 13. Milestone 7 — evaluation harness

Measure:

- Retrieval: Recall@10/50, MRR, NDCG

- Deduplication precision and recall

- Extraction field accuracy

- Citation/claim attribution accuracy

- Unsupported-claim rate

- Expert rating of gap usefulness and correctness

- Counterexample discovery rate

- Latency, API cost, token usage, and cache hit rate

Keep an evaluation set separate from prompt examples and tuning decisions.

## 14. Milestone 8 — local service and persistence

After the CLI pipeline is useful:

1. Wrap application services with FastAPI.

2. Add job IDs for long analyses.

3. Persist ideas, queries, raw provider IDs, normalized papers, evidence, and reports.

4. Begin with SQLite if one-user local persistence is sufficient.

5. Move to PostgreSQL/pgvector only when concurrency or vector-scale requirements justify it.

6. Add migrations, configuration validation, and health checks.

## 15. Milestone 9 — web interface

Build the interface only after retrieval and analysis metrics are acceptable. The UI should expose:

- Idea editor

- Generated facets and editable queries

- Retrieval progress and filters

- Ranked papers with “why retrieved” provenance

- Evidence comparison table

- Gap hypotheses with supporting and contradicting citations

- Coverage limitations and export to Markdown/JSON

## 16. Milestone 10 — citation graph, packaging, and deployment

1. Add citation relationships using OpenAlex referenced-work IDs.

2. Use NetworkX before considering a graph database.

3. Containerize only after local installation is stable.

4. Add secrets management, logging, monitoring, rate limiting, retries, and cost budgets.

5. Deploy an API and UI separately only if operational complexity is justified.

## 17. Configuration policy

Expected future environment variables:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.6-luna
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENALEX_API_KEY=
OPENALEX_MAILTO=
RESEARCH_GAP_CACHE_DIR=data/cache
RESEARCH_GAP_RETRIEVAL_CACHE_TTL_SECONDS=21600
RESEARCH_GAP_EXTRACTION_BATCH_SIZE=3
OPENALEX_CANDIDATE_LIMIT=20
RESEARCH_GAP_MAX_CANDIDATES=100
RESEARCH_GAP_LEXICAL_WEIGHT=0.4
RESEARCH_GAP_SEMANTIC_WEIGHT=0.6
RESEARCH_GAP_SEMANTIC_FALLBACK=lexical
```

`.env.example` may contain names and safe defaults, never real credentials.

The same configurable OpenAI model may initially be used for decomposition and LLM query generation. Do not hard-code provider model names throughout the codebase. If later evaluation shows materially different cost/quality trade-offs, separate model settings may be introduced deliberately.

The CLI must remain usable without `OPENAI_API_KEY` through deterministic decomposition and deterministic query generation. Optional LLM query generation must never become a hidden runtime requirement.

## 18. Prompt to give Codex for the next step

```text
Read IMPORTANT.md and README.md completely.

Implement only Milestone 4: structured evidence extraction.

Preserve Milestones 1 through 3, especially the typed SearchQuery and Paper models, retrieval
provenance, original-idea reranking, explicit semantic fallback, and CLI behavior.

Consume ranked `Paper` objects from `ResearchResult.papers`. Create a provider-independent evidence
model for research question, population/setting, method/intervention, baselines, datasets/sample
sizes, metrics, findings, author-stated limitations, future-work statements, evidence location,
confidence, and missing-data flags. Start with titles and abstracts only.

Never invent evidence, never label an inferred limitation as author-stated, and never parse raw
OpenAlex JSON outside the retrieval layer. Keep any optional LLM extraction behind an interface,
require strict structured output, and make provider failures/fallback behavior explicit.

Add unit tests with local fakes and at least one end-to-end smoke test from ranked Paper models to
structured evidence. Do not implement clustering, gap generation, reporting, a web UI, persistence,
or later milestones.
```

## 19. Decision log

- Start as a local CLI, not a website.
- Use OpenAlex as the first literature source.
- Treat OpenAlex website results and API results as retrieval candidates, not ground truth.
- Use hybrid retrieval rather than relying on a single keyword-search mode.
- For query decomposition, keep a deterministic baseline and a provider-neutral interface.
- Implement OpenAI Structured Outputs as the first optional LLM decomposition backend because it is simpler and more reliable on the current non-NVIDIA laptop.
- Keep deterministic query generation as the always-available baseline.
- Add optional LLM query generation in Milestone 3 because semantic terminology expansion and faithful reformulation can retrieve papers that literal facet concatenation misses.
- LLM-generated queries supplement rather than replace the original idea, deterministic queries, or semantic retrieval.
- Keep LLM query generation provider-neutral behind `QueryGenerator`; do not couple retrieval code directly to the OpenAI SDK.
- Preserve query strategy/source/provider provenance so retrieval quality can be analyzed and ranking features can use evidence from multiple routes.
- Bound the initial lexical query plan to at most six queries, including at most three LLM-generated expansions.
- Require LLM query generation to justify its latency and API cost through ablation results against the deterministic baseline.
- Evaluate original-only, deterministic, LLM, combined, semantic/hybrid, and reranked retrieval variants using judged research ideas rather than anecdotal examples.
- Use bounded routes: the original idea receives broad lexical, title/abstract, and semantic retrieval; generated queries receive broad lexical retrieval once each.
- Use a small swappable OpenAI embedding backend instead of adding a large local Torch/sentence-transformer dependency to the CPU-only baseline. Batch and cache embeddings within a run.
- Rerank against the original idea with normalized 0.4 lexical and 0.6 semantic weights. Keep citation count out of topical relevance.
- Keep missing semantic scoring explicit: either mark a lexical-only fallback with no semantic score or fail when strict mode is configured.
- Consider an open-weight local model only after the pipeline and evaluation set exist; compare it on decomposition accuracy, query quality, retrieval impact, latency, memory, and cost before switching.
