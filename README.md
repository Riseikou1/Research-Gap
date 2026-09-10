# Research GAP

Research GAP turns a research idea into a bounded, inspectable hybrid literature search. It
decomposes the idea, plans complementary queries, retrieves OpenAlex candidates through broad,
title/abstract, and semantic routes, deduplicates them, and reranks them against the original idea.

Read [`IMPORTANT.md`](IMPORTANT.md) before implementing another milestone. It is the architectural
source of truth.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` is loaded for local development without overriding variables already exported by the shell.
Never commit `.env` or API credentials.

## Run

The deterministic default needs no OpenAI key:

```bash
python main.py "RAG using LoRA" --show-queries --show-scores
```

Useful options:

```bash
python main.py "your idea" --limit 10 --json
python main.py "your idea" --decomposer openai
python main.py "your idea" --query-generator openai --show-queries
python main.py "your idea" --show-gaps --show-evidence
python main.py "your idea" --show-evidence --full-text
```

`--query-generator openai` adds up to three validated LLM expansions to the original and
deterministic baseline; it never replaces them. `--limit` controls displayed top results. The
internal unique candidate pool defaults to 100.
`--show-gaps` runs the complete Milestone-6 path: structured evidence extraction, the deterministic
Milestone-5 landscape, pattern-grounded candidate generation, candidate consolidation, and targeted
direct-idea and counterexample verification. Each displayed candidate includes its trigger pattern, landscape
basis, supporting evidence, verification queries, counterexamples (if confirmed), and a qualified
assessment. Verification queries use the existing OpenAlex normalization/retrieval boundary and
are bounded to three queries and ten results per query.

The user-facing labels are `well_studied`, `uncertain`, and `promising_gap`. `well_studied` requires
a direct match to the important idea facets. `uncertain` covers failures, sparse coverage, and
contextual/partial matches. `promising_gap` is reserved for a grounded positive signal after
successful targeted verification finds no direct match; it is not a probability of novelty. Generic
landscape buckets such as `other` and `unknown` are never used as scientific entities or query
terms. Title/abstract evidence cannot establish global novelty, and absence from the analyzed top
papers is never treated as proof that no work exists.
`--show-landscape` prints the deterministic Milestone-5 literature landscape: normalized feature
frequencies, observed combinations, evidence coverage, and conservatively comparable conflicts.

Provider-backed planning and raw retrieval results use the local SQLite cache under
`RESEARCH_GAP_CACHE_DIR`. Planning rows are versioned by normalized idea, provider/model, and
planning configuration; retrieval rows expire after `RESEARCH_GAP_RETRIEVAL_CACHE_TTL_SECONDS`.
Evidence and paper embeddings use the same database, while ranking and gap reasoning still run on
every invocation.

Full-text enrichment is opt-in with `--full-text`. It runs only for the already selected
`RESEARCH_GAP_EVIDENCE_LIMIT` papers, follows metadata-declared open-access locations, and accepts
bounded text-layer PDFs, JATS/XML, or structured article HTML. Each paper falls back independently
to title/abstract extraction when no usable document is available. Evidence from full text carries
the original heading, normalized section role, and stable section/chunk ID. Coverage reports make
unavailable, fetch-failed, parse-failed, and truncated documents explicit; an empty evidence field
means “not extracted,” not “not reported by the paper.” Downloads and parsed documents use the
existing SQLite cache with a parser-versioned key.

## Semantic behavior and fallback

Two distinct semantic capabilities are used:

- OpenAlex semantic candidate retrieval uses the original idea and requires `OPENALEX_API_KEY`.
- Local reranking uses real title-and-abstract embeddings through the batched OpenAI embedding
  backend and requires `OPENAI_API_KEY`.

Without `OPENAI_API_KEY`, ranking explicitly falls back to lexical-only mode and prints a notice;
`semantic_score` remains `null`. Without `OPENALEX_API_KEY`, the semantic retrieval route is
reported as a failed route while successful lexical routes are retained. Set
`RESEARCH_GAP_SEMANTIC_FALLBACK=error` to require semantic reranking instead.

## Configuration

Safe defaults are documented in [`.env.example`](.env.example). The main tuning variables are:

| Variable | Default | Purpose |
|---|---:|---|
| `OPENALEX_CANDIDATE_LIMIT` | `20` | Candidates requested per bounded route |
| `RESEARCH_GAP_MAX_CANDIDATES` | `100` | Unique pool ceiling before reranking |
| `OPENALEX_TIMEOUT_SECONDS` | `20` | Provider request timeout |
| `OPENALEX_MAX_RETRIES` | `2` | Retry count for network, 429, and 5xx failures |
| `RESEARCH_GAP_RETRIEVAL_WORKERS` | `4` | Maximum concurrent routes |
| `RESEARCH_GAP_LEXICAL_WEIGHT` | `0.4` | Normalized lexical fusion weight |
| `RESEARCH_GAP_SEMANTIC_WEIGHT` | `0.6` | Normalized semantic fusion weight |
| `RESEARCH_GAP_CONSTRAINT_WEIGHT` | `0.15` | Maximum topicality-gated constraint boost |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Swappable embedding model |
| `OPENAI_EXTRACTION_MODEL` | `OPENAI_MODEL` | Structured evidence extraction model |
| `RESEARCH_GAP_EVIDENCE_LIMIT` | `10` | Maximum ranked papers sent to extraction |
| `RESEARCH_GAP_EXTRACTION_WORKERS` | `4` | Maximum concurrent evidence extractions |
| `RESEARCH_GAP_EXTRACTION_BATCH_SIZE` | `3` | Uncached papers per bounded extraction request |
| `RESEARCH_GAP_RETRIEVAL_CACHE_TTL_SECONDS` | `21600` | Freshness window for persistent retrieval results |
| `RESEARCH_GAP_CACHE_DIR` | `data/cache` | Local SQLite cache directory |
| `RESEARCH_GAP_DATABASE_PATH` | `data/research_gap.sqlite3` | Durable analysis-history database |
| `DATABASE_URL` | unset | PostgreSQL URL for durable application data; overrides the SQLite path |
| `RESEARCH_GAP_MAX_ANALYSIS_WORKERS` | `2` | Maximum concurrent API analysis jobs |
| `RESEARCH_GAP_FULL_TEXT_TIMEOUT_SECONDS` | `12` | Per-document request timeout |
| `RESEARCH_GAP_FULL_TEXT_MAX_BYTES` | `8000000` | Maximum streamed response bytes |
| `RESEARCH_GAP_FULL_TEXT_MAX_DOCUMENT_CHARS` | `120000` | Maximum normalized document characters |
| `RESEARCH_GAP_FULL_TEXT_MAX_SECTION_CHARS` | `20000` | Maximum characters retained per section |
| `RESEARCH_GAP_FULL_TEXT_MAX_CHUNK_CHARS` | `8000` | Fallback chunk size without headings |
| `RESEARCH_GAP_FULL_TEXT_MAX_CONTEXT_CHARS` | `30000` | Maximum full-text extraction context |
| `RESEARCH_GAP_FULL_TEXT_MAX_REDIRECTS` | `3` | Maximum validated HTTP redirects |
| `RESEARCH_GAP_FULL_TEXT_NEGATIVE_CACHE_TTL_SECONDS` | `3600` | Cache lifetime for failed/unavailable locations |

## Architecture

```text
idea -> decomposition -> typed query plan -> bounded OpenAlex routes
     -> normalized Paper models -> provenance-preserving deduplication
     -> lexical + semantic scoring against original idea -> score fusion -> top papers
```

Every result retains matched queries, query-generator origins, retrieval modes, provider rank and
score where available, and lexical/semantic/final relevance scores. Citation count is retained as
metadata but does not affect Milestone 3 relevance.

## Milestone 8 local API and persistence

The FastAPI service runs the same `ResearchPipeline` used by the CLI. `POST /analyses` writes a
`pending` record and returns immediately; a bounded local thread pool transitions it through
`running` to `completed` or `failed`. Completed records retain the configuration snapshot, generated
queries, every retrieved paper ID, normalized top papers, extracted evidence, landscape, gap
candidates, direct assessment, verification details, timings, and work metrics. This durable history
is stored separately from the expiring provider cache.

Apply the ordered SQLite/PostgreSQL migrations (selected by `DATABASE_URL`):

```bash
python -m src.persistence.migrate
```

Start the local service:

```bash
uvicorn src.api.app:app --reload
```

Start an analysis:

```bash
curl -X POST http://127.0.0.1:8000/analyses \
  -H "Content-Type: application/json" \
  -d '{"research_idea":"federated learning for adaptive traffic signal control","full_text":false}'
```

Use the returned ID to inspect status and retrieve the final result:

```bash
curl http://127.0.0.1:8000/analyses/<analysis_id>
curl 'http://127.0.0.1:8000/analyses?limit=20'
curl -X DELETE http://127.0.0.1:8000/analyses/<analysis_id>
curl http://127.0.0.1:8000/health
```

The service runs the complete evidence, landscape, gap-generation, and verification path, so a real
analysis currently requires `OPENAI_API_KEY`; OpenAlex lexical retrieval itself remains available
without an OpenAlex key. Active jobs cannot be deleted because provider calls cannot be safely
interrupted. Queued and completed/failed records can be deleted without clearing shared caches.

## Milestone 9 web application

Milestone 9 is the multi-user web interface (full-text enrichment is already part of the pipeline).
It adds signed guest Quick Search, verified Supabase accounts, private histories, two lifetime free
Full Gap Analysis credits, an auditable transactional ledger, Stripe test subscriptions, durable
job stages, owner-only exports, profiles/avatars, and a server-authorized admin dashboard. Quick
Search never constructs paid OpenAI decomposition, embedding, extraction, gap, verification, or
full-text components. Full Analysis remains the existing Python pipeline; the Next.js client does
not reproduce scientific logic.

Start the backend and frontend in separate terminals:

```bash
python -m src.persistence.migrate
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. Authentication, email, avatar uploads, payments, and the administrator
require owner-managed Supabase/Stripe test configuration; they are not represented as externally
complete. Every manual action, environment value, webhook URL, payout warning, cost worksheet,
deployment step, and pre-launch check is in
[`docs/MILESTONE_10_OWNER_SETUP.md`](docs/MILESTONE_10_OWNER_SETUP.md).

The report UI uses the provider-independent Pydantic serialization through strict Zod contracts and
presents one vertically readable, citation-connected report. Provider exceptions are retained in
server logs but removed from normal API responses and exports. Full-text coverage explicitly
distinguishes successful inspection, abstract fallback, metadata-only evidence, unavailable text,
fetch/parse failures, and truncation.

Verified email addresses receive the two-credit allowance only once across account recreation. The
server stores a keyed HMAC identity in `lifetime_credit_identities`; it does not store the deleted
email there. Account deletion is blocked while an active paid subscription remains, deletes the
Supabase Auth identity through the server-only Admin API, removes analyses/avatar data, and keeps
only the minimal pseudonymous anti-abuse marker plus required audit/payment records.

The historical unauthenticated single-owner API is available only when
`RESEARCH_GAP_TRUSTED_LOCAL_MODE=true`; use it solely on a loopback-bound local service. The safe
default treats unauthenticated web requests as isolated guests.

## Test

```bash
python -m unittest discover -s tests -v
```

The suite uses fakes for OpenAlex, OpenAI Structured Outputs, and embeddings; it makes no network or
paid API calls. Retrieval metric helpers and the six Milestone 3 ablation identifiers are in
`src/evaluation/`. A genuine manually judged evaluation set is still required before comparing
production retrieval quality.

## Milestone 7 evaluation harness

The evaluation package measures the existing pipeline without changing its scientific behavior. It
supports offline JSONL scoring for retrieval (Recall@10/50, MRR, graded NDCG@10), pairwise
deduplication, structured evidence fields, claim attribution, conservative verification labels and
counterexamples, plus performance/cache accounting. Human usefulness is supported through JSONL/CSV
annotation exports and 1–5 ratings; ratings must come from actual expert review.

Datasets use stable case IDs. An optional first row such as
`{"_meta":{"dataset_version":"m7-v1"}}` records the benchmark version. Gold annotations remain
separate from saved predictions:

```bash
python -m src.evaluation.runner \
  --dataset data/evaluations/retrieval.jsonl \
  --evaluation-type retrieval \
  --predictions evaluation/results/retrieval_predictions.jsonl \
  --output evaluation/results/retrieval_report.json
```

Metric computation is fully offline. Provider failures are recorded by case and stage rather than
silently removed from denominators; malformed or missing predictions therefore contribute zero for
the affected case. Reports use schema `m7-v2` and carry dataset metadata, raw numeric JSON metrics,
existing pipeline timings, provider request counters, token metadata when exposed, and cache hit
rates. The CLI exits with status 1 when any case fails and status 2 for invalid input or
configuration, so it can be used safely in CI. Do not place evaluation examples in production
prompts or benchmark-specific normalization logic. Absence of direct evidence remains `uncertain`
and never proves global novelty; assessments describe retrieved and verified evidence only.
