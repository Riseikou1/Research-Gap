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

## Architecture

```text
idea -> decomposition -> typed query plan -> bounded OpenAlex routes
     -> normalized Paper models -> provenance-preserving deduplication
     -> lexical + semantic scoring against original idea -> score fusion -> top papers
```

Every result retains matched queries, query-generator origins, retrieval modes, provider rank and
score where available, and lexical/semantic/final relevance scores. Citation count is retained as
metadata but does not affect Milestone 3 relevance.

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
silently removed from denominators. Reports carry dataset/schema metadata, raw numeric JSON metrics,
existing pipeline timings, provider request counters, token metadata when exposed, and cache hit
rates. Do not place evaluation examples in production prompts or benchmark-specific normalization
logic. Absence of direct evidence remains `uncertain` and never proves global novelty; assessments
describe retrieved and verified evidence only.
