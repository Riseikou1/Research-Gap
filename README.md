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
```

`--query-generator openai` adds up to three validated LLM expansions to the original and
deterministic baseline; it never replaces them. `--limit` controls displayed top results. The
internal unique candidate pool defaults to 100.

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
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Swappable embedding model |
| `OPENAI_EXTRACTION_MODEL` | `OPENAI_MODEL` | Structured evidence extraction model |
| `RESEARCH_GAP_EVIDENCE_LIMIT` | `10` | Maximum ranked papers sent to extraction |

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
