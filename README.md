# Exam Reconciler

Agent for reconciling exam topics from PDF/Excel sources into a normalized ranking sheet using Supabase hybrid search (pgvector + FTS).

## Overview

The goal is to import question banks, compute embeddings, perform hybrid retrieval and output a consolidated Excel with similarity scoring and provenance. The stack is Python/FastAPI with PostgreSQL (Supabase or local).

### Key features

- PDF and Excel ingestion with idempotent upserts
- Normalization rules and synonym mapping
- Embeddings via sentence-transformers or OpenAI
- Hybrid retrieval using Supabase pgvector + full-text search
- Aggregation and scoring with provenance
- REST API and CLI
- Docker and local compose for development
- Tests and CI workflow

## Repository structure

```
exam-reconciler/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ .env.example
├─ Dockerfile
├─ docker-compose.yml
├─ sql/
│  ├─ 000_init_tables.sql
│  └─ 001_hybrid_search_function.sql
├─ src/
│  ├─ __init__.py
│  ├─ main.py                  # FastAPI app
│  ├─ cli.py                   # Simple CLI commands
│  ├─ config.py                # env parsing
│  ├─ db/
│  │  ├─ __init__.py
│  │  ├─ client.py             # Supabase/Postgres client wrapper
│  │  └─ migrations.md
│  ├─ ingest/
│  │  ├─ __init__.py
│  │  ├─ pdf_parser.py         # extract questions/theme/subtheme
│  │  └─ excel_reader.py       # read input excel
│  ├─ normalize/
│  │  ├─ __init__.py
│  │  └─ normalizer.py         # canonicalization, synonyms mapping
│  ├─ embeddings/
│  │  ├─ __init__.py
│  │  └─ embedder.py           # interface: get_embedding(text)
│  ├─ retriever/
│  │  ├─ __init__.py
│  │  └─ hybrid_retriever.py   # uses Supabase hybrid SQL, returns candidates
│  ├─ aggregator/
│  │  ├─ __init__.py
│  │  └─ consolidate.py        # builds final rows + scoring
│  ├─ exporters/
│  │  ├─ __init__.py
│  │  └─ excel_writer.py       # writes final excel
│  └─ utils/
│     └─ logging.py
├─ tests/
│  ├─ test_normalizer.py
│  ├─ test_aggregator.py
│  └─ test_retriever.py  (mocked)
└─ .github/workflows/ci.yml
```

## Installation

```powershell
# install Python 3.11+ from https://www.python.org/downloads/
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

See `.env.example` for required environment variables. Copy to `.env` and fill values (Supabase URL/KEY, OpenAI key optional).

## Database setup

If using Supabase, create project, enable `pgvector` extension and run SQL in `sql/` folder via SQL editor. If local, start via `docker-compose` then run:

```bash
psql -f sql/000_init_tables.sql
psql -f sql/001_hybrid_search_function.sql   # updated version accepts embedding first, then text
psql -f sql/002_theme_stats.sql              # optional table used by reconcile step
```

The `hybrid_search` function now has signature `(embedding, text, match_count, alpha, beta)` and returns both vector similarity and FTS scores. The reconciliation logic will also look up `theme_stats` by normalized tema/subtema if available and include ranking/color metadata in the output.

## Usage

CLI examples:

```powershell
python -m src.cli ingest-pdf path/to/FAMERP.pdf
python -m src.cli reconcile path/to/manchesters.xlsx output.xlsx
```

API examples:

```bash
curl -F "file=@FAMERP.pdf" http://localhost:8000/ingest/pdf
curl -F "file=@manchesters.xlsx" http://localhost:8000/reconcile -o result.xlsx
```

## Testing

```bash
pip install -r requirements.txt
pytest
ruff .
black --check .
```

## Docker

Build and run local stack:

```bash
docker-compose up --build
```

Application will be available at `http://localhost:8000` and Postgres at port 5432.

## Design choices and alternatives

- **Hybrid search**: Combines vector similarity with FTS for richer relevance. Supabase simplifies maintenance.
- **Simpler path**: drop vectors, rely on FTS + synonyms for small datasets.
- **Scaling**: external vector DB, asynchronous workers for ingestion, human feedback loop for normalization, caching embeddings.
- **Caveats**: PDF parsing heuristics may misclassify; normalization rules are minimal.

## Backlog

1. Web UI for manual review
2. Background workers with Celery/RQ
3. Add synonyms table with admin UI
4. Support multiple languages
5. Caching layer for embeddings

```

```
