# NHL Rules Explainer

A rules-only explainer for new NHL fans: describe a call in plain language and get a plain-English explanation citing the exact rulebook sections that apply.

v0: RAG over the current-season NHL rulebook (FastAPI + Postgres/pgvector + Claude API), with a CLI client and an eval harness.

## Setup

Requires Docker (Docker Desktop or Colima) and [uv](https://docs.astral.sh/uv/).

```sh
cp .env.example .env         # adjust if needed
docker compose up -d --wait  # Postgres 17 + pgvector on localhost:5433
uv sync                      # Python 3.12 venv with all dependencies
uv run nhl ingest 2026-27Rules.pdf --season 2026-27   # parse, embed, load (~15s)
uv run uvicorn nhl_rules.api.main:app --reload   # API on localhost:8000
uv run nhl health            # in another terminal
```

The rulebook PDF is not committed (it's copyrighted); download the current
season's *NHL Official Rules* PDF and put it in the repo root. The first ingest
downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB).

## Data

`nhl ingest` loads one row per rule (its header and any intro text), one per
subsection (e.g. `56.2`), and one per entry in the rulebook's Glossary of Terms
(`Glossary: Possession of the Puck`) into the `rules` table, with an embedding
and a generated `tsvector`. Re-running replaces that season's rows.

The parser reads the PDF span by span and uses fonts and positions for structure.
Page ranges and a few layout special cases (the Rule 29 signals table, the
two-column list in 73.2, right-margin rule references) are specific to the
2026-27 book; `tests/test_parser.py` checks them against the real PDF when it's
present.

Run tests and lint with `uv run pytest` and `uv run ruff check .`.

## Layout

```
src/nhl_rules/
  config.py      settings from environment variables / .env
  schema.sql     `rules` table, applied by `nhl ingest`
  db.py          Postgres connection and health checks
  embeddings.py  local sentence-transformers model
  ingest/        PDF parser and database loader
  api/main.py    FastAPI app
  cli.py         `nhl` CLI: an HTTP client over the API, plus local `ingest`
db/init/         SQL run when the database volume is first created
tests/
```
