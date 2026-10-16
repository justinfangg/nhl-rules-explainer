# NHL Rules Explainer

A rules-only explainer for new NHL fans: describe a call in plain language and get a plain-English explanation citing the exact rulebook sections that apply.

v0: RAG over the current-season NHL rulebook (FastAPI + Postgres/pgvector + Claude API), with a CLI client and an eval harness.

## Setup

Requires Docker (Docker Desktop or Colima) and [uv](https://docs.astral.sh/uv/).

```sh
cp .env.example .env         # adjust if needed
docker compose up -d --wait  # Postgres 17 + pgvector on localhost:5433
uv sync                      # Python 3.12 venv with all dependencies
uv run uvicorn nhl_rules.api.main:app --reload   # API on localhost:8000
uv run nhl health            # in another terminal
```

Run tests and lint with `uv run pytest` and `uv run ruff check .`.

## Layout

```
src/nhl_rules/
  config.py      settings from environment variables / .env
  db.py          Postgres connection and health check
  api/main.py    FastAPI app
  cli.py         `nhl` CLI, a thin HTTP client over the API
db/init/         SQL run when the database volume is first created
tests/
```
