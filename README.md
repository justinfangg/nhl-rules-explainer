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

`data/glossary.json` maps fan language to rulebook terms and rule numbers
("too many men" → *too many men on the ice*, Rule 74). `nhl ingest` loads it
after the rules; after editing it, run `uv run nhl glossary` to reload. Loading
fails, and leaves the table unchanged, if any term cites a rule number that
isn't in the loaded season.

The parser reads the PDF span by span and uses fonts and positions for structure.
Page ranges and a few layout special cases (the Rule 29 signals table, the
two-column list in 73.2, right-margin rule references) are specific to the
2026-27 book; `tests/test_parser.py` checks them against the real PDF when it's
present.

## Using it

```sh
uv run nhl rule 55.1                 # one subsection, with its rule's title
uv run nhl rule 55                   # every subsection of Rule 55
curl 'localhost:8000/search?q=goalie+in+the+corner&mode=hybrid&k=8'   # raw retrieval
uv run nhl eval --retrieval-only     # recall@5/@8 for keyword, vector and hybrid
```

Retrieval searches subsections and glossary definitions. Hybrid mode merges the
top 50 keyword (Postgres full-text, any-term match) and vector (cosine) results
with reciprocal rank fusion and keeps the top 8; each hit carries its parent
rule's title. Rule numbers named in a question are looked up exactly and fill
the first slots. Eval reports go to `eval/results/` (not committed).

Run tests and lint with `uv run pytest` and `uv run ruff check .`.

## Layout

```
src/nhl_rules/
  config.py      settings from environment variables / .env
  schema.sql     `rules` table, applied by `nhl ingest`
  db.py          Postgres connection and health checks
  embeddings.py  local sentence-transformers model
  retrieval.py   exact lookup, keyword / vector / hybrid search
  evals/         eval cases and the retrieval eval runner
  glossary.py    fan-term glossary: validation and loading
  ingest/        PDF parser and database loader
  api/main.py    FastAPI app
  cli.py         `nhl` CLI: an HTTP client over the API, plus local `ingest`
data/glossary.json  hand-curated fan terms
db/init/         SQL run when the database volume is first created
tests/
```
