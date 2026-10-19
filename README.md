# NHL Rules Explainer

A rules-only explainer for new NHL fans. Describe a call in plain language
("why did he get 2 minutes for pulling the guy back with his stick?") and get a
plain-English answer that quotes the exact rules from the current NHL rulebook,
offers alternatives when the description fits more than one rule, and says so
when it can't know something.

v0 is retrieval-augmented generation over the 2026-27 *NHL Official Rules*: a
FastAPI service, Postgres with pgvector, the Claude API for two calls per
question, a CLI client, and an eval harness.

## How it works

```mermaid
flowchart LR
    Q([Question]) --> U["1 · Understand<br/>Claude call 1<br/>slang rewrite, rule numbers,<br/>question type"]
    U --> R{"2 · Route"}
    R -- out of scope --> D([One-sentence deflection])
    R -- situation / definition / judgment --> RT["3 · Retrieve<br/>exact lookup + keyword & vector<br/>search, fused, top 8"]
    RT --> G["4 · Generate<br/>Claude call 2<br/>structured answer citing<br/>only the provided rules"]
    G --> V{"5 · Verify<br/>cited rules provided?<br/>excerpts verbatim?"}
    V -- pass --> A([6 · Render: answer])
    V -- "fail (once)" --> G
    V -- fail twice --> F([Fallback: retrieved rules,<br/>no explanation])

    DB[("Postgres + pgvector<br/>rules · glossary")] -.-> U
    DB -.-> RT
```

1. **Understand** rewrites the question in rulebook terms using a hand-curated
   fan glossary ("the box" → penalty bench), extracts rule numbers the user typed,
   and classifies it as a situation, definition, judgment call or out of scope.
2. **Route** deflects out-of-scope questions (scores, trades, stats, other
   leagues) and flags judgment calls ("was the ref wrong?") for neutral framing.
3. **Retrieve** looks up named rules exactly, then runs Postgres full-text and
   vector search separately, merges them with reciprocal rank fusion, keeps the
   top 8 subsections, and attaches each one's parent rule title.
4. **Generate** writes the answer object (summary, primary rule, alternatives,
   terms, caveats) from the retrieved rules plus a fixed set of core penalty
   definitions (minor, major, misconduct, match, short-handed, ...).
5. **Verify** checks deterministically that every cited rule was provided and
   every excerpt is a word-for-word quote of it. A failed answer gets one retry
   with the errors listed; a second failure returns the retrieved rules with a
   note instead of an unverified explanation.
6. **Render**: the API returns the answer object; `nhl ask` formats it.

## Results

From the 73-case eval set in [`eval/cases.json`](eval/cases.json) (plain,
slang and vague questions, judgment calls and out-of-scope questions), on
`claude-opus-5-5` (Understand effort `low`, Generate and judge `medium`) and
`BAAI/bge-small-en-v1.5` embeddings.

| Metric | Result | What it measures |
|---|---|---|
| Routing accuracy | 95.9% | Question type classified as labeled (all 73 cases) |
| Recall@5 / @8 | 92.1% / 92.1% | An expected rule reached the generator (63 in-scope cases) |
| Verification pass rate | 100% | Answers passing citation checks on the first try (63) |
| Citation accuracy | 95.2% | Primary citation is an expected rule (63) |
| Faithfulness | 4.38 / 5; 95% score ≥ 4 | LLM judge: claims supported by the provided rule text (63) |

A full run costs about $5 and takes about 5 minutes; a single question takes a
median of 16 s and about $0.04.

How retrieval got there (recall@5 on the 63 in-scope cases):

| Search query | Keyword | Vector | Hybrid |
|---|---|---|---|
| Raw question | 57.1% | 66.7% | 81.0% |
| Understand rewrite | 81.0% | 92.1% | 92.1% |

Adding the core penalty definitions to the generator's context, and telling it
to explain only what the provided rules say, raised faithfulness from 3.49 to
4.38 (answers scoring ≥ 4: 46% → 95%). Before, answers routinely explained
"a minor is two minutes" from general knowledge when no retrieved rule said so.

The eval set is small, so differences of a case or two between runs are noise;
LLM stages also vary slightly from run to run.

## Setup

Requires Docker (Docker Desktop or Colima), [uv](https://docs.astral.sh/uv/),
and an Anthropic API key scoped to a workspace.

```sh
cp .env.example .env         # then set ANTHROPIC_API_KEY
docker compose up -d --wait  # Postgres 17 + pgvector on localhost:5433
uv sync                      # Python 3.12 venv with all dependencies
uv run nhl ingest 2026-27Rules.pdf --season 2026-27   # parse, embed, load (~15 s)
uv run uvicorn nhl_rules.api.main:app                 # API on localhost:8000
uv run nhl health            # in another terminal
```

The rulebook PDF isn't committed (it's copyrighted): download the current
season's *NHL Official Rules* PDF and put it in the repo root. The first ingest
downloads the embedding model (~130 MB).

## Using it

```sh
uv run nhl ask "why did he get 2 minutes for pulling the guy back with his stick"
uv run nhl ask "was the ref wrong to call goalie interference?" --debug
uv run nhl rule 55.1                 # one subsection, with its rule's title
uv run nhl rule 55                   # every subsection of Rule 55
```

`--debug` adds the Understand output, the retrieved rules with keyword and
vector ranks, verification results, token usage and timing.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question": str, "debug": bool}` → the answer object |
| `GET /rules/{number}` | One rule or subsection with parent context; a whole rule lists its subsections |
| `GET /search?q=&mode=hybrid&k=8` | Raw retrieval results with scores (`keyword`, `vector` or `hybrid`) |
| `GET /health` | Database, loaded rules, and embedding model readiness |

## Evaluating

```sh
uv run nhl eval                    # full pipeline + faithfulness judge (~$5)
uv run nhl eval --no-judge         # skip the judge (~$3)
uv run nhl eval --understand       # routing accuracy + recall with rewrites (~$0.15)
uv run nhl eval --retrieval-only   # recall for keyword, vector, hybrid (free)
```

Each run writes a timestamped Markdown report and a JSON file with every case to
`eval/results/` (not committed). See [`eval/README.md`](eval/README.md) for the
case format and how rule matching works.

Tests and lint: `uv run pytest` and `uv run ruff check .`. Tests that need the
rulebook PDF or a loaded database skip themselves when those aren't present;
none of them call the Claude API.

## Data

`nhl ingest` loads one row per rule, one per subsection (e.g. `56.2`), and one
per entry in the rulebook's Glossary of Terms (`Glossary: Possession of the
Puck`) into the `rules` table, with an embedding and a generated `tsvector`.
Re-running replaces that season's rows; it refuses to load a PDF that doesn't
parse like the rulebook.

The parser reads the PDF span by span and uses fonts and positions for
structure. Page ranges and a few layout special cases (the Rule 29 signals
table, the two-column list in 73.2, right-margin rule references) are specific
to the 2026-27 book; `tests/test_parser.py` checks them against the real PDF
when it's present.

`data/glossary.json` maps 71 fan phrases to rulebook terms and rule numbers.
`nhl ingest` loads it after the rules; after editing it, run `uv run nhl
glossary`. Loading fails, and leaves the table unchanged, if a term cites a rule
number that isn't in the loaded season.

## Configuration

Set in `.env` (see [`.env.example`](.env.example)):

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Required for `ask` and LLM evals |
| `LLM_MODEL` | `claude-opus-5-5` | Understand and Generate |
| `UNDERSTAND_EFFORT` / `GENERATE_EFFORT` | `low` / `medium` | |
| `JUDGE_MODEL` / `JUDGE_EFFORT` | `claude-opus-5-5` / `medium` | Faithfulness judge (evals only) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Changing it requires re-running `nhl ingest` |
| `RULEBOOK_SEASON` | `2026-27` | |
| `DATABASE_URL` | `postgresql://nhl:nhl@localhost:5433/nhl_rules` | |
| `NHL_API_URL` | `http://localhost:8000` | Where the CLI finds the API |

The Claude calls enable the API's server-side refusal fallback, so a rare
safety-classifier decline is retried on another model instead of failing.

## Layout

```
src/nhl_rules/
  api/main.py    FastAPI app
  cli.py         `nhl` CLI: an HTTP client over the API, plus local ingest and evals
  pipeline.py    the six stages wired together (POST /ask)
  understand.py  Understand: slang rewrite, rule extraction, classification
  retrieval.py   exact lookup, keyword / vector / hybrid search
  generate.py    Generate: the answer prompt, core definitions, structured output
  verify.py      Verify: deterministic citation and excerpt checks
  answer.py      the answer object
  judge.py       faithfulness judge (evals only)
  llm.py         shared Anthropic client and token/cost accounting
  embeddings.py  local sentence-transformers model
  glossary.py    fan-term glossary: validation and loading
  ingest/        PDF parser and database loader
  evals/         eval cases and the retrieval / Understand / full eval runners
  config.py      settings from environment variables / .env
  db.py          Postgres connection, schema, health checks
  schema.sql     `rules` and `glossary` tables
data/glossary.json  hand-curated fan terms
eval/cases.json     the eval set
db/init/            SQL run when the database volume is first created
tests/
```

## Limits

- One rulebook season, and the parser is tuned to the 2026-27 PDF's layout.
- The Reference Tables (summaries such as Table 14, goalkeeper interference
  situations) aren't ingested.
- Answers are based only on the user's description; the tool never rules on
  whether a call was correct.
- The rulebook is copyrighted; this is fine for a personal or portfolio project
  but should be revisited before any public launch.
