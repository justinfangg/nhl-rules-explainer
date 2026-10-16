-- Applied idempotently by `nhl ingest`.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rules (
    id              serial PRIMARY KEY,
    season          text NOT NULL,               -- e.g. '2026-27'
    rule_number     text NOT NULL,               -- '56.2', '56', or 'Glossary: Torso'
    parent_number   text,                        -- '56' for subsections, NULL otherwise
    title           text NOT NULL,
    section         text NOT NULL,
    text            text NOT NULL,               -- cleaned text; paragraphs separated by \n
    -- No fixed dimension, so embedding models can be swapped for experiments.
    -- At ~600 rows an exact scan is fast, so there is no ANN index.
    embedding       vector,
    embedding_model text,
    tsv             tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', title), 'A') ||
        setweight(to_tsvector('english', text), 'B')
    ) STORED,
    UNIQUE (season, rule_number)
);

CREATE INDEX IF NOT EXISTS rules_tsv_idx ON rules USING gin (tsv);

-- Hand-curated fan language -> rulebook terms (data/glossary.json), loaded by `nhl glossary`.
CREATE TABLE IF NOT EXISTS glossary (
    id              serial PRIMARY KEY,
    term            text NOT NULL UNIQUE,        -- fan phrase, lowercase: 'too many men'
    rule_numbers    text[] NOT NULL CHECK (cardinality(rule_numbers) > 0),
    canonical_term  text NOT NULL                -- rulebook wording: 'too many men on the ice'
);
