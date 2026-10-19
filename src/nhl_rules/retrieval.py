"""Retrieval over the `rules` table: exact lookup, keyword, vector and hybrid search.

Searchable rows are subsections and glossary definitions. Rule-level rows ("56") hold
only the rule's title in the 2026-27 book, so they are attached to hits as parent
context rather than returned as hits themselves.
"""

from typing import Literal

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from nhl_rules.config import get_settings
from nhl_rules.db import connect
from nhl_rules.embeddings import embed_query

Mode = Literal["keyword", "vector", "hybrid"]
MODES: tuple[Mode, ...] = ("keyword", "vector", "hybrid")

TOP_K = 8
CANDIDATES = 50  # per-method candidates fed into rank fusion
RRF_K = 60  # standard reciprocal rank fusion constant


class ParentContext(BaseModel):
    rule_number: str
    title: str
    text: str


class Rule(BaseModel):
    rule_number: str
    parent_number: str | None
    title: str
    section: str
    text: str
    parent: ParentContext | None = None


class Hit(Rule):
    score: float
    keyword_rank: int | None = None  # 1-based rank in keyword results, if present
    vector_rank: int | None = None


_COLUMNS = "rule_number, parent_number, title, section, text"
_SEARCHABLE = "season = %(season)s AND text <> ''"


def _season(season: str | None) -> str:
    return season or get_settings().rulebook_season


# --- exact lookup ------------------------------------------------------------


def get_rule(number: str, season: str | None = None) -> Rule | None:
    """One rule or subsection by number, with its parent rule's header attached."""
    with connect() as conn:
        rules = _fetch(conn, [number], _season(season))
    return rules[0] if rules else None


def get_rules(numbers: list[str], season: str | None = None) -> list[Rule]:
    """Several rules by number, in the given order, with parent context; unknown numbers
    are skipped."""
    with connect() as conn:
        return _fetch(conn, numbers, _season(season))


def get_subsections(number: str, season: str | None = None) -> list[Rule]:
    """All subsections of a top-level rule, in order."""
    with connect() as conn:
        return _subsections(conn, number, _season(season))


def _subsections(conn: psycopg.Connection, number: str, season: str) -> list[Rule]:
    rows = conn.execute(
        f"""
        SELECT {_COLUMNS} FROM rules
        WHERE season = %s AND parent_number = %s
        ORDER BY string_to_array(rule_number, '.')::int[]
        """,
        (season, number),
    ).fetchall()
    return _with_parents(conn, [_rule(r) for r in rows], season)


def _fetch(conn: psycopg.Connection, numbers: list[str], season: str) -> list[Rule]:
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM rules WHERE season = %s AND rule_number = ANY(%s)",
        (season, numbers),
    ).fetchall()
    by_number = {r[0]: _rule(r) for r in rows}
    return _with_parents(conn, [by_number[n] for n in numbers if n in by_number], season)


def _rule(row: tuple) -> Rule:
    return Rule(rule_number=row[0], parent_number=row[1], title=row[2], section=row[3], text=row[4])


def _with_parents[R: Rule](conn: psycopg.Connection, rules: list[R], season: str) -> list[R]:
    parents = {r.parent_number for r in rules if r.parent_number}
    if parents:
        rows = conn.execute(
            "SELECT rule_number, title, text FROM rules"
            " WHERE season = %s AND rule_number = ANY(%s)",
            (season, list(parents)),
        ).fetchall()
        ctx = {n: ParentContext(rule_number=n, title=t, text=x) for n, t, x in rows}
        for r in rules:
            if r.parent_number:
                r.parent = ctx.get(r.parent_number)
    return rules


# --- search ------------------------------------------------------------------


def keyword_search(
    conn: psycopg.Connection, query: str, season: str, limit: int
) -> list[tuple[str, float]]:
    """Full-text search with OR semantics: a natural-language question rarely has every
    one of its words in a single rule, so any matching term counts and ts_rank orders."""
    return conn.execute(
        f"""
        WITH q AS (
            SELECT NULLIF(replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | '), '')
                   ::tsquery AS tsq
        )
        SELECT rule_number, ts_rank(tsv, q.tsq) AS score
        FROM rules, q
        WHERE {_SEARCHABLE} AND tsv @@ q.tsq
        ORDER BY score DESC, rule_number
        LIMIT %(limit)s
        """,
        {"query": query, "season": season, "limit": limit},
    ).fetchall()


def vector_search(
    conn: psycopg.Connection, query: str, season: str, limit: int
) -> list[tuple[str, float]]:
    """Cosine similarity between the query embedding and each row's embedding."""
    register_vector(conn)
    vec = embed_query(query)
    return conn.execute(
        f"""
        SELECT rule_number, 1 - (embedding <=> %(vec)s) AS score
        FROM rules
        WHERE {_SEARCHABLE} AND embedding_model = %(model)s
        ORDER BY embedding <=> %(vec)s
        LIMIT %(limit)s
        """,
        {"vec": vec, "season": season, "model": get_settings().embedding_model, "limit": limit},
    ).fetchall()


def rrf(*rankings: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, number in enumerate(ranking, start=1):
            scores[number] = scores.get(number, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def search(
    query: str,
    mode: Mode = "hybrid",
    k: int = TOP_K,
    season: str | None = None,
    explicit_rules: list[str] | None = None,
) -> list[Hit]:
    """Top-k rules for `query`. Explicitly named rules (e.g. "56.2", or "56" for all of
    its subsections) come first, by exact lookup; search fills the remaining slots."""
    season = _season(season)
    with connect() as conn:
        exact = _explicit(conn, explicit_rules or [], season)[:k]

        kw = keyword_search(conn, query, season, CANDIDATES) if mode != "vector" else []
        vec = vector_search(conn, query, season, CANDIDATES) if mode != "keyword" else []
        kw_rank = {n: i for i, (n, _) in enumerate(kw, start=1)}
        vec_rank = {n: i for i, (n, _) in enumerate(vec, start=1)}
        if mode == "keyword":
            ranked = kw
        elif mode == "vector":
            ranked = vec
        else:
            ranked = rrf([n for n, _ in kw], [n for n, _ in vec])

        seen = {r.rule_number for r in exact}
        picked = [(n, s) for n, s in ranked if n not in seen][: k - len(exact)]
        rules = {r.rule_number: r for r in _fetch(conn, [n for n, _ in picked], season)}

    hits = [Hit(**r.model_dump(), score=1.0) for r in exact]
    hits += [
        Hit(
            **rules[n].model_dump(),
            score=s,
            keyword_rank=kw_rank.get(n),
            vector_rank=vec_rank.get(n),
        )
        for n, s in picked
    ]
    return hits


def _explicit(conn: psycopg.Connection, numbers: list[str], season: str) -> list[Rule]:
    out: list[Rule] = []
    for number in dict.fromkeys(numbers):
        rule = _fetch(conn, [number], season)
        if not rule:
            continue
        if rule[0].parent_number is None and not rule[0].text:
            out += _subsections(conn, number, season)  # a whole rule: its subsections
        else:
            out += rule
    return out
