"""Database access."""

import psycopg

from nhl_rules.config import get_settings


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, **kwargs)


def check_database() -> tuple[bool, str]:
    """Return (ok, detail): the database is reachable and has pgvector installed."""
    try:
        with connect(connect_timeout=3) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
    except psycopg.Error as e:
        return False, f"cannot connect: {str(e).strip().splitlines()[0]}"
    if row is None:
        return False, "connected, but the pgvector extension is not installed"
    return True, f"connected, pgvector {row[0]}"


def check_rules(season: str, embedding_model: str) -> tuple[bool, str]:
    """Return (ok, detail): `season` is ingested and embedded with `embedding_model`."""
    try:
        with connect(connect_timeout=3) as conn:
            total, embedded = conn.execute(
                """
                SELECT count(*), count(*) FILTER (WHERE embedding_model = %s)
                FROM rules WHERE season = %s
                """,
                (embedding_model, season),
            ).fetchone()
    except psycopg.errors.UndefinedTable:
        return False, "rules table missing; run `nhl ingest`"
    except psycopg.Error as e:
        return False, f"query failed: {str(e).strip().splitlines()[0]}"
    if total == 0:
        return False, f"no rules loaded for season {season}; run `nhl ingest`"
    if embedded < total:
        return False, (
            f"{total - embedded} of {total} rows not embedded with {embedding_model}; "
            "re-run `nhl ingest`"
        )
    return True, f"{total} rows for season {season}, embedded with {embedding_model}"
