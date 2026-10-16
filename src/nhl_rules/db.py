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
