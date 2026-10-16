"""Fan-language glossary: a hand-curated JSON file loaded into the `glossary` table."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from nhl_rules.db import connect

DEFAULT_PATH = Path("data/glossary.json")


class GlossaryTerm(BaseModel):
    term: str
    canonical_term: str
    rule_numbers: list[str] = Field(min_length=1)

    @field_validator("term")
    @classmethod
    def normalize(cls, v: str) -> str:
        return " ".join(v.lower().split())


def read_terms(path: Path = DEFAULT_PATH) -> list[GlossaryTerm]:
    terms = [GlossaryTerm(**t) for t in json.loads(path.read_text())]
    seen: set[str] = set()
    dupes = [t.term for t in terms if t.term in seen or seen.add(t.term)]
    if dupes:
        raise ValueError(f"duplicate glossary terms: {', '.join(dupes)}")
    return terms


def missing_rules(terms: list[GlossaryTerm], season: str) -> dict[str, list[str]]:
    """Rule numbers referenced by each term that aren't in `rules` for `season`."""
    with connect() as conn:
        known = {
            r[0] for r in conn.execute("SELECT rule_number FROM rules WHERE season = %s", (season,))
        }
    missing = {t.term: [n for n in t.rule_numbers if n not in known] for t in terms}
    return {term: nums for term, nums in missing.items() if nums}


def load(terms: list[GlossaryTerm]) -> int:
    """Replace the glossary table's contents with `terms`."""
    with connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM glossary")
        cur.executemany(
            "INSERT INTO glossary (term, rule_numbers, canonical_term) VALUES (%s, %s, %s)",
            [(t.term, t.rule_numbers, t.canonical_term) for t in terms],
        )
    return len(terms)
