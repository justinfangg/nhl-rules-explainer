import json
from pathlib import Path

import pytest

from nhl_rules.glossary import DEFAULT_PATH, read_terms

ROOT = Path(__file__).parent.parent
PDF = ROOT / "2026-27Rules.pdf"


def test_read_terms_normalizes_and_rejects_duplicates(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(
        json.dumps([{"term": "  Too  Many Men ", "canonical_term": "x", "rule_numbers": ["74"]}])
    )
    assert read_terms(f)[0].term == "too many men"

    f.write_text(
        json.dumps(
            [
                {"term": "icing", "canonical_term": "icing", "rule_numbers": ["81"]},
                {"term": "Icing", "canonical_term": "icing", "rule_numbers": ["81"]},
            ]
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_terms(f)


def test_seed_glossary_is_valid():
    terms = read_terms(ROOT / DEFAULT_PATH)
    assert len(terms) >= 30
    assert all(t.rule_numbers and t.canonical_term for t in terms)


def test_seed_glossary_rules_exist_in_rulebook():
    """The same check `nhl glossary` runs against the database, against the parsed PDF."""
    if not PDF.exists():
        pytest.skip(f"{PDF.name} not present")
    from nhl_rules.ingest.parser import parse_pdf

    known = {c.rule_number for c in parse_pdf(PDF)}
    missing = {
        t.term: [n for n in t.rule_numbers if n not in known]
        for t in read_terms(ROOT / DEFAULT_PATH)
    }
    assert {k: v for k, v in missing.items() if v} == {}
