from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from nhl_rules.evals.cases import DEFAULT_PATH, EvalCase, missing_rules, read_cases, rule_matches

ROOT = Path(__file__).parent.parent
PDF = ROOT / "2026-27Rules.pdf"


@pytest.fixture(scope="module")
def cases():
    return read_cases(ROOT / DEFAULT_PATH)


def test_eval_set_size_and_mix(cases):
    assert len(cases) >= 50
    types = Counter(c.type for c in cases)
    styles = Counter(c.style for c in cases if c.type in ("situation", "definition"))
    assert all(types[t] >= 8 for t in ("situation", "definition", "judgment", "out_of_scope"))
    assert all(styles[s] >= 10 for s in ("plain", "slang", "vague"))


def test_case_validation():
    with pytest.raises(ValidationError, match="must not list rules"):
        EvalCase(id="x", question="q", type="out_of_scope", style="plain", expected_rules=["1"])
    with pytest.raises(ValidationError, match="expected_rules is empty"):
        EvalCase(id="x", question="q", type="situation", style="plain", expected_rules=[])


def test_rule_matches():
    assert rule_matches("55", "55.2")  # parent retrieved
    assert rule_matches("55.2", "55")  # subsection of the expected rule
    assert rule_matches("55.2", "55.2")
    assert not rule_matches("5", "55")
    assert not rule_matches("55.1", "55.2")


def test_eval_rules_exist_in_rulebook(cases):
    """Build-step-4 check, against the parsed PDF (the same rows `nhl ingest` loads)."""
    if not PDF.exists():
        pytest.skip(f"{PDF.name} not present")
    from nhl_rules.ingest.parser import parse_pdf

    known = {c.rule_number for c in parse_pdf(PDF)}
    assert missing_rules(cases, known) == {}
