import pytest
from fastapi.testclient import TestClient

from nhl_rules import retrieval
from nhl_rules.api import main
from nhl_rules.config import get_settings
from nhl_rules.db import check_rules


def test_rrf_rewards_agreement_between_rankings():
    fused = retrieval.rrf(["a", "b", "c"], ["b", "c", "d"])
    assert [n for n, _ in fused] == ["b", "c", "a", "d"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rule_endpoint_404(monkeypatch):
    monkeypatch.setattr(main.retrieval, "get_rule", lambda number: None)
    assert TestClient(main.app).get("/rules/999").status_code == 404


def test_search_endpoint_validates_mode():
    assert (
        TestClient(main.app).get("/search", params={"q": "icing", "mode": "fuzzy"}).status_code
        == 422
    )


# --- against the loaded database (skipped when it isn't available) ----------


@pytest.fixture(scope="module")
def loaded_db():
    s = get_settings()
    ok, detail = check_rules(s.rulebook_season, s.embedding_model)
    if not ok:
        pytest.skip(f"rules not loaded: {detail}")


@pytest.mark.parametrize("mode", retrieval.MODES)
def test_search_finds_too_many_men(loaded_db, mode):
    hits = retrieval.search("too many men on the ice", mode, k=5)
    assert len(hits) == 5
    assert any(h.rule_number.startswith("74.") for h in hits)
    assert all(h.text for h in hits)  # rule-level title rows are never hits


def test_search_attaches_parent_context(loaded_db):
    hit = next(h for h in retrieval.search("hooking", "keyword") if h.rule_number == "55.1")
    assert hit.parent is not None
    assert (hit.parent.rule_number, hit.parent.title) == ("55", "Hooking")


def test_explicit_rules_come_first(loaded_db):
    hits = retrieval.search("what does this rule say", explicit_rules=["56.2", "74"], k=8)
    assert [h.rule_number for h in hits[:6]] == ["56.2", "74.1", "74.2", "74.3", "74.4", "74.5"]
    assert len(hits) == 8


def test_get_rule_and_subsections(loaded_db):
    rule = retrieval.get_rule("Glossary: Torso")
    assert rule is not None and rule.parent is None
    assert [r.rule_number for r in retrieval.get_subsections("55")][:2] == ["55.1", "55.2"]
    assert retrieval.get_rule("999") is None
