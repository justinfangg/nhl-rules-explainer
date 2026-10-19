from fastapi.testclient import TestClient

from nhl_rules.answer import Answer
from nhl_rules.api import main
from nhl_rules.understand import UnderstandError


def test_ask_returns_answer(monkeypatch):
    answer = Answer(
        status="out_of_scope", summary="I only explain rules.", question_type="out_of_scope"
    )
    monkeypatch.setattr(main.pipeline, "ask", lambda q, debug=False: answer)
    resp = TestClient(main.app).post("/ask", json={"question": "who won?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "out_of_scope"


def test_ask_maps_model_failures_to_502(monkeypatch):
    def boom(q, debug=False):
        raise UnderstandError("model refused")

    monkeypatch.setattr(main.pipeline, "ask", boom)
    resp = TestClient(main.app).post("/ask", json={"question": "q"})
    assert resp.status_code == 502


def test_ask_validates_question():
    assert TestClient(main.app).post("/ask", json={"question": ""}).status_code == 422


def test_database_outage_is_503(monkeypatch):
    import psycopg

    def down(number):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main.retrieval, "get_rule", down)
    resp = TestClient(main.app).get("/rules/55.1")
    assert resp.status_code == 503
    assert "Database unavailable" in resp.json()["detail"]
