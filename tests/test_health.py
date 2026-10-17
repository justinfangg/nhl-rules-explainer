import httpx
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from nhl_rules import cli
from nhl_rules.api import main


def test_health_ok(monkeypatch):
    monkeypatch.setattr(main, "_model_status", (True, "model loaded"))
    monkeypatch.setattr(main, "check_database", lambda: (True, "connected, pgvector 0.8.0"))
    monkeypatch.setattr(main, "check_rules", lambda season, model: (True, "615 rows"))
    resp = TestClient(main.app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert set(resp.json()["checks"]) == {"database", "rules", "embedding_model"}


def test_health_rules_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "check_database", lambda: (True, "connected"))
    monkeypatch.setattr(main, "check_rules", lambda season, model: (False, "no rules loaded"))
    resp = TestClient(main.app).get("/health")
    assert resp.status_code == 503
    assert resp.json()["checks"]["rules"] == {"ok": False, "detail": "no rules loaded"}


def test_health_db_down(monkeypatch):
    monkeypatch.setattr(main, "_model_status", (True, "model loaded"))
    monkeypatch.setattr(main, "check_database", lambda: (False, "cannot connect"))
    resp = TestClient(main.app).get("/health")
    assert resp.status_code == 503
    assert resp.json() == {
        "status": "unhealthy",
        "checks": {
            "database": {"ok": False, "detail": "cannot connect"},
            "embedding_model": {"ok": True, "detail": "model loaded"},
        },
    }


def _fake_get(status_code, body=None, error=None):
    def get(url, **kwargs):
        if error:
            raise error
        return httpx.Response(status_code, json=body)

    return get


def test_cli_health_ok(monkeypatch):
    body = {"status": "ok", "checks": {"database": {"ok": True, "detail": "connected"}}}
    monkeypatch.setattr(cli.httpx, "get", _fake_get(200, body))
    result = CliRunner().invoke(cli.app, ["health"])
    assert result.exit_code == 0
    assert "healthy" in result.output


def test_cli_health_unhealthy(monkeypatch):
    body = {"status": "unhealthy", "checks": {"database": {"ok": False, "detail": "down"}}}
    monkeypatch.setattr(cli.httpx, "get", _fake_get(503, body))
    result = CliRunner().invoke(cli.app, ["health"])
    assert result.exit_code == 1
    assert "unhealthy" in result.output


def test_cli_api_unreachable(monkeypatch):
    monkeypatch.setattr(cli.httpx, "get", _fake_get(0, error=httpx.ConnectError("refused")))
    result = CliRunner().invoke(cli.app, ["health"])
    assert result.exit_code == 1


def test_health_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "_model_status", (False, "not loaded"))
    monkeypatch.setattr(main, "check_database", lambda: (True, "connected"))
    monkeypatch.setattr(main, "check_rules", lambda season, model: (True, "615 rows"))
    resp = TestClient(main.app).get("/health")
    assert resp.status_code == 503
    assert resp.json()["checks"]["embedding_model"]["ok"] is False
