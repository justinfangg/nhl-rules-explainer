"""FastAPI service."""

from fastapi import FastAPI, Response
from pydantic import BaseModel

from nhl_rules.db import check_database

app = FastAPI(title="NHL Rules Explainer", version="0.1.0")


class Check(BaseModel):
    ok: bool
    detail: str


class Health(BaseModel):
    status: str  # "ok" or "unhealthy"
    checks: dict[str, Check]


@app.get("/health", response_model=Health)
def health(response: Response) -> Health:
    db_ok, db_detail = check_database()
    checks = {"database": Check(ok=db_ok, detail=db_detail)}
    healthy = all(c.ok for c in checks.values())
    if not healthy:
        response.status_code = 503
    return Health(status="ok" if healthy else "unhealthy", checks=checks)
