"""FastAPI service."""

from contextlib import asynccontextmanager
from typing import Annotated

import anthropic
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from nhl_rules import pipeline, retrieval
from nhl_rules.answer import Answer
from nhl_rules.config import get_settings
from nhl_rules.db import check_database, check_rules
from nhl_rules.embeddings import load_model
from nhl_rules.generate import GenerateError
from nhl_rules.understand import UnderstandError

# Embedding model status for /health: (ok, detail). Set at startup.
_model_status: tuple[bool, str] = (False, "not loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model at startup so the first search isn't slow.
    global _model_status
    name = get_settings().embedding_model
    try:
        load_model()
        _model_status = (True, f"{name} loaded")
    except Exception as e:  # reported by /health rather than crashing the service
        _model_status = (False, f"{name} failed to load: {type(e).__name__}: {e}")
    yield


app = FastAPI(title="NHL Rules Explainer", version="0.1.0", lifespan=lifespan)


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
    if db_ok:
        settings = get_settings()
        ok, detail = check_rules(settings.rulebook_season, settings.embedding_model)
        checks["rules"] = Check(ok=ok, detail=detail)
    checks["embedding_model"] = Check(ok=_model_status[0], detail=_model_status[1])
    healthy = all(c.ok for c in checks.values())
    if not healthy:
        response.status_code = 503
    return Health(status="ok" if healthy else "unhealthy", checks=checks)


class RuleResponse(BaseModel):
    rule: retrieval.Rule
    # For a whole rule ("56"), its subsections; empty for a subsection or definition.
    subsections: list[retrieval.Rule]


@app.get("/rules/{number}", response_model=RuleResponse)
def get_rule(number: str) -> RuleResponse:
    rule = retrieval.get_rule(number)
    if rule is None:
        raise HTTPException(404, f"No rule {number!r} in season {get_settings().rulebook_season}")
    subsections = retrieval.get_subsections(number) if rule.parent_number is None else []
    return RuleResponse(rule=rule, subsections=subsections)


class SearchResponse(BaseModel):
    query: str
    mode: retrieval.Mode
    hits: list[retrieval.Hit]


@app.get("/search", response_model=SearchResponse)
def search(
    q: Annotated[str, Query(min_length=1, max_length=1000)],
    mode: retrieval.Mode = "hybrid",
    k: Annotated[int, Query(ge=1, le=50)] = retrieval.TOP_K,
) -> SearchResponse:
    return SearchResponse(query=q, mode=mode, hits=retrieval.search(q, mode, k))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    debug: bool = False


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    try:
        return pipeline.ask(req.question, debug=req.debug)
    except (UnderstandError, GenerateError) as e:
        raise HTTPException(502, f"Model response unusable: {e}") from e
    except anthropic.RateLimitError as e:
        raise HTTPException(503, "Rate limited by the model API; try again shortly.") from e
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Model API error ({e.status_code}).") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(502, "Could not reach the model API.") from e
