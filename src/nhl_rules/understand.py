"""Understand stage (LLM call 1): rewrite fan language into rulebook terms, pull out
explicit rule numbers, and classify the question."""

import json
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nhl_rules.config import get_settings
from nhl_rules.db import connect
from nhl_rules.llm import Usage, client

QuestionType = Literal["situation", "definition", "judgment", "out_of_scope"]
RULE_NUMBER = re.compile(r"^\d{1,2}(\.\d{1,2})?$")


class Understanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(
        description="The question restated in NHL rulebook terminology, for searching the "
        "rulebook. Keep the details the user gave; map slang to rulebook terms."
    )
    explicit_rules: list[str] = Field(
        description='Rule numbers the user named explicitly, e.g. ["56", "74.1"]. Empty if none.'
    )
    question_type: QuestionType


SYSTEM_PROMPT = """\
You are the first stage of an NHL rules explainer for new fans. A later stage searches \
the official NHL rulebook with your output and explains the relevant rules, so your job \
is to make the question searchable and to classify it. You do not answer the question.

Question types:
- situation: describes something that happened in a play or game and asks why or what \
it was ("why did he get 2 minutes for...").
- definition: asks what a term or rule means, or how something works in general.
- judgment: asks whether a referee's call was right, wrong, fair or missed. The tool \
explains the rule but never rules on the call, so classify these as judgment even when \
they also describe a play.
- out_of_scope: anything that is not about the NHL's playing rules: scores, results, \
schedules, standings, statistics, trades, contracts and the salary cap, player news, \
opinions about players or teams, requests for video, and the rules of other leagues or \
levels of hockey.

rewritten_query: rewrite the question as a short search query in the rulebook's own \
language. Fans use slang and describe plays loosely; the rulebook uses formal terms \
("too many men" -> "too many men on the ice", "the box" -> "penalty bench"). Use the \
glossary below where it applies, and your own knowledge of hockey and the NHL rulebook \
where it doesn't. Keep the specifics of the play (who did what, with what, where on the \
ice). When a description could fit more than one infraction, name each plausible one. \
Don't add details the user didn't give. For out_of_scope questions, repeat the question.

explicit_rules: rule numbers the user wrote themselves ("rule 56", "56.2"), as bare \
numbers. Don't add rule numbers you inferred.

Fan glossary (fan phrase -> rulebook term):
"""


@lru_cache
def glossary_text() -> str:
    """The glossary as prompt lines. Stable order keeps the prompt cacheable."""
    with connect() as conn:
        rows = conn.execute("SELECT term, canonical_term FROM glossary ORDER BY term").fetchall()
    return "\n".join(f"- {term} -> {canonical}" for term, canonical in rows)


def understand(question: str) -> Understanding:
    return understand_with_usage(question)[0]


def understand_with_usage(question: str) -> tuple[Understanding, Usage]:
    settings = get_settings()
    response = client().beta.messages.create(
        model=settings.llm_model,
        max_tokens=4000,
        # Cache breakpoint on the system prompt: it's identical across questions, while
        # the question (the last block) differs every time.
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT + glossary_text(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
        output_config={
            "effort": settings.understand_effort,
            "format": {"type": "json_schema", "schema": _schema()},
        },
        # On a safety-classifier refusal, re-run on a fallback model chosen by the API.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise UnderstandError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise UnderstandError("response hit max_tokens")
    text = next(b.text for b in response.content if b.type == "text")
    result = Understanding.model_validate(json.loads(text))
    # Defensive: keep only well-formed rule numbers, deduplicated.
    result.explicit_rules = list(
        dict.fromkeys(r.strip() for r in result.explicit_rules if RULE_NUMBER.match(r.strip()))
    )
    return result, Usage.of(response)


class UnderstandError(RuntimeError):
    pass


@lru_cache
def _schema() -> dict:
    schema = Understanding.model_json_schema()
    schema.pop("title", None)
    for prop in schema["properties"].values():
        prop.pop("title", None)
    return schema
