"""Faithfulness judge (eval only): does an answer stay within the retrieved rule text?"""

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nhl_rules.answer import Answer
from nhl_rules.config import get_settings
from nhl_rules.generate import render_rules
from nhl_rules.llm import Usage, client
from nhl_rules.retrieval import Rule


class Faithfulness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unsupported_claims: list[str] = Field(
        description="Each claim about the rules that the provided rule text does not "
        "support or that contradicts it, quoted or closely paraphrased, with a few words "
        "on why. Empty if there are none."
    )
    score: Literal[1, 2, 3, 4, 5]
    rationale: str = Field(description="One or two sentences justifying the score.")


SYSTEM_PROMPT = """\
You grade answers from an NHL rules explainer for faithfulness: whether everything the \
answer says about the rules is supported by the rule text it was given. You are not \
grading helpfulness, style, or whether you would have answered differently.

You get the fan's question, the rules that were provided to the explainer, and the \
explainer's answer (summary, explanations, alternatives, term definitions, caveats).

Check every claim about what the rules say, require, allow or result in. A claim is \
supported if the provided text states it or directly implies it. Plain-language \
restatements are fine. General hockey context that doesn't assert anything about the \
rules (for example, that the penalty bench is often called "the box") is fine. A claim \
about the rules that the provided text doesn't support is unsupported even if it is \
true in the real rulebook: the point is whether the answer stayed within what it was \
given. Statements about what the explainer can't know (caveats) don't need support.

Score:
5 - Every claim about the rules is supported by the provided text.
4 - One minor unsupported detail that doesn't change the answer (a small \
extrapolation); nothing contradicts the text.
3 - At least one substantive claim about the rules is unsupported, but the main answer \
is supported.
2 - The main explanation relies on unsupported claims, or a claim contradicts the text.
1 - The answer is mostly unsupported by, or contradicts, the provided text.

List the unsupported claims first, then score.
"""


def judge(question: str, rules: list[Rule], answer: Answer) -> tuple[Faithfulness, Usage]:
    settings = get_settings()
    shown = answer.model_dump(
        include={"summary", "primary", "alternatives", "terms", "caveats"}, mode="json"
    )
    content = (
        f"<question>{question}</question>\n\n{render_rules(rules)}\n\n"
        f"<answer>\n{json.dumps(shown, indent=1, ensure_ascii=False)}\n</answer>"
    )
    response = client().beta.messages.create(
        model=settings.judge_model,
        max_tokens=8000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={
            "effort": settings.judge_effort,
            "format": {"type": "json_schema", "schema": _schema()},
        },
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"judge response unusable: stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    return Faithfulness.model_validate(json.loads(text)), Usage.of(response)


@lru_cache
def _schema() -> dict:
    schema = Faithfulness.model_json_schema()
    schema.pop("title", None)
    for prop in schema["properties"].values():
        prop.pop("title", None)
    return schema
