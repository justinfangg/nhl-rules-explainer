"""Generate stage (LLM call 2): a structured answer that cites only the provided rules
(the retrieved rules plus a fixed set of core penalty definitions)."""

import json
from functools import lru_cache
from xml.sax.saxutils import escape, quoteattr

from nhl_rules.answer import GeneratedAnswer
from nhl_rules.config import get_settings
from nhl_rules.llm import Usage, client
from nhl_rules.retrieval import Rule, get_rules
from nhl_rules.understand import QuestionType

SYSTEM_PROMPT = """\
You explain NHL rules to new fans who don't know the rulebook or its jargon. Each \
request gives you a fan's question and a set of rules retrieved from the official NHL \
rulebook for the current season. You also always have the reference definitions at the \
end of these instructions, which you can cite the same way. Together these are "the \
provided rules". Answer from the provided rules only.

Everything you say about what the rules require, allow or result in (penalty lengths, \
who serves what, what happens next) must come from the provided rules, not from your \
own knowledge of hockey, even when you know it to be true. If part of the question \
isn't covered by the provided rules, say so in caveats instead of filling the gap.

How to answer:
- summary: one plain sentence a new fan understands, that answers the question directly.
- primary: the rule or rules that most likely apply, and an explanation in plain \
English tied to what the fan described.
- alternatives: when the description could fit more than one rule, give each plausible \
alternative with the condition that would make it apply instead ("If he grabbed him \
with his hand instead of his stick"). Make your best guess rather than asking a \
clarifying question. Leave alternatives empty when one rule clearly covers it.
- terms: rulebook terms in your answer that a new fan wouldn't know, each with a short \
definition based on what the provided rules say about it. Leave out a term the provided \
rules don't explain rather than defining it from memory.
- caveats: be honest about limits. You can't see the play, so for a described situation \
say the answer depends on the description. Never claim to know details of a play that \
the fan didn't give. If the provided rules don't cover the question, say so here and \
answer only as far as they do.

Citations:
- Cite only rules from the provided set, using their numbers exactly as given (e.g. \
"55.1", or "Glossary: Attacking Zone"). Never cite a rule that isn't provided, even if \
you know it exists.
- Each excerpt must be copied word for word from the cited rule's text: one contiguous \
phrase or sentence, short, with no ellipses and no changes. Every excerpt is checked \
against the rulebook text, and an answer with a misquoted excerpt is rejected.
- The explanation can paraphrase; only excerpts must be exact.

Judgment questions: when the fan asks whether a referee's call was right, wrong, fair \
or missed, explain the rule and what the official has to judge to make that call. Never \
say whether the call was correct; you only know what the fan described.
"""

# Basic penalty definitions new fans ask about constantly. Always in context (in the cached
# system prompt) so answers can explain and cite them even when retrieval didn't return them.
CORE_DEFINITIONS = ["16.1", "16.2", "17.1", "18.1", "20.1", "21.1", "22.1", "23.1"]

JUDGMENT_NOTE = (
    "This is a judgment question: explain the rule and what the official has to judge. "
    "Do not say whether the call was right or wrong."
)


def render_rules(rules: list[Rule]) -> str:
    """Retrieved rules as XML-ish blocks the model can cite from."""
    blocks = []
    for r in rules:
        attrs = f"number={quoteattr(r.rule_number)} title={quoteattr(r.title)}"
        if r.parent:
            attrs += f" rule={quoteattr(f'Rule {r.parent.rule_number} – {r.parent.title}')}"
        attrs += f" section={quoteattr(r.section)}"
        blocks.append(f"<rule {attrs}>\n{escape(r.text)}\n</rule>")
    return "<rules>\n" + "\n".join(blocks) + "\n</rules>"


def user_message(question: str, question_type: QuestionType, rules: list[Rule]) -> str:
    parts = [
        f"<question>{escape(question)}</question>",
        f"<question_type>{question_type}</question_type>",
    ]
    if question_type == "judgment":
        parts.append(JUDGMENT_NOTE)
    parts.append(render_rules(rules))
    return "\n\n".join(parts)


@lru_cache
def core_rules() -> tuple[Rule, ...]:
    return tuple(get_rules(CORE_DEFINITIONS))


@lru_cache
def system_prompt() -> str:
    return (
        SYSTEM_PROMPT
        + "\nReference definitions (always provided; cite them like any provided rule):\n"
        + render_rules(list(core_rules()))
    )


class GenerateError(RuntimeError):
    pass


class Generation:
    """One conversation with the model: an initial answer, then at most one revision."""

    def __init__(self, question: str, question_type: QuestionType, rules: list[Rule]):
        self.messages: list[dict] = [
            {"role": "user", "content": user_message(question, question_type, rules)}
        ]
        # Everything the model may cite: the retrieved rules plus the reference definitions.
        seen = {r.rule_number for r in rules}
        self.context_rules: list[Rule] = rules + [
            r for r in core_rules() if r.rule_number not in seen
        ]
        self.usage = Usage()

    def first(self) -> GeneratedAnswer:
        return self._call()

    def revise(self, errors: list[str]) -> GeneratedAnswer:
        """Ask for a corrected answer, listing what failed verification."""
        self.messages.append(
            {
                "role": "user",
                "content": "Your answer failed verification:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nReturn a corrected answer. Fix the excerpts by copying them exactly "
                "from the rule text, or cite a different provided rule.",
            }
        )
        return self._call()

    def _call(self) -> GeneratedAnswer:
        settings = get_settings()
        response = client().beta.messages.create(
            model=settings.llm_model,
            max_tokens=16000,
            system=[
                {"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}
            ],
            messages=self.messages,
            output_config={
                "effort": settings.generate_effort,
                "format": {"type": "json_schema", "schema": _schema()},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage = self.usage + Usage.of(response)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise GenerateError(f"unusable response: stop_reason={response.stop_reason}")
        # Append the full response (thinking blocks included) so a revision continues
        # the same conversation unchanged.
        self.messages.append({"role": "assistant", "content": response.content})
        text = next(b.text for b in response.content if b.type == "text")
        return GeneratedAnswer.model_validate(json.loads(text))


@lru_cache
def _schema() -> dict:
    """JSON schema for structured output. Schema titles are dropped (string-valued "title"
    keys; the RuleCitation.title property is a dict and stays), and so is minItems, which
    Pydantic still enforces when parsing."""

    def clean(node):
        if isinstance(node, dict):
            return {
                k: clean(v)
                for k, v in node.items()
                if not (k == "title" and isinstance(v, str)) and k != "minItems"
            }
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(GeneratedAnswer.model_json_schema())
