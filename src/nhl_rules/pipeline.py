"""The six-stage question pipeline: Understand -> Route -> Retrieve -> Generate -> Verify
-> Render (the caller renders the returned Answer)."""

import time
from dataclasses import asdict

from nhl_rules import retrieval
from nhl_rules.answer import Answer, Attempt, Debug, GeneratedAnswer, RetrievedRule, RuleRef
from nhl_rules.generate import Generation
from nhl_rules.llm import Usage
from nhl_rules.understand import understand_with_usage
from nhl_rules.verify import verify

DEFLECTION = (
    "I can only explain the NHL's playing rules (penalties, calls and hockey terms), so I "
    "can't help with scores, stats, schedules, trades or news."
)
UNVERIFIED = (
    "I couldn't produce an explanation I could verify against the rulebook, but these are "
    "the rules that look most relevant."
)


def ask(question: str, debug: bool = False) -> Answer:
    start = time.monotonic()

    # 1. Understand (LLM call 1)
    understanding, usage = understand_with_usage(question)

    # 2. Route
    if understanding.question_type == "out_of_scope":
        answer = Answer(status="out_of_scope", summary=DEFLECTION, question_type="out_of_scope")
        if debug:
            answer.debug = Debug(
                understanding=understanding,
                retrieved=[],
                verification=[],
                usage=asdict(usage),
                seconds=round(time.monotonic() - start, 2),
            )
        return answer

    # 3. Retrieve
    hits = retrieval.search(
        understanding.rewritten_query, "hybrid", explicit_rules=understanding.explicit_rules
    )
    texts = {h.rule_number: h.text for h in hits}
    titles = {h.rule_number: h.title for h in hits}

    # 4-5. Generate (LLM call 2), Verify, at most one revision
    generation = Generation(question, understanding.question_type, hits)
    attempts: list[Attempt] = []
    generated: GeneratedAnswer | None = generation.first()
    for revision in (False, True):
        errors = verify(generated, texts)
        attempts.append(Attempt(passed=not errors, errors=errors))
        if not errors:
            break
        generated = generation.revise(errors) if not revision else None
    usage = usage + generation.usage

    if generated is not None:
        for c in generated.citations():  # canonical titles, not the model's
            c.title = titles[c.number]
        answer = Answer(
            status="answered",
            summary=generated.summary,
            primary=generated.primary,
            alternatives=generated.alternatives,
            terms=generated.terms,
            caveats=generated.caveats,
            question_type=understanding.question_type,
        )
    else:
        answer = Answer(
            status="unverified",
            summary=UNVERIFIED,
            question_type=understanding.question_type,
            fallback_rules=[RuleRef(number=h.rule_number, title=h.title) for h in hits],
        )

    if debug:
        answer.debug = Debug(
            understanding=understanding,
            retrieved=[
                RetrievedRule(
                    number=h.rule_number,
                    title=h.title,
                    score=h.score,
                    keyword_rank=h.keyword_rank,
                    vector_rank=h.vector_rank,
                )
                for h in hits
            ],
            verification=attempts,
            usage=asdict(usage),
            seconds=round(time.monotonic() - start, 2),
        )
    return answer


def ask_with_trace(question: str) -> tuple[Answer, Usage]:
    """For evals: the answer with debug info, plus token usage as a Usage object."""
    answer = ask(question, debug=True)
    return answer, Usage(**answer.debug.usage)
