"""The hand-written eval set (eval/cases.json)."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

DEFAULT_PATH = Path("eval/cases.json")

QuestionType = Literal["situation", "definition", "judgment", "out_of_scope"]


class EvalCase(BaseModel):
    id: str
    question: str
    type: QuestionType
    # plain = rulebook language, slang = fan language, vague = several rules could apply
    style: Literal["plain", "slang", "vague"]
    expected_rules: list[str]
    acceptable_rules: list[str] = []
    notes: str = ""

    @model_validator(mode="after")
    def rules_match_type(self) -> "EvalCase":
        if self.type == "out_of_scope" and (self.expected_rules or self.acceptable_rules):
            raise ValueError(f"{self.id}: out_of_scope cases must not list rules")
        if self.type != "out_of_scope" and not self.expected_rules:
            raise ValueError(f"{self.id}: expected_rules is empty")
        return self

    @property
    def all_rules(self) -> list[str]:
        return self.expected_rules + self.acceptable_rules


def read_cases(path: Path = DEFAULT_PATH) -> list[EvalCase]:
    cases = [EvalCase(**c) for c in json.loads(path.read_text())]
    seen: set[str] = set()
    dupes = [c.id for c in cases if c.id in seen or seen.add(c.id)]
    if dupes:
        raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
    return cases


def missing_rules(cases: list[EvalCase], known: set[str]) -> dict[str, list[str]]:
    """Rule numbers each case cites that aren't in `known`."""
    missing = {c.id: [n for n in c.all_rules if n not in known] for c in cases}
    return {cid: nums for cid, nums in missing.items() if nums}


def rule_matches(retrieved: str, expected: str) -> bool:
    """A retrieved rule counts for an expected one if it is the same rule, its parent
    rule, or one of its subsections ('55' and '55.2' match each other)."""
    return (
        retrieved == expected
        or expected.startswith(retrieved + ".")
        or retrieved.startswith(expected + ".")
    )
