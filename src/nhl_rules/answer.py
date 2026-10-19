"""The answer object: what the model generates, and what the API returns."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nhl_rules.understand import QuestionType, Understanding


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuleCitation(_Strict):
    number: str = Field(description='Rule number exactly as given, e.g. "55.1".')
    title: str
    excerpt: str = Field(
        description="A short quote copied word for word from that rule's text: a phrase or "
        "a sentence, never a paraphrase."
    )


class Primary(_Strict):
    rules: list[RuleCitation] = Field(min_length=1)
    explanation: str = Field(description="Plain-English explanation tied to the question.")


class Alternative(_Strict):
    condition: str = Field(description='What would make this apply, e.g. "If he used his hand".')
    rules: list[RuleCitation] = Field(min_length=1)
    explanation: str


class Term(_Strict):
    term: str
    definition: str


class GeneratedAnswer(_Strict):
    """The structured output of the Generate stage."""

    summary: str = Field(description="One plain sentence that answers the question.")
    primary: Primary
    alternatives: list[Alternative]
    terms: list[Term]
    caveats: list[str]

    def citations(self) -> list[RuleCitation]:
        return self.primary.rules + [r for a in self.alternatives for r in a.rules]


class RuleRef(BaseModel):
    number: str
    title: str


class RetrievedRule(RuleRef):
    score: float
    keyword_rank: int | None
    vector_rank: int | None


class Attempt(BaseModel):
    passed: bool
    errors: list[str]


class Debug(BaseModel):
    understanding: Understanding
    retrieved: list[RetrievedRule]
    verification: list[Attempt]  # one per generation attempt
    usage: dict[str, int]
    seconds: float


class Answer(BaseModel):
    """What POST /ask returns."""

    # answered: verified explanation; out_of_scope: deflected at Route;
    # unverified: no explanation passed verification, so only the retrieved rules are shown.
    status: Literal["answered", "out_of_scope", "unverified"]
    summary: str
    primary: Primary | None = None
    alternatives: list[Alternative] = []
    terms: list[Term] = []
    caveats: list[str] = []
    question_type: QuestionType
    fallback_rules: list[RuleRef] = []  # set only when status is "unverified"
    debug: Debug | None = None
