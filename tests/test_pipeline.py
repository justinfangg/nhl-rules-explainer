import pytest

from nhl_rules import pipeline
from nhl_rules.answer import GeneratedAnswer
from nhl_rules.llm import Usage
from nhl_rules.retrieval import Hit
from nhl_rules.understand import Understanding

HIT = Hit(
    rule_number="74.1",
    parent_number="74",
    title="Too Many Men on the Ice",
    section="Other Infractions",
    text="Players may be changed at any time during the play from the players’ bench.",
    score=0.03,
)


def generated(excerpt):
    return GeneratedAnswer.model_validate(
        {
            "summary": "That was too many men on the ice.",
            "primary": {
                "rules": [{"number": "74.1", "title": "model's title", "excerpt": excerpt}],
                "explanation": "e",
            },
            "alternatives": [],
            "terms": [],
            "caveats": ["I can't see the play."],
        }
    )


class FakeGeneration:
    def __init__(self, outputs):
        self.outputs, self.revisions = list(outputs), []
        self.usage = Usage(output_tokens=100)

    def __call__(self, question, question_type, rules):
        self.context_rules = rules
        return self

    def first(self):
        return self.outputs.pop(0)

    def revise(self, errors):
        self.revisions.append(errors)
        return self.outputs.pop(0)


@pytest.fixture
def setup(monkeypatch):
    def install(question_type, outputs):
        u = Understanding(rewritten_query="q", explicit_rules=[], question_type=question_type)
        monkeypatch.setattr(pipeline, "understand_with_usage", lambda q: (u, Usage()))
        monkeypatch.setattr(pipeline.retrieval, "search", lambda *a, **k: [HIT])
        gen = FakeGeneration(outputs)
        monkeypatch.setattr(pipeline, "Generation", gen)
        return gen

    return install


GOOD = "Players may be changed at any time during the play"
BAD = "Teams can change players whenever they like"


def test_out_of_scope_is_deflected_without_generation(setup):
    gen = setup("out_of_scope", [])
    a = pipeline.ask("who won last night", debug=True)
    assert a.status == "out_of_scope" and a.primary is None
    assert a.debug.retrieved == [] and gen.revisions == []


def test_verified_answer_gets_canonical_titles(setup):
    setup("situation", [generated(GOOD)])
    a = pipeline.ask("q", debug=True)
    assert a.status == "answered"
    assert a.primary.rules[0].title == "Too Many Men on the Ice"
    assert [att.passed for att in a.debug.verification] == [True]


def test_failed_verification_retries_once_with_errors(setup):
    gen = setup("situation", [generated(BAD), generated(GOOD)])
    a = pipeline.ask("q", debug=True)
    assert a.status == "answered"
    assert len(gen.revisions) == 1 and "word-for-word" in gen.revisions[0][0]
    assert [att.passed for att in a.debug.verification] == [False, True]


def test_two_failures_fall_back_to_retrieved_rules(setup):
    setup("judgment", [generated(BAD), generated(BAD)])
    a = pipeline.ask("q", debug=True)
    assert a.status == "unverified" and a.primary is None
    assert [r.number for r in a.fallback_rules] == ["74.1"]
    assert [att.passed for att in a.debug.verification] == [False, False]
