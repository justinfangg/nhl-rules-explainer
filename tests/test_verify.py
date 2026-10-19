from nhl_rules.answer import GeneratedAnswer
from nhl_rules.verify import verify

RULES = {
    "55.1": "Hooking is the act of using the stick in a manner that enables a player to "
    "restrain an opponent.\nWhen a player is checking another in such a way that there is "
    "only stick-to-stick contact, such action is not to be penalized as hooking.",
    "54.2": "A player is not permitted to hold an opponent’s stick.",
}


def answer(*citations, alt=()):
    def cite(n, e):
        return {"number": n, "title": "t", "excerpt": e}

    return GeneratedAnswer.model_validate(
        {
            "summary": "s",
            "primary": {"rules": [cite(*c) for c in citations], "explanation": "e"},
            "alternatives": [
                {"condition": "if", "rules": [cite(*c)], "explanation": "e"} for c in alt
            ],
            "terms": [],
            "caveats": [],
        }
    )


def test_exact_quote_passes():
    assert (
        verify(answer(("55.1", "using the stick in a manner that enables a player")), RULES) == []
    )


def test_quote_across_paragraph_break_and_typography_passes():
    a = answer(
        ("55.1", "restrain an opponent. When a player is checking another"),
        alt=[("54.2", "A player is not permitted to hold an opponent's stick.")],
    )
    assert verify(a, RULES) == []


def test_paraphrase_fails():
    errors = verify(answer(("55.1", "Hooking means using your stick to restrain someone")), RULES)
    assert len(errors) == 1 and "word-for-word" in errors[0]


def test_rule_not_retrieved_fails_including_alternatives():
    a = answer(("55.1", "restrain an opponent"), alt=[("56.1", "anything")])
    errors = verify(a, RULES)
    assert len(errors) == 1 and "56.1" in errors[0]


def test_quote_from_a_different_rule_fails():
    errors = verify(answer(("54.2", "restrain an opponent")), RULES)
    assert len(errors) == 1


def test_ellipsis_elision_fails():
    assert verify(answer(("55.1", "Hooking is the act ... restrain an opponent")), RULES)
