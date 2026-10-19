import json
from types import SimpleNamespace

from nhl_rules import judge as jd
from nhl_rules.answer import Answer
from nhl_rules.retrieval import Rule


def test_judge_request_and_parse(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        payload = {"unsupported_claims": ["minor = two minutes"], "score": 3, "rationale": "r"}
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(jd, "client", lambda: client)
    rule = Rule(
        rule_number="55.2",
        parent_number="55",
        title="Minor Penalty",
        section="Restraining Infractions",
        text="A minor penalty shall be imposed.",
    )
    answer = Answer(
        status="answered",
        summary="Hooking.",
        question_type="situation",
        caveats=["I can't see the play."],
    )

    f, _ = jd.judge("why 2 minutes?", [rule], answer)
    assert f.score == 3 and f.unsupported_claims == ["minor = two minutes"]
    req = calls[0]
    content = req["messages"][0]["content"]
    assert "A minor penalty shall be imposed." in content and "Hooking." in content
    assert "debug" not in content and "fallback_rules" not in content
    schema = req["output_config"]["format"]["schema"]
    assert schema["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]
    assert list(schema["properties"]) == ["unsupported_claims", "score", "rationale"]
