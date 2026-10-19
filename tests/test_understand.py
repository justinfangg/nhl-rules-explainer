import json
from types import SimpleNamespace

import pytest

from nhl_rules import understand as und


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=None,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=json.dumps(self.payload)),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=40,
                cache_read_input_tokens=2000,
                cache_creation_input_tokens=0,
            ),
        )


@pytest.fixture
def fake(monkeypatch):
    def install(payload, stop_reason="end_turn"):
        messages = FakeMessages(payload, stop_reason)
        client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
        monkeypatch.setattr(und, "client", lambda: client)
        monkeypatch.setattr(
            und, "glossary_text", lambda: "- too many men -> too many men on the ice"
        )
        return messages

    return install


PAYLOAD = {
    "rewritten_query": "too many men on the ice bench minor penalty",
    "explicit_rules": ["74", " 74 ", "rule 56", "56.2"],
    "question_type": "situation",
}


def test_understand_parses_and_cleans_explicit_rules(fake):
    fake(PAYLOAD)
    result, usage = und.understand_with_usage("why was it too many men, see rule 74")
    assert result.question_type == "situation"
    assert result.explicit_rules == ["74", "56.2"]  # deduplicated, malformed dropped
    assert usage.cache_read_tokens == 2000


def test_request_caches_system_prompt_and_constrains_output(fake):
    messages = fake(PAYLOAD)
    und.understand("q")
    req = messages.calls[0]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "too many men on the ice" in req["system"][0]["text"]
    assert "cache_control" not in req  # not on the per-question last block
    schema = req["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"rewritten_query", "explicit_rules", "question_type"}
    assert req["fallbacks"] == "default"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_responses_raise(fake, stop_reason):
    fake(PAYLOAD, stop_reason)
    with pytest.raises(und.UnderstandError):
        und.understand("q")
