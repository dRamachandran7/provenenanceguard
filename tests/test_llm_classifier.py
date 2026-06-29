"""Tests for Signal 1: the LLM holistic classifier.

We inject a fake Groq-compatible client so these run offline and
deterministically — no API key or network required.
"""

import json
from types import SimpleNamespace

import pytest

from provenanceguard.signals.llm_classifier import (
    NEUTRAL_FALLBACK,
    SIGNAL_NAME,
    classify_with_llm,
)


class FakeClient:
    """Minimal stand-in for groq.Groq exposing chat.completions.create."""

    def __init__(self, content, *, raise_exc=None):
        self._content = content
        self._raise_exc = raise_exc
        self.calls = []

        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _payload(prob, reasoning="because"):
    return json.dumps({"ai_probability": prob, "reasoning": reasoning})


def test_high_ai_probability_is_passed_through():
    client = classify_with_llm(
        "some smooth generic text", client=FakeClient(_payload(0.92))
    )
    assert client.name == SIGNAL_NAME
    assert client.ok is True
    assert client.score == pytest.approx(0.92)
    assert client.reasoning == "because"


def test_low_ai_probability_is_passed_through():
    result = classify_with_llm("gritty human poem", client=FakeClient(_payload(0.07)))
    assert result.ok is True
    assert result.score == pytest.approx(0.07)


def test_score_is_clamped_into_unit_interval():
    # Model misbehaves and returns an out-of-range probability.
    result = classify_with_llm("x", client=FakeClient(_payload(1.8)))
    assert result.score == 1.0


def test_empty_text_short_circuits_without_calling_model():
    fake = FakeClient(_payload(0.9))
    result = classify_with_llm("   ", client=fake)
    assert result.ok is False
    assert result.score == NEUTRAL_FALLBACK
    assert fake.calls == []  # model was never invoked


def test_malformed_json_falls_back_to_neutral():
    result = classify_with_llm("text", client=FakeClient("not json at all"))
    assert result.ok is False
    assert result.score == NEUTRAL_FALLBACK
    assert "unavailable" in result.reasoning.lower()


def test_missing_field_falls_back_to_neutral():
    result = classify_with_llm("text", client=FakeClient(json.dumps({"foo": 1})))
    assert result.ok is False
    assert result.score == NEUTRAL_FALLBACK


def test_api_exception_is_caught_and_reported():
    client = FakeClient(None, raise_exc=RuntimeError("rate limited"))
    result = classify_with_llm("text", client=client)
    assert result.ok is False
    assert result.score == NEUTRAL_FALLBACK
    assert "rate limited" in result.details["error"]


def test_request_uses_json_response_format_and_zero_temperature():
    fake = FakeClient(_payload(0.5))
    classify_with_llm("text", client=fake)
    sent = fake.calls[0]
    assert sent["temperature"] == 0.0
    assert sent["response_format"] == {"type": "json_object"}
    # System + user messages are both present.
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user"]
