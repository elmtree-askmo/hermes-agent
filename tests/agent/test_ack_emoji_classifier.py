"""Tests for the acknowledgment-reaction emoji classifier.

A narrow auxiliary LLM that picks a single warmth-reaction emoji for the
user's turn (or null). Runs in the gateway's on_processing_complete hook,
NOT on the Coach critical path. Mirrors the turn_intent_detector contract:
closed-set output, tolerant parse, silent failure -> null.

Signature is user-text-only: the simulation's emoji choice depends almost
entirely on the user's answer, and "is this a response to Coach" is gated
deterministically in the adapter (not fed to the classifier), so no prior
Coach turn is passed in.
"""

from __future__ import annotations

import pytest

from agent import ack_emoji_classifier as aec


def _fake_response(content: str):
    """Stub OpenAI-style response object (same shape as detector tests)."""
    class _FakeMsg:
        pass
    class _FakeChoice:
        pass
    class _FakeResponse:
        pass
    msg = _FakeMsg()
    msg.content = content
    choice = _FakeChoice()
    choice.message = msg
    resp = _FakeResponse()
    resp.choices = [choice]
    return resp


class TestDetectAckEmoji:
    def test_fire_for_strong_exclusion(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": "fire"}'),
            raising=False,
        )
        result = aec.detect_ack_emoji(
            "scheduling posts someone else wrote all day"
        )
        assert result["ack_emoji"] == "fire"
        assert result["checked"] is True

    def test_muscle_for_lets_go(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": "muscle"}'),
            raising=False,
        )
        result = aec.detect_ack_emoji("ok lets go")
        assert result["ack_emoji"] == "muscle"

    def test_raised_hands_for_positive_affect(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": "raised_hands"}'),
            raising=False,
        )
        result = aec.detect_ack_emoji("chicago but open to nyc or la")
        assert result["ack_emoji"] == "raised_hands"

    def test_thumbsup_for_crisp_answer(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": "thumbsup"}'),
            raising=False,
        )
        result = aec.detect_ack_emoji("tomorrow")
        assert result["ack_emoji"] == "thumbsup"

    def test_thumbsup_for_decided_preference_with_mild_qualifier(
        self, monkeypatch
    ):
        """thumbsup is the workhorse for a deliberate answer that resolves
        Coach's question — a clear pick / stated preference — even when it
        carries a mild self-qualifier. Such turns must NOT be pulled into
        fire just because they mention a dislike in passing. (Closed-set
        parse only; the tone judgment itself lives in the LLM prompt and is
        validated live, not unit-tested.)"""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": "thumbsup"}'),
            raising=False,
        )
        result = aec.detect_ack_emoji(
            "marketing strategy was good but I'm not a data person"
        )
        assert result["ack_emoji"] == "thumbsup"

    def test_null_when_no_signal(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": null}'),
            raising=False,
        )
        result = aec.detect_ack_emoji("what does the team do overnight?")
        assert result["ack_emoji"] is None
        assert result["checked"] is True

    def test_invalid_emoji_falls_to_null(self, monkeypatch):
        """An out-of-set emoji name must be rejected, not passed through."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response('{"ack_emoji": "tada"}'),
            raising=False,
        )
        result = aec.detect_ack_emoji("woohoo")
        assert result["ack_emoji"] is None

    def test_empty_user_message_skipped_no_llm(self, monkeypatch):
        called = {"aux": False}
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: (called.__setitem__("aux", True), None)[1],
            raising=False,
        )
        result = aec.detect_ack_emoji("")
        assert result["ack_emoji"] is None
        assert result["checked"] is False
        assert called["aux"] is False

    def test_aux_failure_silent(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("network down")
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _boom, raising=False
        )
        result = aec.detect_ack_emoji("tomorrow")
        assert result["ack_emoji"] is None
        assert result["checked"] is False

    def test_aux_garbage_json_silent(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response("not json at all"),
            raising=False,
        )
        result = aec.detect_ack_emoji("tomorrow")
        assert result["ack_emoji"] is None
        assert result["checked"] is False


class TestEmojiToSlackName:
    """The classifier returns logical names; Slack reactions.add needs the
    Slack short-name. Map must cover the full closed set."""

    def test_maps_all_closed_set_names(self):
        assert aec.slack_reaction_name("fire") == "fire"
        assert aec.slack_reaction_name("muscle") == "muscle"
        assert aec.slack_reaction_name("raised_hands") == "raised_hands"
        assert aec.slack_reaction_name("thumbsup") == "thumbsup"

    def test_unknown_name_returns_none(self):
        assert aec.slack_reaction_name("tada") is None
        assert aec.slack_reaction_name(None) is None
