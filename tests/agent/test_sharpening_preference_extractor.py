"""Unit tests for agent.sharpening_preference_extractor (B-0624-04).

The extractor reads an onboarding conversation transcript and returns the
preference axes the user expressed during the sharpening series, as a flat
dict suitable for save_user_profile(preferences=...). It is the shared
extraction core called by BOTH backfill layers:
  - Layer 2 (timely): the gateway post-reply hook, mid-conversation.
  - Layer 1 (consumption-point): run-strategist.sh, before the briefing reads
    the profile.

Core contract (the bug being fixed: Coach acknowledges sharpening answers in
chat but defers / skips save_user_profile, so preferences land null):
  - Given a transcript with sharpening Q&A, return a dict of the axes the user
    stated (location, exclusion, etc.).
  - Given chit-chat / no preference content, return {} (nothing to write).
  - Only emit a key when the value is confident — an uncertain axis is OMITTED,
    never emitted as "" / null. (Deep-merge on the write side preserves an
    existing value when the key is absent; an emitted empty value would clobber
    it — see mcp-server/tools/profile.py merge-on-write.)
"""

from __future__ import annotations

import pytest

from agent import sharpening_preference_extractor as spe


def _fake_response(content: str):
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


# A realistic onboarding sharpening transcript (mirrors the B-0624-04 dev
# session: tone -> location -> prestige-vs-fit -> exclusion).
_SHARPENING_TRANSCRIPT = [
    {"role": "assistant", "content": "Where are you looking — stay East Coast, open to relocate?"},
    {"role": "user", "content": "Boston, in person. I want to be on-site for my first real role, not remote."},
    {"role": "assistant", "content": "Got it. Next — do you care more about brand prestige or the actual work?"},
    {"role": "user", "content": "the work, 100%. I'd rather be at a smaller team where I own real problems than a tiny cog at some big-name place."},
    {"role": "assistant", "content": "Good call. Any industries or roles you want to skip?"},
    {"role": "user", "content": "honestly the worst would be a pure reporting job — just pulling the same numbers into a dashboard with no modeling."},
]


class TestExtractSharpeningPreferences:
    def test_extracts_stated_axes(self, monkeypatch):
        """A full sharpening series -> a preferences dict carrying the stated axes."""
        extracted_json = (
            '{"preferences": {'
            '"location": "Boston, on-site only (not remote)", '
            '"priority": "role fit over brand prestige; small team with real ownership", '
            '"avoid": "pure reporting / dashboard-only roles with no modeling"'
            '}}'
        )

        def _fake_call_llm(**kwargs):
            return _fake_response(extracted_json)

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)

        out = spe.extract_sharpening_preferences(_SHARPENING_TRANSCRIPT)
        assert out["checked"] is True
        prefs = out["preferences"]
        assert isinstance(prefs, dict)
        assert "location" in prefs and "Boston" in prefs["location"]
        assert "avoid" in prefs
        assert "priority" in prefs

    def test_no_preference_content_returns_empty(self, monkeypatch):
        """Chit-chat with no preference signal -> empty preferences, nothing to write."""

        def _fake_call_llm(**kwargs):
            return _fake_response('{"preferences": {}}')

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)

        out = spe.extract_sharpening_preferences(
            [
                {"role": "assistant", "content": "Talk tomorrow."},
                {"role": "user", "content": "ok thanks, see you"},
            ]
        )
        assert out["checked"] is True
        assert out["preferences"] == {}

    def test_empty_and_null_values_are_dropped(self, monkeypatch):
        """An axis the model returned as "" or null must be dropped, never kept —
        an empty value would clobber an existing profile value under deep-merge."""
        extracted_json = (
            '{"preferences": {'
            '"location": "Boston", '
            '"priority": "", '
            '"avoid": null}}'
        )

        def _fake_call_llm(**kwargs):
            return _fake_response(extracted_json)

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)

        out = spe.extract_sharpening_preferences(_SHARPENING_TRANSCRIPT)
        prefs = out["preferences"]
        assert prefs == {"location": "Boston"}
        assert "priority" not in prefs
        assert "avoid" not in prefs

    def test_empty_transcript_skips_aux_call(self, monkeypatch):
        """No messages -> no aux LLM spend, empty result."""
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("aux LLM should not be called on empty transcript")

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)

        out = spe.extract_sharpening_preferences([])
        assert called["aux"] is False
        assert out["checked"] is False
        assert out["preferences"] == {}

    def test_aux_failure_is_fail_safe(self, monkeypatch):
        """An aux call exception degrades to empty (never raises into the caller)."""

        def _fake_call_llm(**kwargs):
            raise RuntimeError("aux down")

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)

        out = spe.extract_sharpening_preferences(_SHARPENING_TRANSCRIPT)
        assert out["checked"] is False
        assert out["preferences"] == {}
        assert "aux_call_failed" in (out.get("skipped") or "")

    def test_malformed_json_is_fail_safe(self, monkeypatch):
        """Unparseable aux output degrades to empty."""

        def _fake_call_llm(**kwargs):
            return _fake_response("not json at all")

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)

        out = spe.extract_sharpening_preferences(_SHARPENING_TRANSCRIPT)
        assert out["checked"] is False
        assert out["preferences"] == {}
        assert out.get("skipped") == "aux_parse_failed"
