"""Unit tests for agent.followup_acceptance_detector (B-0625-04 fix A).

The detector mirrors onboarding_complete_detector: a post-reply auxiliary LLM
classifier that decides whether the user's message accepts a follow-up-draft
offer that a recent briefing surfaced. On a true trigger the gateway fires the
Publicist dispatch (enqueue_action + announce_subagent) deterministically, so
Coach can never narrate a dispatch it skipped (the B-0625-04 regression).

These cover the pure logic only: trigger judgment (strict `is True` + a
non-empty ref_company), tolerant JSON parse, per-offer flag dedup, prompt
substitution safety, and the helper subprocess boundary. The end-to-end
dispatch (detector → helper → enqueue_action → executor spawn → Slack) is
verified live on dev (SIM-FOLLOWUP manual case), not here.
"""

from __future__ import annotations

import pytest

from agent import followup_acceptance_detector as fad


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


_OFFER_BRIEFING = (
    "Good morning. One thing needs a decision today.\n"
    "📌 Follow-ups\n"
    "⭐ TODAY — Manulife went quiet past the response window. "
    "Want me to have Publicist draft a follow-up you can send?\n"
)

_NO_OFFER_BRIEFING = (
    "Good morning. Two new roles landed overnight.\n"
    "🚀 New roles\n"
    "🎯 Acme Data Scientist — 88% match\n"
)

_TRIGGER_JSON = (
    '{"accepted": true, "confidence": "high", '
    '"ref_company": "Manulife", '
    '"reasoning": "user said draft it, briefing offered Manulife follow-up"}'
)

_NO_TRIGGER_JSON = (
    '{"accepted": false, "confidence": "high", '
    '"ref_company": null, '
    '"reasoning": "user is asking an unrelated question"}'
)


class TestDetectFollowupAcceptance:
    def test_short_message_skipped(self, monkeypatch):
        """A message below the minimum length can't carry an acceptance
        signal worth an auxiliary call — but a bare 'yeah' must still be
        checkable, so the floor is low. Empty message is skipped."""
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm, raising=False,
        )
        result = fad.detect_followup_acceptance("", _OFFER_BRIEFING, "Coach reply")
        assert result["checked"] is False
        assert result["skipped"] == "message_empty"
        assert result["trigger"] is False
        assert called["aux"] is False

    def test_no_offer_in_briefing_skips_aux(self, monkeypatch):
        """The ambiguity guard (hermes.md § Follow-up-draft offer): a bare
        'yeah' is only an acceptance when a recent briefing actually carried a
        follow-up-draft offer. No '📌 Follow-ups' offer line → skip the
        auxiliary call entirely, treat as an ordinary turn."""
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm, raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah do it", _NO_OFFER_BRIEFING, "Coach reply text here long enough",
        )
        assert result["checked"] is False
        assert result["skipped"] == "no_offer_in_briefing"
        assert result["trigger"] is False
        assert called["aux"] is False

    def test_empty_briefing_skips_aux(self, monkeypatch):
        """No recent briefing at all (None / empty) → no offer to accept."""
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm, raising=False,
        )
        result = fad.detect_followup_acceptance("yeah do it", None, "Coach reply")
        assert result["checked"] is False
        assert result["skipped"] == "no_offer_in_briefing"
        assert called["aux"] is False

    def test_flag_file_dedups_after_classify(self, monkeypatch, tmp_path):
        """Per-offer flag at <hermes_home>/artemis/<uid>/followup_dispatched_
        <slug>.flag dedups: a Coach re-run or next turn must not double-enqueue
        the same company's follow-up. The slug is only known after the
        classifier resolves the company, so the dedup check runs AFTER the
        auxiliary call (checked=True) but still demotes the trigger to false."""
        coach_dir = tmp_path / "artemis" / "U_FLAG"
        coach_dir.mkdir(parents=True)
        (coach_dir / "followup_dispatched_manulife.flag").write_text("")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_TRIGGER_JSON),
            raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Coach reply", user_id="U_FLAG",
        )
        assert result["checked"] is True
        assert result["skipped"] == "already_dispatched"
        assert result["trigger"] is False
        assert result["action_slug"] is None

    def test_in_flight_dispatch_demotes(self, monkeypatch):
        """Coach may have fired its own follow-up dispatch this same turn (its
        behavior is probabilistic — sometimes it fires, sometimes it doesn't).
        If the action_queue already carries an in-flight follow-up dispatch for
        this company, the detector must NOT fire a second one (the double-fire
        the gateway would otherwise cause). Mirrors hermes.md § Follow-up-draft
        offer trigger ('AND no live follow-up dispatch already in flight for
        that company'). Demote after classify (slug only known post-aux)."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_TRIGGER_JSON),
            raising=False,
        )
        # Coach's own enqueue this turn uses the coach-commit-draft-<co>-follow-up
        # id shape (observed live); detector's helper would use
        # coach-commit-followup-<co>. Either must count as in-flight.
        queue = [{"id": "coach-commit-draft-manulife-follow-up", "status": "pending"}]
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Drafting it.", action_queue=queue,
        )
        assert result["checked"] is True
        assert result["skipped"] == "already_in_flight"
        assert result["trigger"] is False
        assert result["action_slug"] is None

    def test_in_flight_different_company_still_fires(self, monkeypatch):
        """An in-flight follow-up for a DIFFERENT company must not block this
        one — the guard is per-company."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_TRIGGER_JSON),
            raising=False,
        )
        queue = [{"id": "coach-commit-followup-acme", "status": "pending"}]
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Drafting it.", action_queue=queue,
        )
        assert result["checked"] is True
        assert result["trigger"] is True
        assert result["action_slug"] == "coach-commit-followup-manulife"

    def test_trigger_with_company(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_TRIGGER_JSON),
            raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah lets draft the Manulife follow-up", _OFFER_BRIEFING, "Drafting it.",
        )
        assert result["checked"] is True
        assert result["trigger"] is True
        assert result["ref_company"] == "Manulife"
        assert result["action_slug"] == "coach-commit-followup-manulife"
        assert result["confidence"] == "high"

    def test_no_trigger_clears_company(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_NO_TRIGGER_JSON),
            raising=False,
        )
        result = fad.detect_followup_acceptance(
            "what's the deadline for the Acme role again?", _OFFER_BRIEFING, "It's Friday.",
        )
        assert result["checked"] is True
        assert result["trigger"] is False
        assert result["ref_company"] is None
        assert result["action_slug"] is None

    def test_trigger_string_true_is_not_truthy(self, monkeypatch):
        """Strings like "true" are truthy under bool() — only a JSON boolean
        true may fire. Mirrors onboarding detector's codex-round-7 fix."""
        bad_json = (
            '{"accepted": "true", "confidence": "high", '
            '"ref_company": "Manulife", "reasoning": "model returned a string"}'
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(bad_json),
            raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Coach reply",
        )
        assert result["checked"] is True
        assert result["trigger"] is False
        assert result["action_slug"] is None

    def test_trigger_without_company_demoted(self, monkeypatch):
        """accepted=true but no ref_company → demote. Without a company there
        is no offer to bind the dispatch to (no slug to enqueue / dedup on)."""
        bad_json = (
            '{"accepted": true, "confidence": "high", '
            '"ref_company": "", "reasoning": "yes but no company named"}'
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(bad_json),
            raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Coach reply",
        )
        assert result["checked"] is True
        assert result["trigger"] is False
        assert result["ref_company"] is None
        assert result["action_slug"] is None

    def test_aux_failure_silent(self, monkeypatch):
        def _raise(**kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _raise, raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Coach reply",
        )
        assert result["checked"] is False
        assert "aux_call_failed" in result["skipped"]
        assert result["trigger"] is False

    def test_aux_garbage_json(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response("not json"),
            raising=False,
        )
        result = fad.detect_followup_acceptance(
            "yeah do it", _OFFER_BRIEFING, "Coach reply",
        )
        assert result["checked"] is False
        assert result["skipped"] == "aux_parse_failed"


class TestSlugForCompany:
    def test_lowercases_and_hyphenates(self):
        assert fad._slug_for_company("Manulife") == "manulife"
        assert fad._slug_for_company("Beth Israel") == "beth-israel"

    def test_strips_punctuation(self):
        assert fad._slug_for_company("AT&T, Inc.") == "at-t-inc"

    def test_empty_returns_none(self):
        assert fad._slug_for_company("") is None
        assert fad._slug_for_company("   ") is None
        assert fad._slug_for_company(None) is None


class TestCompanyInFlight:
    def test_matches_helper_id_shape(self):
        q = [{"id": "coach-commit-followup-manulife"}]
        assert fad._company_in_flight(q, "manulife") is True

    def test_matches_coach_draft_id_shape(self):
        q = [{"id": "coach-commit-draft-manulife-follow-up"}]
        assert fad._company_in_flight(q, "manulife") is True

    def test_different_company_no_match(self):
        q = [{"id": "coach-commit-followup-acme"}]
        assert fad._company_in_flight(q, "manulife") is False

    def test_non_followup_action_ignored(self):
        q = [{"id": "tailor-resume-manulife"}, {"id": "scout-recheck-manulife"}]
        assert fad._company_in_flight(q, "manulife") is False

    def test_empty_or_none(self):
        assert fad._company_in_flight([], "manulife") is False
        assert fad._company_in_flight(None, "manulife") is False


class TestHasOfferLine:
    def test_detects_followups_offer(self):
        assert fad._has_offer_line(_OFFER_BRIEFING) is True

    def test_no_offer_when_no_followups_marker(self):
        assert fad._has_offer_line(_NO_OFFER_BRIEFING) is False

    def test_none_and_empty(self):
        assert fad._has_offer_line(None) is False
        assert fad._has_offer_line("") is False


class TestParseResponse:
    def test_clean(self):
        assert fad._parse_response('{"x": 1}') == {"x": 1}

    def test_fence(self):
        assert fad._parse_response('```json\n{"x": 1}\n```') == {"x": 1}

    def test_garbage(self):
        assert fad._parse_response("nope") is None

    def test_array(self):
        assert fad._parse_response("[1,2]") is None


class TestPromptSubstitutionSafety:
    """Single-pass regex substitution: a briefing / message containing a
    literal placeholder token must not be re-templated. Mirrors the onboarding
    detector's codex-round-5 fix."""

    def test_message_with_placeholder_token_preserved(self, monkeypatch):
        captured = {}

        def _capture(**kw):
            captured["messages"] = kw.get("messages")
            return _fake_response(_NO_TRIGGER_JSON)

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _capture, raising=False,
        )
        user_message = "yeah {briefing_text} do it"
        fad.detect_followup_acceptance(user_message, _OFFER_BRIEFING, "Coach reply")
        prompt = captured["messages"][1]["content"]
        assert "{briefing_text} do it" in prompt
        assert _OFFER_BRIEFING in prompt


class TestExecuteViaHelper:
    def test_helper_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = fad.execute_via_helper(
            "U_TEST", ref_company="Manulife", action_slug="coach-commit-followup-manulife",
            announcement="Drafting your Manulife follow-up.",
        )
        assert result["ok"] is False
        assert "helper not found" in result["error"]

    def test_missing_slug(self, tmp_path):
        result = fad.execute_via_helper(
            "U", ref_company="Manulife", action_slug="",
            announcement="x", helper_path=str(tmp_path / "any"),
        )
        assert result["ok"] is False
        assert "action_slug" in result["error"]

    def test_helper_success(self, tmp_path):
        helper = tmp_path / "dispatch.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, 'slug': payload['action_slug']}))\n"
        )
        helper.chmod(0o755)
        result = fad.execute_via_helper(
            "U_TEST", ref_company="Manulife",
            action_slug="coach-commit-followup-manulife",
            announcement="Drafting your Manulife follow-up.",
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["slug"] == "coach-commit-followup-manulife"

    def test_helper_error_passes_through(self, tmp_path):
        helper = tmp_path / "fail.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': False, 'stage': 'enqueue', "
            "'error': 'spawn failed'}))\n"
        )
        helper.chmod(0o755)
        result = fad.execute_via_helper(
            "U", ref_company="Manulife",
            action_slug="coach-commit-followup-manulife",
            announcement="x", helper_path=str(helper),
        )
        assert result["ok"] is False
        assert result["stage"] == "enqueue"
        assert "spawn failed" in result["error"]
