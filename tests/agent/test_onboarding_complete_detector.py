"""Unit tests for agent.onboarding_complete_detector (S-0518-01 Type A)."""

from __future__ import annotations

import pytest

from agent import onboarding_complete_detector as ocd


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


_TRIGGER_JSON = (
    '{"trigger": true, "confidence": "high", '
    '"reasoning": "Coach said briefing the team now", '
    '"intros": ['
    '{"sub_agent": "scout", "text": "Scout text."},'
    '{"sub_agent": "analyst", "text": "Analyst text."},'
    '{"sub_agent": "publicist", "text": "Publicist text."}'
    ']}'
)

_NO_TRIGGER_JSON = (
    '{"trigger": false, "confidence": "high", '
    '"reasoning": "Coach is asking a sharpening question", '
    '"intros": []}'
)


class TestDetectOnboardingComplete:
    def test_short_reply_skipped(self, monkeypatch):
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            _fake_call_llm,
            raising=False,
        )
        result = ocd.detect_onboarding_complete("short reply", {})
        assert result["checked"] is False
        assert result["skipped"] == "reply_too_short"
        assert result["trigger"] is False
        assert called["aux"] is False

    def test_flag_file_short_circuits(self, monkeypatch, tmp_path):
        """Persistent flag file at <hermes_home>/artemis/<uid>/onboarding_pushed.flag
        wins over profile field — survives Coach's save_user_profile overwrites."""
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        user_dir = tmp_path / "artemis" / "U_FLAG_TEST"
        user_dir.mkdir(parents=True)
        (user_dir / "onboarding_pushed.flag").write_text("")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            _fake_call_llm,
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for unit-test length purposes",
            {},  # profile field NOT set — only the file is
            "U_FLAG_TEST",
        )
        assert result["checked"] is False
        assert result["skipped"] == "intros_already_pushed"
        assert called["aux"] is False

    def test_intros_already_pushed_short_circuits(self, monkeypatch):
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            _fake_call_llm,
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for unit-test length purposes",
            {"sub_agent_intros_pushed": True},
        )
        assert result["checked"] is False
        assert result["skipped"] == "intros_already_pushed"
        assert called["aux"] is False

    def test_trigger_with_valid_intros(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_TRIGGER_JSON),
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Marketing Coordinator at a consumer brand. I'm briefing the team now. "
            "You'll hear from them as they spin up.",
            {"goal": "marketing"},
        )
        assert result["checked"] is True
        assert result["trigger"] is True
        assert result["confidence"] == "high"
        assert len(result["intros"]) == 3
        assert [i["sub_agent"] for i in result["intros"]] == [
            "scout", "analyst", "publicist",
        ]

    def test_no_trigger_clears_intros(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_NO_TRIGGER_JSON),
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "What classes have you actually enjoyed, not just done well in? "
            "I'm curious which ones you'd pick again.",
            {},
        )
        assert result["checked"] is True
        assert result["trigger"] is False
        assert result["intros"] == []

    def test_trigger_string_false_is_not_truthy(self, monkeypatch):
        """Regression for codex round 7 P2 — strings like "false" / "no"
        are truthy under bool(), so the old `bool(parsed.get("trigger"))`
        would fire dispatch for a JSON response that meant NOT to trigger.
        Strict identity check (`is True`) prevents this."""
        bad_json = (
            '{"trigger": "false", "confidence": "high", '
            '"reasoning": "model returned a string by mistake", '
            '"intros": ['
            '{"sub_agent": "scout", "text": "Scout here."},'
            '{"sub_agent": "analyst", "text": "Analyst here."},'
            '{"sub_agent": "publicist", "text": "Publicist here."}'
            ']}'
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(bad_json),
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for the purposes of this length-check unit test.",
            {},
        )
        assert result["checked"] is True
        # String "false" must NOT fire dispatch.
        assert result["trigger"] is False
        assert result["intros"] == []

    def test_trigger_with_invalid_intros_demoted(self, monkeypatch):
        """If LLM says trigger=true but intros are malformed, demote to
        no-trigger to avoid dispatching half-baked self-intros."""
        bad_json = (
            '{"trigger": true, "confidence": "high", '
            '"reasoning": "x", '
            '"intros": ['
            '{"sub_agent": "scout", "text": "x"},'
            '{"sub_agent": "scout", "text": "duplicate scout"},'
            '{"sub_agent": "publicist", "text": "y"}'
            ']}'
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(bad_json),
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for the purposes of this length-check unit test.",
            {},
        )
        assert result["checked"] is True
        assert result["trigger"] is False
        assert result["intros"] == []

    def test_intros_force_canonical_order(self, monkeypatch):
        """Regardless of LLM output order, helper returns scout / analyst /
        publicist (matching Maya + Jordan simulation order)."""
        out_of_order = (
            '{"trigger": true, "confidence": "high", "reasoning": "x", '
            '"intros": ['
            '{"sub_agent": "publicist", "text": "P"},'
            '{"sub_agent": "scout", "text": "S"},'
            '{"sub_agent": "analyst", "text": "A"}'
            ']}'
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(out_of_order),
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply about briefing the team to pass length check.",
            {},
        )
        assert [i["sub_agent"] for i in result["intros"]] == [
            "scout", "analyst", "publicist",
        ]
        assert result["intros"][0]["text"] == "S"
        assert result["intros"][1]["text"] == "A"
        assert result["intros"][2]["text"] == "P"

    def test_aux_failure_silent(self, monkeypatch):
        def _raise(**kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _raise, raising=False
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for unit-test length purposes.",
            {},
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
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for unit-test length purposes.",
            {},
        )
        assert result["checked"] is False
        assert result["skipped"] == "aux_parse_failed"

    def test_none_profile_treated_as_empty(self, monkeypatch):
        """Profile=None (cold-start before save_user_profile lands) is OK."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(_TRIGGER_JSON),
            raising=False,
        )
        result = ocd.detect_onboarding_complete(
            "Long enough reply that mentions briefing the team now for unit-test length purposes.",
            None,
        )
        assert result["checked"] is True
        assert result["trigger"] is True


class TestPromptSubstitutionSafety:
    """Regression for codex round 5 P2 — chained `.replace()` would let a
    profile field containing the literal `{coach_reply}` token get
    rewritten with the actual coach reply, silently flipping the
    onboarding-complete decision (and potentially spawning intros for
    the wrong turn). Single-pass regex substitution must keep injected
    content literal."""

    def test_profile_containing_coach_reply_token_is_preserved(self, monkeypatch):
        captured = {}

        def _capture(**kw):
            captured["messages"] = kw.get("messages")
            return _fake_response(
                '{"trigger": false, "confidence": "high", '
                '"reasoning": "", "intros": []}'
            )

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _capture, raising=False,
        )

        # Reply long enough to pass the _MIN_REPLY_LEN_FOR_CHECK gate
        # (60 chars). Profile carries the literal token that the old
        # chained-replace impl would have rewritten.
        coach_reply = (
            "I'm pulling the team in on this one — they'll introduce "
            "themselves in a moment. Hang tight."
        )
        profile = {
            "summary": "user once asked: {coach_reply} please",
            "intro_complete": False,
        }

        ocd.detect_onboarding_complete(coach_reply, profile)

        user_prompt = captured["messages"][1]["content"]
        # The literal {coach_reply} token inside the profile JSON must
        # survive substitution — it must NOT have been overwritten with
        # the actual coach reply.
        assert "{coach_reply} please" in user_prompt
        # And the real coach reply still landed in its placeholder slot.
        assert coach_reply in user_prompt

    def test_reply_containing_user_profile_token_is_preserved(self, monkeypatch):
        captured = {}

        def _capture(**kw):
            captured["messages"] = kw.get("messages")
            return _fake_response(
                '{"trigger": false, "confidence": "high", '
                '"reasoning": "", "intros": []}'
            )

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _capture, raising=False,
        )

        # Coach reply itself contains the OTHER placeholder token;
        # single-pass substitution must not re-scan reply text for
        # `{user_profile}`. Pad to clear the min-length gate (60 chars).
        coach_reply = (
            "I'll send {user_profile} updates over to the team shortly "
            "and you'll hear from them in a moment."
        )
        profile = {"name": "Sam"}
        ocd.detect_onboarding_complete(coach_reply, profile)
        user_prompt = captured["messages"][1]["content"]
        assert coach_reply in user_prompt


class TestNormalizeIntros:
    def test_complete_valid(self):
        intros = ocd._normalize_intros([
            {"sub_agent": "scout", "text": "S"},
            {"sub_agent": "analyst", "text": "A"},
            {"sub_agent": "publicist", "text": "P"},
        ])
        assert len(intros) == 3

    def test_missing_sub_agent(self):
        assert ocd._normalize_intros([
            {"sub_agent": "scout", "text": "S"},
            {"sub_agent": "analyst", "text": "A"},
        ]) == []

    def test_invalid_sub_agent(self):
        assert ocd._normalize_intros([
            {"sub_agent": "strategist", "text": "x"},
            {"sub_agent": "analyst", "text": "y"},
            {"sub_agent": "publicist", "text": "z"},
        ]) == []

    def test_duplicate_sub_agent(self):
        assert ocd._normalize_intros([
            {"sub_agent": "scout", "text": "x"},
            {"sub_agent": "scout", "text": "y"},
            {"sub_agent": "publicist", "text": "z"},
        ]) == []

    def test_empty_text(self):
        assert ocd._normalize_intros([
            {"sub_agent": "scout", "text": ""},
            {"sub_agent": "analyst", "text": "y"},
            {"sub_agent": "publicist", "text": "z"},
        ]) == []

    def test_not_a_list(self):
        assert ocd._normalize_intros("nope") == []
        assert ocd._normalize_intros(None) == []


class TestExecuteViaHelper:
    def test_helper_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = ocd.execute_via_helper(
            "U_TEST",
            [{"sub_agent": "scout", "text": "S"}],
        )
        assert result["ok"] is False
        assert "helper not found" in result["error"]

    def test_empty_intros(self, tmp_path):
        result = ocd.execute_via_helper(
            "U", [], helper_path=str(tmp_path / "any")
        )
        assert result["ok"] is False
        assert "no intros" in result["error"]

    def test_helper_success(self, tmp_path):
        helper = tmp_path / "dispatch.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, 'pushed': len(payload['intros'])}))\n"
        )
        helper.chmod(0o755)
        result = ocd.execute_via_helper(
            "U_TEST",
            [
                {"sub_agent": "scout", "text": "S"},
                {"sub_agent": "analyst", "text": "A"},
                {"sub_agent": "publicist", "text": "P"},
            ],
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["pushed"] == 3

    def test_helper_error_passes_through(self, tmp_path):
        helper = tmp_path / "fail.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': False, 'stage': 'announce', "
            "'error': 'channel missing'}))\n"
        )
        helper.chmod(0o755)
        result = ocd.execute_via_helper(
            "U",
            [{"sub_agent": "scout", "text": "S"}],
            helper_path=str(helper),
        )
        assert result["ok"] is False
        assert result["stage"] == "announce"
        assert "channel missing" in result["error"]


class TestParseResponse:
    def test_clean(self):
        assert ocd._parse_response('{"x": 1}') == {"x": 1}

    def test_fence(self):
        assert ocd._parse_response('```json\n{"x": 1}\n```') == {"x": 1}

    def test_garbage(self):
        assert ocd._parse_response("nope") is None

    def test_array(self):
        assert ocd._parse_response("[1,2]") is None


class TestLogResult:
    def test_emits_single_line(self, caplog):
        with caplog.at_level("INFO", logger="agent.onboarding_complete_detector"):
            ocd.log_result(
                "DTEST",
                {
                    "checked": True,
                    "skipped": None,
                    "trigger": True,
                    "confidence": "high",
                    "reasoning": "x",
                    "intros": [{}, {}, {}],
                },
            )
        lines = [
            r.message for r in caplog.records
            if r.message.startswith("onboarding-complete:")
        ]
        assert len(lines) == 1
        assert "trigger=True" in lines[0]
        assert "n_intros=3" in lines[0]
