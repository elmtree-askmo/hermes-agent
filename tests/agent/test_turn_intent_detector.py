"""Unit tests for agent.turn_intent_detector (S-0518-01 directions B+C + Type F)."""

from __future__ import annotations

import pytest

from agent import turn_intent_detector as tid


# =========================================================================
# detect_turn_intent — schema + dispatch_type handling
# =========================================================================

def _fake_response(content: str):
    """Helper that builds a stub OpenAI-style response object."""
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


class TestDetectTurnIntent:
    def test_short_message_skipped(self, monkeypatch):
        called = {"aux": False}
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: (called.__setitem__("aux", True), None)[1],
            raising=False,
        )
        result = tid.detect_turn_intent("ok")
        assert result["checked"] is False
        assert result["skipped"] == "msg_too_short"
        assert result["dispatch_type"] == "none"
        assert result["dispatches"] == []
        assert result["lead_in"] is None
        assert called["aux"] is False

    def test_single_dispatch_parsed(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "single", '
                '"dispatches": [{"sub_agent": "analyst", '
                '"id_slug": "draft-cheat-sheet", '
                '"action": "Draft metrics cheat sheet", '
                '"announcement": "Analyst is on it."}], '
                '"lead_in": "On it.", '
                '"confidence": "high", "reasoning": "user asked for artifact"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you put together a metrics cheat sheet for me?"
        )
        assert result["checked"] is True
        assert result["dispatch_type"] == "single"
        assert len(result["dispatches"]) == 1
        d = result["dispatches"][0]
        assert d["sub_agent"] == "analyst"
        assert d["id_slug"] == "draft-cheat-sheet"
        assert result["lead_in"] == "On it."
        assert result["confidence"] == "high"

    def test_multi_dispatch_parsed(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "multi", '
                '"dispatches": ['
                '  {"sub_agent": "analyst", "id_slug": "diagnose-rejection", '
                '   "action": "Diagnose Glossier rejection", '
                '   "announcement": "Analyst is digging in."},'
                '  {"sub_agent": "scout", "id_slug": "find-alts", '
                '   "action": "Find similar-profile alternatives", '
                '   "announcement": "Scout is scanning."},'
                '  {"sub_agent": "publicist", "id_slug": "rewrite-bullet", '
                '   "action": "Rewrite metrics bullet across apps", '
                '   "announcement": "Publicist is rewriting."}'
                '], '
                '"lead_in": "Pulling the team in.", '
                '"confidence": "high", "reasoning": "dig-in moment"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "dig in. I wanna get better",
            history=[
                {"role": "user", "content": "glossier said no"},
                {"role": "assistant", "content": "Ugh, how are you feeling?"},
            ],
        )
        assert result["checked"] is True
        assert result["dispatch_type"] == "multi"
        assert len(result["dispatches"]) == 3
        assert [d["sub_agent"] for d in result["dispatches"]] == [
            "analyst", "scout", "publicist",
        ]
        assert result["lead_in"] == "Pulling the team in."

    def test_none_dispatch_no_lead_in(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"lead_in": "should be dropped", '
                '"confidence": "high", "reasoning": "emotional"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "i'm just feeling really stuck about all this honestly"
        )
        assert result["dispatch_type"] == "none"
        assert result["dispatches"] == []
        assert result["lead_in"] is None  # forced to None for dispatch_type=none

    def test_multi_with_one_dispatch_demoted_to_single(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "multi", '
                '"dispatches": [{"sub_agent": "analyst", '
                '"id_slug": "x", "action": "y", "announcement": "z"}], '
                '"lead_in": "OK.", "confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("can you analyze this?")
        assert result["dispatch_type"] == "single"
        assert len(result["dispatches"]) == 1

    def test_multi_with_zero_dispatch_demoted_to_none(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "multi", "dispatches": [], '
                '"lead_in": "OK.", "confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("can you help me think?")
        assert result["dispatch_type"] == "none"
        assert result["dispatches"] == []
        assert result["lead_in"] is None

    def test_invalid_dispatch_type_falls_to_none(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "team-fanout", "dispatches": [], '
                '"lead_in": null, "confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("can you help?")
        assert result["dispatch_type"] == "none"

    def test_invalid_sub_agent_dropped(self, monkeypatch):
        """Dispatch with invalid sub_agent gets dropped from list."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "multi", '
                '"dispatches": ['
                '  {"sub_agent": "strategist", "id_slug": "x", '
                '   "action": "y", "announcement": "z"},'
                '  {"sub_agent": "analyst", "id_slug": "a", '
                '   "action": "b", "announcement": "c"},'
                '  {"sub_agent": "scout", "id_slug": "d", '
                '   "action": "e", "announcement": "f"}'
                '], "lead_in": "team in", "confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("can you do everything?")
        # strategist dropped, leaves 2 valid → still multi
        assert result["dispatch_type"] == "multi"
        assert len(result["dispatches"]) == 2

    def test_aux_failure_silent(self, monkeypatch):
        def _raise(**kw):
            raise RuntimeError("network timeout")
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _raise, raising=False,
        )
        result = tid.detect_turn_intent("can you draft me a cover letter?")
        assert result["checked"] is False
        assert "aux_call_failed" in result["skipped"]
        assert result["dispatch_type"] == "none"

    def test_aux_garbage_json(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response("not json"),
            raising=False,
        )
        result = tid.detect_turn_intent("can you draft me a cover letter?")
        assert result["checked"] is False
        assert result["skipped"] == "aux_parse_failed"


# =========================================================================
# Prompt substitution safety — regression for codex round 5 P2
# (chained .replace re-templates injected content)
# =========================================================================

class TestPromptSubstitutionSafety:
    """Verify the auxiliary prompt is built via single-pass substitution
    so that history content containing the literal token `{user_message}`
    is NOT rewritten with the actual user message. The previous chained
    `.replace()` impl would corrupt classifier context this way."""

    def test_history_containing_user_message_token_is_preserved(self, monkeypatch):
        captured = {}

        def _capture(**kw):
            captured["messages"] = kw.get("messages")
            return _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"lead_in": null, "confidence": "high", "reasoning": ""}'
            )

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _capture, raising=False,
        )

        booby_trap = "The literal token {user_message} should stay intact."
        history = [{"role": "user", "content": booby_trap}]
        tid.detect_turn_intent(
            "what should I do today?",
            history=history,
        )

        user_prompt = captured["messages"][1]["content"]
        # The literal `{user_message}` token from history must not have
        # been re-substituted with the actual user message.
        assert "{user_message} should stay intact" in user_prompt
        # The real user message still landed in its placeholder slot.
        assert "what should I do today?" in user_prompt

    def test_history_containing_conversation_history_token_is_preserved(
        self, monkeypatch,
    ):
        captured = {}

        def _capture(**kw):
            captured["messages"] = kw.get("messages")
            return _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"lead_in": null, "confidence": "high", "reasoning": ""}'
            )

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _capture, raising=False,
        )

        # User message itself contains the other placeholder token —
        # single-pass substitution must not re-scan it for
        # `{conversation_history}`.
        booby = "Tell me about {conversation_history} please."
        tid.detect_turn_intent(booby, history=None)

        user_prompt = captured["messages"][1]["content"]
        assert booby in user_prompt


# =========================================================================
# _format_history
# =========================================================================

class TestFormatHistory:
    def test_empty_history(self):
        assert "no prior exchanges" in tid._format_history(None)
        assert "no prior exchanges" in tid._format_history([])

    def test_basic_history(self):
        out = tid._format_history([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ])
        assert "User: hi" in out
        assert "Coach: hey" in out

    def test_truncates_long_content(self):
        long_text = "x" * 600
        out = tid._format_history([{"role": "user", "content": long_text}])
        assert "[…]" in out

    def test_skips_malformed(self):
        out = tid._format_history([
            "not a dict",
            {"role": "user", "content": "valid"},
            {"role": "user"},  # no content
        ])
        assert "User: valid" in out


# =========================================================================
# render_injection_block (fallback)
# =========================================================================

class TestRenderInjectionBlock:
    def test_returns_none_for_none_dispatch(self):
        detection = {
            "checked": True,
            "dispatch_type": "none",
            "dispatches": [],
        }
        assert tid.render_injection_block(detection) is None

    def test_renders_for_single(self):
        detection = {
            "checked": True,
            "dispatch_type": "single",
            "dispatches": [{
                "sub_agent": "analyst",
                "id_slug": "draft-cheat-sheet",
                "action": "Draft cheat sheet",
                "announcement": "Analyst is on it.",
            }],
        }
        block = tid.render_injection_block(detection)
        assert block is not None
        assert "coach-commit-draft-cheat-sheet" in block
        assert "Draft cheat sheet" in block
        assert "Analyst is on it." in block
        assert "enqueue_action" in block
        assert "announce_subagent" in block

    def test_renders_for_multi(self):
        detection = {
            "checked": True,
            "dispatch_type": "multi",
            "dispatches": [
                {"sub_agent": "analyst", "id_slug": "a",
                 "action": "Diagnose", "announcement": "A on it"},
                {"sub_agent": "scout", "id_slug": "s",
                 "action": "Find alts", "announcement": "S scanning"},
            ],
        }
        block = tid.render_injection_block(detection)
        assert block is not None
        assert "analyst" in block
        assert "scout" in block
        assert "coach-commit-a" in block
        assert "coach-commit-s" in block

    def test_returns_none_when_not_checked(self):
        assert tid.render_injection_block({"checked": False}) is None


# =========================================================================
# render_already_executed_block (single Type E)
# =========================================================================

class TestRenderAlreadyExecutedBlock:
    def test_block_contains_id_action_subagent(self):
        block = tid.render_already_executed_block(
            sub_agent="analyst",
            action="Draft cheat sheet",
            full_id="coach-commit-x",
        )
        assert "Sub-agent action already executed" in block
        assert "coach-commit-x" in block
        assert "Draft cheat sheet" in block
        assert "analyst" in block
        assert "Do NOT call either tool again" in block


# =========================================================================
# render_team_dispatch_executed_block (multi Type F)
# =========================================================================

class TestRenderTeamDispatchExecutedBlock:
    def test_with_lead_in_pushed(self):
        block = tid.render_team_dispatch_executed_block(
            dispatches=[
                {"sub_agent": "analyst", "action": "Diagnose", "id_slug": "x"},
                {"sub_agent": "scout", "action": "Find alts", "id_slug": "y"},
            ],
            lead_in_pushed=True,
        )
        assert "Team dispatch already executed" in block
        assert "analyst" in block
        assert "scout" in block
        assert "Diagnose" in block
        assert "lead-in has ALREADY been pushed" in block
        assert "post_activity_log" in block

    def test_without_lead_in_pushed(self):
        block = tid.render_team_dispatch_executed_block(
            dispatches=[
                {"sub_agent": "analyst", "action": "Diagnose", "id_slug": "x"},
            ],
            lead_in_pushed=False,
        )
        assert "Provide a 1-sentence Coach-voice lead-in" in block


# =========================================================================
# execute_via_helper
# =========================================================================

class TestExecuteViaHelper:
    def test_helper_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        detection = {
            "dispatch_type": "single",
            "dispatches": [{
                "sub_agent": "analyst", "id_slug": "x",
                "action": "y", "announcement": "z",
            }],
        }
        result = tid.execute_via_helper("U", detection)
        assert result["ok"] is False
        assert "helper not found" in result["error"]

    def test_dispatch_type_none_short_circuits(self, tmp_path):
        result = tid.execute_via_helper(
            "U",
            {"dispatch_type": "none", "dispatches": []},
            helper_path=str(tmp_path / "x"),
        )
        assert result["ok"] is False
        assert "not dispatchable" in result["error"]

    def test_empty_dispatches_short_circuits(self, tmp_path):
        result = tid.execute_via_helper(
            "U",
            {"dispatch_type": "single", "dispatches": []},
            helper_path=str(tmp_path / "x"),
        )
        assert result["ok"] is False
        assert "not dispatchable" in result["error"]

    def test_single_dispatch_helper_success(self, tmp_path):
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, 'lead_in_pushed': False, "
            "'results': [{'sub_agent': payload['dispatches'][0]['sub_agent']}]}))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {
                "dispatch_type": "single",
                "dispatches": [{
                    "sub_agent": "analyst", "id_slug": "x",
                    "action": "y", "announcement": "z",
                }],
            },
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["results"][0]["sub_agent"] == "analyst"

    def test_multi_with_lead_in(self, tmp_path):
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "out = {'ok': True, 'lead_in_pushed': 'lead_in' in payload, "
            "'results': [{'sub_agent': d['sub_agent']} for d in payload['dispatches']]}\n"
            "print(json.dumps(out))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {
                "dispatch_type": "multi",
                "lead_in": "Pulling the team in.",
                "dispatches": [
                    {"sub_agent": "analyst", "id_slug": "a",
                     "action": "1", "announcement": "x"},
                    {"sub_agent": "scout", "id_slug": "s",
                     "action": "2", "announcement": "y"},
                ],
            },
            helper_path=str(helper),
            push_lead_in=True,
        )
        assert result["ok"] is True
        assert result["lead_in_pushed"] is True
        assert len(result["results"]) == 2

    def test_helper_returns_error(self, tmp_path):
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'ok': False, 'stage': 'enqueue', "
            "'error': 'queue full'}))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U",
            {
                "dispatch_type": "single",
                "dispatches": [{"sub_agent": "analyst", "id_slug": "x",
                                "action": "y", "announcement": "z"}],
            },
            helper_path=str(helper),
        )
        assert result["ok"] is False
        assert result["stage"] == "enqueue"
        assert result["error"] == "queue full"

    def test_helper_non_json_stdout(self, tmp_path):
        helper = tmp_path / "bad.py"
        helper.write_text(
            "#!/usr/bin/env python3\nprint('not json')\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U",
            {
                "dispatch_type": "single",
                "dispatches": [{"sub_agent": "analyst", "id_slug": "x",
                                "action": "y", "announcement": "z"}],
            },
            helper_path=str(helper),
        )
        assert result["ok"] is False
        assert "non-JSON" in result["error"]


# =========================================================================
# log_result
# =========================================================================

class TestLogResult:
    def test_logs_single_dispatch(self, caplog):
        detection = {
            "checked": True,
            "skipped": None,
            "dispatch_type": "single",
            "dispatches": [{"sub_agent": "analyst", "id_slug": "x",
                            "action": "y", "announcement": "z"}],
            "lead_in": "On it.",
            "confidence": "high",
            "reasoning": "asked for artifact",
        }
        with caplog.at_level("INFO", logger="agent.turn_intent_detector"):
            tid.log_result("DTEST", detection)
        lines = [
            r.message for r in caplog.records
            if r.message.startswith("turn-intent:")
        ]
        assert len(lines) == 1
        assert "dispatch_type=single" in lines[0]
        assert "sub_agents=analyst" in lines[0]
        assert "n=1" in lines[0]

    def test_logs_multi_dispatch(self, caplog):
        detection = {
            "checked": True,
            "skipped": None,
            "dispatch_type": "multi",
            "dispatches": [
                {"sub_agent": "analyst", "id_slug": "a", "action": "x", "announcement": "y"},
                {"sub_agent": "scout", "id_slug": "s", "action": "x", "announcement": "y"},
            ],
            "lead_in": "Pulling the team in.",
            "confidence": "high",
            "reasoning": "dig-in",
        }
        with caplog.at_level("INFO", logger="agent.turn_intent_detector"):
            tid.log_result("DTEST", detection)
        lines = [
            r.message for r in caplog.records
            if r.message.startswith("turn-intent:")
        ]
        assert "dispatch_type=multi" in lines[0]
        assert "n=2" in lines[0]
        assert "sub_agents=analyst,scout" in lines[0]


# =========================================================================
# _sanitize_slug
# =========================================================================

class TestSanitizeSlug:
    def test_clean(self):
        assert tid._sanitize_slug("draft-cover-letter") == "draft-cover-letter"

    def test_strip_prefix(self):
        assert tid._sanitize_slug("coach-commit-x") == "x"

    def test_empty(self):
        assert tid._sanitize_slug("") is None
        assert tid._sanitize_slug(None) is None
        assert tid._sanitize_slug(42) is None

    def test_too_long(self):
        assert tid._sanitize_slug("a-" * 40) is None


# =========================================================================
# _parse_response
# =========================================================================

class TestParseResponse:
    def test_clean(self):
        assert tid._parse_response('{"a": 1}') == {"a": 1}

    def test_fence(self):
        assert tid._parse_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_garbage(self):
        assert tid._parse_response("nope") is None

    def test_array(self):
        assert tid._parse_response("[1,2]") is None
