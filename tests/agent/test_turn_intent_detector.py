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
    def test_short_surface_message_runs_detector(self, monkeypatch):
        """Bug #9: short surface pulls ('walk me through it', 18 chars) must reach
        the detector — the old length gate (_MIN_USER_MSG_LEN=20) silently killed
        them when sent as a bare DM. The gate is removed; short messages now run."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"lead_in": null, "confidence": "high", '
                '"reasoning": "surface pull"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("walk me through it")
        assert result["checked"] is True
        assert result["skipped"] is None

    def test_empty_message_still_skipped(self, monkeypatch):
        """An empty/whitespace message has nothing to classify — still skip,
        never call the aux LLM."""
        called = {"aux": False}
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: (called.__setitem__("aux", True), None)[1],
            raising=False,
        )
        result = tid.detect_turn_intent("")
        assert result["checked"] is False
        assert result["skipped"] == "empty_message"
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

    # ---------------------------------------------------------------------
    # surface_existing — user-pull of artifacts the backend already produced
    # (read archive[] → announce_subagent, NOT future-work). The fourth
    # dispatch_type: it carries NO dispatches[] (no new action to enqueue)
    # but DOES keep a lead_in (Coach-voice opener before the sub-agent
    # messages). Server reads this type, then surfaces existing archive
    # items as standalone sub-agent messages.
    # ---------------------------------------------------------------------

    def test_surface_existing_parsed_keeps_lead_in_drops_dispatches(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", '
                '"dispatches": [], '
                '"lead_in": "Here\'s what the team put together.", '
                '"confidence": "high", '
                '"reasoning": "user wants to see existing materials"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "walk me through what scout and the publicist actually found"
        )
        assert result["checked"] is True
        assert result["dispatch_type"] == "surface_existing"
        # No future-work items — the artifacts already exist in archive[].
        assert result["dispatches"] == []
        # Unlike `none`, surface_existing keeps its lead_in: Coach opens
        # before the sub-agent messages are surfaced.
        assert result["lead_in"] == "Here's what the team put together."

    def test_surface_existing_strips_stray_dispatches(self, monkeypatch):
        """An LLM that wrongly emits dispatches[] for surface_existing has
        them dropped — surface_existing never enqueues future work."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", '
                '"dispatches": [{"sub_agent": "publicist", '
                '"id_slug": "draft-something", "action": "Draft", '
                '"announcement": "Publicist is on it."}], '
                '"lead_in": "Pulling those up.", '
                '"confidence": "high", "reasoning": "show me the docs"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("show me the docs you made")
        assert result["dispatch_type"] == "surface_existing"
        assert result["dispatches"] == []
        assert result["lead_in"] == "Pulling those up."

    def test_surface_existing_clears_affect_report(self, monkeypatch):
        """affect_report is only meaningful on a `none` turn. A pull turn
        is the user acting on existing work, not holding affect."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"lead_in": "Here you go.", "affect_report": true, '
                '"confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("can you show me the resume again")
        assert result["dispatch_type"] == "surface_existing"
        assert result["affect_report"] is False

    # ---------------------------------------------------------------------
    # surface_deliver (B-0623-05) — a surface_existing pull splits two ways:
    #   - replay summary (default): "walk me through the changes" → text only
    #   - deliver artifact: "send me the PDF" → text + the on-disk file
    # The detector (already reading the turn semantically) sets surface_deliver
    # so the helper knows to attach the artifact. Default False; only
    # meaningful on a surface_existing turn.
    # ---------------------------------------------------------------------

    def test_surface_deliver_true_on_send_file_pull(self, monkeypatch):
        """A pull that asks to RECEIVE the file ("send me the PDF") sets
        surface_deliver=True so the helper attaches the artifact."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"surface_item_ids": ["tailor-resume-healthcare-ds"], '
                '"surface_deliver": true, '
                '"lead_in": "Here is the resume.", '
                '"confidence": "high", "reasoning": "user wants the file"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you send me the healthcare resume PDF so I can upload it"
        )
        assert result["dispatch_type"] == "surface_existing"
        assert result["surface_deliver"] is True

    def test_surface_deliver_false_on_walkthrough_pull(self, monkeypatch):
        """A pull that asks to UNDERSTAND ("walk me through the changes")
        leaves surface_deliver False — summary replay, no attachment."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"surface_deliver": false, '
                '"lead_in": "Here\'s what changed.", '
                '"confidence": "high", "reasoning": "user wants the reframe explained"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "walk me through the changes before I send anything"
        )
        assert result["dispatch_type"] == "surface_existing"
        assert result["surface_deliver"] is False

    def test_surface_deliver_defaults_false_when_absent(self, monkeypatch):
        """No surface_deliver key → False (replay summary, the safe default)."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"lead_in": "Here you go.", '
                '"confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("show me the docs you made")
        assert result["surface_deliver"] is False

    def test_surface_deliver_cleared_off_surface_existing(self, monkeypatch):
        """surface_deliver is only meaningful on a surface_existing turn. A
        leaked true on any other dispatch_type is cleared so a stray flag
        can't trigger an attachment path downstream."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "single", '
                '"dispatches": [{"sub_agent": "publicist", '
                '"id_slug": "draft-x", "action": "Draft", '
                '"announcement": "On it."}], '
                '"surface_deliver": true, '
                '"confidence": "high", "reasoning": "new work"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("draft me a cover letter for acme")
        assert result["dispatch_type"] == "single"
        assert result["surface_deliver"] is False

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
# Capability bucket — A+ design: bucket + user_action_required +
# off_domain_no_fallback parsing and cross-check
# =========================================================================

def _bucket_response(bucket, user_action_required=False, off_domain_no_fallback=False):
    """Build a fake LLM response that's a 'none' dispatch but carries the
    capability bucket fields. Lets tests focus on bucket schema without
    coupling to dispatch shape."""
    import json
    bucket_lit = json.dumps(bucket)
    return _fake_response(
        '{"dispatch_type": "none", "dispatches": [], "lead_in": null, '
        f'"capability_bucket": {bucket_lit}, '
        f'"user_action_required": {str(user_action_required).lower()}, '
        f'"off_domain_no_fallback": {str(off_domain_no_fallback).lower()}, '
        '"confidence": "high", "reasoning": "stub"}'
    )


class TestCapabilityBucketSchema:
    def test_default_fields_when_short_skip(self, monkeypatch):
        # Short-message skip path must still expose the new fields with
        # safe defaults so downstream consumers can read them unconditionally.
        result = tid.detect_turn_intent("ok")
        assert result["capability_bucket"] == "non_capability"
        assert result["user_action_required"] is False
        assert result["off_domain_no_fallback"] is False

    def test_missing_bucket_defaults_to_non_capability(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], "lead_in": null, '
                '"confidence": "low", "reasoning": "no bucket field"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "i'm just feeling really stuck about all this honestly"
        )
        assert result["capability_bucket"] == "non_capability"
        assert result["user_action_required"] is False
        assert result["off_domain_no_fallback"] is False

    @pytest.mark.parametrize("raw,expected", [
        (1, 1), (2, 2), (3, 3), (4, 4),
        ("1", 1), ("4", 4),
        ("non_capability", "non_capability"),
        ("bogus", "non_capability"),
        (None, "non_capability"),
        (5, "non_capability"),
    ])
    def test_bucket_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _bucket_response(raw),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you call her on the phone for me please?"
        )
        assert result["capability_bucket"] == expected

    def test_user_action_required_only_valid_for_bucket_3(self, monkeypatch):
        # bucket=4 + user_action_required=true → flag cleared
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _bucket_response(4, user_action_required=True),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "book me a flight to new york next thursday"
        )
        assert result["capability_bucket"] == 4
        assert result["user_action_required"] is False

    def test_user_action_required_kept_for_bucket_3(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _bucket_response(3, user_action_required=True),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you apply to this job for me at https://example.com/jobs/123"
        )
        assert result["capability_bucket"] == 3
        assert result["user_action_required"] is True

    def test_off_domain_no_fallback_only_valid_for_bucket_4(self, monkeypatch):
        # bucket=3 + off_domain_no_fallback=true → flag cleared
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _bucket_response(3, off_domain_no_fallback=True),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you apply to this job for me please right now"
        )
        assert result["capability_bucket"] == 3
        assert result["off_domain_no_fallback"] is False

    def test_off_domain_no_fallback_kept_for_bucket_4(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _bucket_response(4, off_domain_no_fallback=True),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you sign this NDA for me before tomorrow morning?"
        )
        assert result["capability_bucket"] == 4
        assert result["off_domain_no_fallback"] is True

    def test_non_strict_truthy_for_booleans(self, monkeypatch):
        # The LLM may return 1 or "true" or other truthy non-bool values;
        # we use `is True` semantics so anything not literally True is
        # treated as False. Defends against accidental classification
        # leakage when the LLM returns ambiguous output.
        import json
        raw = (
            '{"dispatch_type": "none", "dispatches": [], "lead_in": null, '
            '"capability_bucket": 3, "user_action_required": "yes", '
            '"off_domain_no_fallback": 1, '
            '"confidence": "high", "reasoning": "stub"}'
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(raw),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "can you apply to this job for me please right now"
        )
        assert result["user_action_required"] is False
        assert result["off_domain_no_fallback"] is False


class TestRenderCapabilityBlock:
    def test_returns_none_when_not_checked(self):
        assert tid.render_capability_block({"checked": False}) is None

    def test_returns_none_for_non_capability(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": "non_capability",
        })
        assert block is None

    def test_returns_none_for_bucket_1(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": 1,
        })
        assert block is None

    def test_renders_for_bucket_2(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": 2,
        })
        assert block is not None
        assert "deliverable" in block.lower()
        # No bucket disclosure in user-visible vocabulary.
        assert "bucket" not in block.lower()

    def test_renders_for_bucket_3_with_user_action_required(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": 3,
            "user_action_required": True,
        })
        assert block is not None
        # Must instruct lead-with-user-step (the tool-call-regrade fix).
        assert "lead" in block.lower()
        assert "user's step" in block.lower()
        assert "bucket" not in block.lower()

    def test_renders_for_bucket_3_without_user_action_required(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": 3,
            "user_action_required": False,
        })
        assert block is not None
        assert "user's step" in block.lower()
        assert "bucket" not in block.lower()

    def test_renders_for_bucket_4_with_off_domain_no_fallback(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": 4,
            "off_domain_no_fallback": True,
        })
        assert block is not None
        assert "clean refusal" in block.lower()
        assert "bucket" not in block.lower()

    def test_renders_for_bucket_4_with_adjacent_capability(self):
        block = tid.render_capability_block({
            "checked": True,
            "capability_bucket": 4,
            "off_domain_no_fallback": False,
        })
        assert block is not None
        assert "career-adjacent" in block.lower()
        assert "bucket" not in block.lower()


# =========================================================================
# affect_report — second-layer signal for Scene 4 #1.
# When the turn is an emotional event-report with no explicit work request
# (dispatch_type=none + affect present), the detector flags affect_report so
# the server can inject a "lead with one affect check-in beat" block. This
# is the prompt-layer half of Scene 4 #1; the routing half (none vs multi)
# is the dispatch_type fix above.
# =========================================================================

def _affect_response(affect_report, dispatch_type="none"):
    """Stub detector response carrying an affect_report flag.

    For dispatch_type != none, supplies a valid dispatches list so the
    normalize/cross-check step keeps the dispatch_type (an empty list
    would demote multi → none and defeat the cross-check under test).
    """
    val = "true" if affect_report is True else (
        "false" if affect_report is False else json.dumps(affect_report)
    )
    if dispatch_type == "multi":
        dispatches = (
            '[{"sub_agent": "analyst", "id_slug": "diagnose", '
            '"action": "Diagnose", "announcement": "Analyst is on it."}, '
            '{"sub_agent": "scout", "id_slug": "find-roles", '
            '"action": "Find roles", "announcement": "Scout is on it."}]'
        )
    elif dispatch_type == "single":
        dispatches = (
            '[{"sub_agent": "publicist", "id_slug": "draft", '
            '"action": "Draft", "announcement": "Publicist is on it."}]'
        )
    else:
        dispatches = "[]"
    return _fake_response(
        '{"dispatch_type": "%s", "dispatches": %s, "lead_in": null, '
        '"capability_bucket": "non_capability", '
        '"affect_report": %s, '
        '"confidence": "high", "reasoning": "report with affect"}'
        % (dispatch_type, dispatches, val)
    )


class TestAffectReportSchema:
    def test_default_false_on_short_skip(self):
        result = tid.detect_turn_intent("ok")
        assert result["affect_report"] is False

    def test_parsed_true(self, monkeypatch):
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _affect_response(True),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "just got out of the screen, think it went ok?? blanked ugh"
        )
        assert result["affect_report"] is True

    @pytest.mark.parametrize("raw", [False, "true", "yes", 1, None, "bogus"])
    def test_only_strict_true_counts(self, monkeypatch, raw):
        # Only a JSON boolean true sets the flag — strings / ints / null
        # must coerce to False so a sloppy LLM output can't misfire the
        # injection.
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _affect_response(raw),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "just got out of the screen, think it went ok?? blanked ugh"
        )
        assert result["affect_report"] is False

    def test_cleared_when_dispatch_not_none(self, monkeypatch):
        # affect_report only meaningful on dispatch_type=none — a turn that
        # dispatches is being acted on, not affect-held. Cross-check clears it.
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _affect_response(True, dispatch_type="multi"),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "dig in and rewrite my materials and find me three more roles"
        )
        # dispatch wins; affect_report must not ride along on a dispatch turn
        assert result["affect_report"] is False


class TestAffectReportBlock:
    def test_returns_none_when_not_checked(self):
        assert tid.render_affect_report_block({"checked": False}) is None

    def test_returns_none_when_flag_false(self):
        assert tid.render_affect_report_block({
            "checked": True,
            "affect_report": False,
        }) is None

    def test_renders_when_flag_true(self):
        block = tid.render_affect_report_block({
            "checked": True,
            "affect_report": True,
        })
        assert block is not None
        # Load-bearing instruction: lead with a one-beat affect check before
        # any analysis / debrief / action.
        low = block.lower()
        assert "check" in low or "feel" in low
        assert "before" in low


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

    def test_surface_existing_dispatches_without_dispatches(self, tmp_path):
        """surface_existing carries no dispatches but is still dispatchable —
        the helper reads the archive. Payload must forward dispatch_type and
        lead_in so the helper routes to the surface path."""
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "out = {'ok': True, "
            "'dispatch_type_seen': payload.get('dispatch_type'), "
            "'lead_in_seen': payload.get('lead_in'), "
            "'surfaced': []}\n"
            "print(json.dumps(out))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {
                "dispatch_type": "surface_existing",
                "dispatches": [],
                "lead_in": "Here's what the team put together.",
            },
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["dispatch_type_seen"] == "surface_existing"
        assert result["lead_in_seen"] == "Here's what the team put together."

    def test_surface_existing_forwards_surface_item_ids(self, tmp_path):
        """S-0622-03: a directed pull's surface_item_ids must reach the helper
        payload so the helper surfaces only the named items."""
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, "
            "'ids_seen': payload.get('surface_item_ids'), 'surfaced': []}))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {
                "dispatch_type": "surface_existing",
                "dispatches": [],
                "surface_item_ids": ["tailor-resume-target"],
                "lead_in": "Pulling up your Target materials.",
            },
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["ids_seen"] == ["tailor-resume-target"]

    def test_surface_existing_unscoped_omits_item_ids(self, tmp_path):
        """An unscoped pull (empty/absent surface_item_ids) forwards no ids →
        helper falls back to full-team replay."""
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, "
            "'has_ids_key': 'surface_item_ids' in payload, 'surfaced': []}))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {"dispatch_type": "surface_existing", "dispatches": [],
             "surface_item_ids": []},
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["has_ids_key"] is False

    def test_surface_deliver_true_forwarded(self, tmp_path):
        """B-0623-05: a deliver pull forwards surface_deliver=True so the
        helper attaches the artifact."""
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, "
            "'deliver_seen': payload.get('surface_deliver'), 'surfaced': []}))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {
                "dispatch_type": "surface_existing",
                "dispatches": [],
                "surface_item_ids": ["tailor-resume-healthcare-ds"],
                "surface_deliver": True,
            },
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["deliver_seen"] is True

    def test_surface_deliver_false_omits_key(self, tmp_path):
        """A replay pull (surface_deliver False/absent) forwards no key →
        helper attaches nothing, summary-only walkthrough shape preserved."""
        helper = tmp_path / "exec.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'ok': True, "
            "'has_deliver_key': 'surface_deliver' in payload, 'surfaced': []}))\n"
        )
        helper.chmod(0o755)
        result = tid.execute_via_helper(
            "U_TEST",
            {"dispatch_type": "surface_existing", "dispatches": [],
             "surface_deliver": False},
            helper_path=str(helper),
        )
        assert result["ok"] is True
        assert result["has_deliver_key"] is False

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

class TestNormalizeDispatches:
    """Regression for codex round 6 P2 (dedupe) + round 7 P3
    (type-checking LLM string fields)."""

    def test_drops_item_with_non_string_action(self):
        # Regression for codex round 7 — class B (str method on
        # unchecked LLM value). Old impl called .strip() unconditionally
        # and crashed on bool/int values, swallowed by outer except,
        # dropping the entire dispatch.
        raw = [
            {
                "sub_agent": "analyst",
                "id_slug": "bad-one",
                "action": True,  # not a string
                "announcement": "Analyst on it.",
            },
            {
                "sub_agent": "scout",
                "id_slug": "good-one",
                "action": "Scan PM roles",
                "announcement": "Scout scanning.",
            },
        ]
        out = tid._normalize_dispatches(raw)
        # Bad item dropped, good one kept.
        assert len(out) == 1
        assert out[0]["sub_agent"] == "scout"

    def test_drops_item_with_non_string_announcement(self):
        raw = [
            {
                "sub_agent": "analyst",
                "id_slug": "x",
                "action": "x",
                "announcement": 42,  # not a string
            },
        ]
        assert tid._normalize_dispatches(raw) == []

    def test_dedupes_repeated_sub_agent(self):
        raw = [
            {
                "sub_agent": "analyst",
                "id_slug": "draft-one",
                "action": "Draft thing A",
                "announcement": "Analyst on A.",
            },
            {
                "sub_agent": "analyst",
                "id_slug": "draft-two",
                "action": "Draft thing B",
                "announcement": "Analyst on B.",
            },
            {
                "sub_agent": "scout",
                "id_slug": "scan-roles",
                "action": "Scan PM roles",
                "announcement": "Scout scanning.",
            },
        ]
        out = tid._normalize_dispatches(raw)
        assert len(out) == 2
        assert [d["sub_agent"] for d in out] == ["analyst", "scout"]
        # First analyst wins — second is dropped.
        assert out[0]["id_slug"] == "draft-one"

    def test_keeps_distinct_sub_agents(self):
        raw = [
            {
                "sub_agent": "scout",
                "id_slug": "a",
                "action": "x",
                "announcement": "y",
            },
            {
                "sub_agent": "analyst",
                "id_slug": "b",
                "action": "x",
                "announcement": "y",
            },
            {
                "sub_agent": "publicist",
                "id_slug": "c",
                "action": "x",
                "announcement": "y",
            },
        ]
        out = tid._normalize_dispatches(raw)
        assert len(out) == 3
        assert {d["sub_agent"] for d in out} == {"scout", "analyst", "publicist"}


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


# =========================================================================
# Prompt-rule guard — emotional event-report with no explicit work request
# must route to `none` so Coach handles the affect check-in inline, not a
# premature multi-dispatch (Artemis Scene 4 #1 / Maya post-interview debrief).
#
# This is an A-layer guard: it asserts the classification rule survives in
# the prompt text. The behavioral red-green for this rule is verified on the
# dev VPS via state-injection (the LLM is mocked out in every unit test
# here, so a mocked dispatch_type assertion would be tautological).
# =========================================================================

class TestEmotionalReportRoutesNonePromptRule:
    def test_prompt_states_report_without_request_is_none(self):
        prompt = tid._DETECT_PROMPT.lower()
        # The rule must tie three things together: reporting an outcome/event,
        # affect being present, and the ABSENCE of an explicit analysis/
        # review/action request → none (Coach handles inline).
        assert "report" in prompt
        assert "none" in prompt
        # Load-bearing phrase: a report+affect turn with no explicit ask is
        # NOT a dispatch trigger. Guards against the rule being silently
        # dropped in a future prompt edit.
        assert "explicit" in prompt and "request" in prompt

    def test_prompt_distinguishes_report_from_dig_in(self):
        # The fix must NOT break the real dig-in trigger: an explicit
        # "dig in" / "walk me through what happened" AFTER a setback still
        # routes to multi. The prompt must keep that example.
        prompt = tid._DETECT_PROMPT.lower()
        assert "dig in" in prompt


class TestSurfaceExistingPromptRule:
    def test_prompt_defines_surface_existing_shape(self):
        # The fourth shape must be described in the prompt + the JSON schema
        # so the LLM can emit it. Guards against the value being added to the
        # parser's allow-list but never taught to the model.
        prompt = tid._DETECT_PROMPT.lower()
        assert "surface_existing" in prompt

    def test_prompt_ties_surface_existing_to_already_produced(self):
        # The load-bearing distinction: surface_existing is for artifacts the
        # backend ALREADY produced (in archive), pulled by the user — NOT new
        # work. The prompt must tie the shape to "already" existing output so
        # it doesn't blur into single/multi (which create new work).
        prompt = tid._DETECT_PROMPT.lower()
        assert "surface_existing" in prompt
        assert "already" in prompt
        # The user-pull triggers from the simulation must be present as
        # examples so the model recognizes the shape.
        assert "walk me through" in prompt
        assert "show me" in prompt

    def test_surface_existing_in_schema_block(self):
        # The strict-JSON schema enumeration must list surface_existing as a
        # valid dispatch_type value alongside none/single/multi.
        prompt = tid._DETECT_PROMPT
        # The schema line enumerates the dispatch_type union.
        assert '"surface_existing"' in prompt


# =========================================================================
# render_short_circuit_transcript_text
# =========================================================================

class TestRenderShortCircuitTranscriptText:
    def test_surface_existing_joins_products(self):
        surfaced = [
            {"sub_agent": "Scout", "summary": "Found 3 Topicals roles."},
            {"sub_agent": "Researcher", "summary": "Topicals raised a Series B."},
        ]
        out = tid.render_short_circuit_transcript_text(surfaced=surfaced)
        assert "Topicals" in out
        assert "Scout: Found 3 Topicals roles." in out
        assert "Researcher: Topicals raised a Series B." in out

    def test_surface_existing_missing_fields_degrade(self):
        surfaced = [{"summary": "no sub_agent key"}, {"sub_agent": "Scout"}]
        out = tid.render_short_circuit_transcript_text(surfaced=surfaced)
        # Missing sub_agent falls back to a generic label; missing summary -> empty.
        assert "no sub_agent key" in out
        assert "Scout" in out

    def test_lead_in_used_verbatim(self):
        out = tid.render_short_circuit_transcript_text(lead_in="Pulling the team in.")
        assert out == "Pulling the team in."

    def test_lead_in_stripped(self):
        out = tid.render_short_circuit_transcript_text(lead_in="  Team's on it.  ")
        assert out == "Team's on it."

    def test_empty_inputs_return_empty(self):
        assert tid.render_short_circuit_transcript_text() == ""
        assert tid.render_short_circuit_transcript_text(surfaced=[]) == ""
        assert tid.render_short_circuit_transcript_text(lead_in="") == ""
        assert tid.render_short_circuit_transcript_text(lead_in="   ") == ""

    def test_surfaced_takes_precedence_over_lead_in(self):
        # surface_existing path never carries a lead_in, but guard the
        # contract: if both passed, surfaced wins (it is the richer record).
        out = tid.render_short_circuit_transcript_text(
            surfaced=[{"sub_agent": "Scout", "summary": "x"}],
            lead_in="ignored",
        )
        assert out == "Scout: x"


# =========================================================================
# render_onboarding_sharpening_block (S-0617-01)
# =========================================================================

class TestRenderOnboardingSharpeningBlock:
    def test_none_returns_blocking_reverse_engineering_instruction(self):
        block = tid.render_onboarding_sharpening_block("none")
        assert block is not None
        assert "one axis" in block.lower()
        # substring appears in the "do NOT say 'briefing the team' yet" instruction
        assert "briefing the team" in block.lower()
        assert "direction" in block.lower()

    def test_multi_returns_none(self):
        # non-blocking path moved to onboarding_preference_detector (S-0617-01 v3
        # correction). The cold-start block no longer carries it.
        assert tid.render_onboarding_sharpening_block("multi") is None

    def test_single_returns_none(self):
        assert tid.render_onboarding_sharpening_block("single") is None

    def test_surface_existing_returns_none(self):
        assert tid.render_onboarding_sharpening_block("surface_existing") is None

    def test_unknown_returns_none(self):
        assert tid.render_onboarding_sharpening_block("bogus") is None

    def test_none_arg_returns_none(self):
        assert tid.render_onboarding_sharpening_block(None) is None


# =========================================================================
# S-0622-03 — directed surface_existing: the detector selects which archive
# items a directed pull points at and returns them in `surface_item_ids`.
# The detector is fed a compact archive index so it can resolve the user's
# phrasing ("the target stuff") to specific archive ids semantically,
# sidestepping the company/adjective string-match collision.
# =========================================================================


class TestSurfaceItemIds:
    def test_surface_existing_parses_surface_item_ids(self, monkeypatch):
        """surface_existing turn whose LLM output names archive ids → the
        detector surfaces them in `surface_item_ids`."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"surface_item_ids": ["tailor-resume-target"], '
                '"lead_in": "Pulling up your Target materials.", '
                '"confidence": "high", "reasoning": "directed pull for Target"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent(
            "lets see the target stuff",
            archive_index=[
                {"id": "tailor-resume-target", "sub_agent": "publicist",
                 "artifact_name": "target-marketing-coordinator"},
                {"id": "job-match-20260620", "sub_agent": "scout",
                 "artifact_name": None},
            ],
        )
        assert result["dispatch_type"] == "surface_existing"
        assert result["surface_item_ids"] == ["tailor-resume-target"]

    def test_unscoped_surface_existing_empty_item_ids(self, monkeypatch):
        """An unscoped pull ('walk me through what the team did') → no
        surface_item_ids → empty list → helper does full-team replay."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"lead_in": "Here\'s what the team put together.", '
                '"confidence": "high", "reasoning": "unscoped walkthrough"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("walk me through what the team did")
        assert result["dispatch_type"] == "surface_existing"
        assert result["surface_item_ids"] == []

    def test_surface_item_ids_empty_for_non_surface_dispatch(self, monkeypatch):
        """surface_item_ids is only meaningful on surface_existing turns; any
        value the LLM emits on a none/single/multi turn is dropped."""
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"surface_item_ids": ["leaked-id"], '
                '"lead_in": null, "confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("how are you")
        assert result["dispatch_type"] == "none"
        assert result["surface_item_ids"] == []

    def test_archive_index_injected_into_prompt(self, monkeypatch):
        """The compact archive index reaches the LLM prompt so it can resolve
        phrasing → ids. Capture the prompt the detector builds."""
        captured = {}
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: captured.update(messages=kw.get("messages")) or _fake_response(
                '{"dispatch_type": "surface_existing", "dispatches": [], '
                '"surface_item_ids": [], "lead_in": null, '
                '"confidence": "low", "reasoning": "x"}'
            ),
            raising=False,
        )
        tid.detect_turn_intent(
            "show me the target stuff",
            archive_index=[
                {"id": "tailor-resume-target", "sub_agent": "publicist",
                 "artifact_name": "target-marketing-coordinator"},
            ],
        )
        prompt = captured["messages"][1]["content"]
        assert "tailor-resume-target" in prompt


# =========================================================================
# S-0622-03 — build_archive_index: the gateway builds a compact index of
# surfaceable archive items to feed the detector each turn. Candidate set =
# sub-agent-attributed + non-empty summary (NOT 24h-gated — a directed pull
# may name an older item; matches the helper's directed-pull collector).
# =========================================================================

import time as _time


def _ai_item(action_id, sub_agent, summary="work", artifact_name=None,
             hours_ago=1.0):
    from datetime import datetime, timezone, timedelta
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    item = {"id": action_id, "sub_agent": sub_agent, "summary": summary,
            "status": "done", "completed_at": ts}
    if artifact_name is not None:
        item["artifact_name"] = artifact_name
    return item


class TestBuildArchiveIndex:
    def test_emits_id_sub_agent_artifact_name(self):
        archive = [_ai_item("tailor-resume-target", "publicist", "Target packet.",
                             artifact_name="target-marketing-coordinator")]
        idx = tid.build_archive_index(archive)
        assert idx == [{"id": "tailor-resume-target", "sub_agent": "publicist",
                        "artifact_name": "target-marketing-coordinator"}]

    def test_includes_items_older_than_24h(self):
        """A directed pull may name an old item — index is not 24h-gated."""
        archive = [_ai_item("old-pub", "publicist", "Old draft.", hours_ago=72)]
        idx = tid.build_archive_index(archive)
        assert [i["id"] for i in idx] == ["old-pub"]

    def test_skips_non_subagent_and_summaryless(self):
        archive = [
            {"id": "plain", "summary": "no sub_agent", "status": "done"},
            _ai_item("no-summary", "scout", summary=""),
            _ai_item("good", "analyst", "Has summary."),
        ]
        idx = tid.build_archive_index(archive)
        assert [i["id"] for i in idx] == ["good"]

    def test_artifact_name_null_when_absent(self):
        idx = tid.build_archive_index([_ai_item("scan-1", "scout", "Scan.")])
        assert idx[0]["artifact_name"] is None

    def test_empty_or_non_list_returns_empty(self):
        assert tid.build_archive_index([]) == []
        assert tid.build_archive_index(None) == []
        assert tid.build_archive_index("nope") == []
