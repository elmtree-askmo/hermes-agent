"""B-0722-02 (Artemis): precision guards for the user-report keyword detectors.

The outcome/interview signal tables were only ever tested on the recall axis
(genuine reports fire; signal-free turns don't). This suite covers the missing
precision axis: innocent casual chat containing a signal substring must NOT
write ledger state. Three contracts under test:

1. Word-boundary matching — bare-substring hits inside larger words
   ("sunscreen" ⊃ "screen") never fire.
2. Outcome no-name fallback removed — a keyword hit with no named ledger
   company does not guess the most-recent application (unless the same-turn
   turn-intent verdict confirms an outcome report — contract 3).
3. Turn-intent veto — ``detect_outcome`` accepts the same-turn turn-intent
   result; when available, its ``application_event`` field decides whether a
   keyword hit is a real outcome report. Fail-open: verdict unavailable ⇒
   named-company hits still write (pre-veto behavior), unnamed hits don't.
"""

import json

import pytest

from agent.milestone_detector import detect_interview, detect_outcome

from tests.agent.test_milestone_detector import _apprec, _seed


def _ti(application_event=None, checked=True):
    """Minimal same-turn turn-intent result as the gateway would pass it."""
    out = {"checked": checked, "dispatch_type": "none"}
    if application_event is not None:
        out["application_event"] = application_event
    return out


def _one_app(tmp_path):
    return _seed(tmp_path, [
        _apprec("polymarket", "submitted", submitted_at="2026-07-17"),
    ])


def _two_apps(tmp_path):
    return _seed(tmp_path, [
        _apprec("polymarket", "submitted", submitted_at="2026-07-17"),
        _apprec("impiricus", "submitted", submitted_at="2026-07-21"),
    ])


class TestOutcomeUnnamedInnocentPhrases:
    """The B-0722-02 incident class: no company named, signal substring hit,
    most-recent fallback fabricates a rejection. All must return None."""

    @pytest.mark.parametrize("text", [
        # 2026-07-17 T11 verbatim (wrote polymarket:rejected on dev)
        "wait, stripe? i thought we said no big corps lol. was that in the list from this morning?",
        # 2026-07-22 local live repro (wrote impiricus:rejected)
        "no thanks",
        "no thanks, i'll skip hrcap for today",
        "i passed on that one, didn't feel right",
        "i didn't get around to the review yesterday",
        "i did not get much sleep last night honestly",
        "i'm not moving forward with the bootcamp idea",
        "moving forward with other priorities this week",
        "i always feel rejected when recruiters ghost lol",
        "my landlord turned me down for the lease extension",
        "my roommate went with someone else as a reference",
    ])
    def test_unnamed_innocent_chat_writes_nothing(self, tmp_path, text):
        ud = _two_apps(tmp_path)
        assert detect_outcome(text, ud) is None

    def test_unnamed_hit_without_verdict_does_not_guess(self, tmp_path):
        # Even a genuinely-shaped unnamed report must not guess a company
        # when no turn-intent verdict is available to confirm it.
        ud = _two_apps(tmp_path)
        assert detect_outcome("they turned me down :(", ud) is None


class TestOutcomeTurnIntentVeto:
    """Same-turn turn-intent verdict gates the write (B-0722-02 fix ④)."""

    def test_named_hit_vetoed_when_verdict_says_not_outcome(self, tmp_path):
        ud = _one_app(tmp_path)
        # "i passed on polymarket" = the user declining, not the company.
        res = detect_outcome(
            "i passed on polymarket honestly, not for me", ud,
            turn_intent=_ti(application_event="none"),
        )
        assert res is None

    def test_named_genuine_report_confirmed_by_verdict_fires(self, tmp_path):
        ud = _one_app(tmp_path)
        res = detect_outcome(
            "polymarket turned me down, got the email this morning", ud,
            turn_intent=_ti(application_event="outcome"),
        )
        assert res is not None and res["company"] == "polymarket" and res["result"] == "rejected"

    def test_unnamed_genuine_report_confirmed_by_verdict_falls_back(self, tmp_path):
        # Double-keyed fallback: keyword hit + LLM confirms an outcome report
        # but no company named → most-recent guess is allowed again.
        ud = _two_apps(tmp_path)
        res = detect_outcome(
            "ugh. they turned me down, got the email this morning", ud,
            turn_intent=_ti(application_event="outcome"),
        )
        assert res is not None and res["company"] == "impiricus" and res["result"] == "rejected"

    def test_named_hit_fails_open_when_verdict_unavailable(self, tmp_path):
        # Aux LLM down (checked=False) → named hits keep pre-veto behavior.
        ud = _one_app(tmp_path)
        res = detect_outcome(
            "polymarket turned me down", ud,
            turn_intent=_ti(checked=False),
        )
        assert res is not None and res["company"] == "polymarket" and res["result"] == "rejected"

    def test_unnamed_hit_with_unavailable_verdict_does_not_guess(self, tmp_path):
        ud = _one_app(tmp_path)
        res = detect_outcome(
            "they turned me down", ud,
            turn_intent=_ti(checked=False),
        )
        assert res is None


class TestOutcomeSignalTablePrune:
    """`no thanks` is a Coach-offer refusal shape, never a rejection report on
    its own — pruned from the table entirely (B-0722-02 fix ③, narrowed)."""

    def test_no_thanks_never_fires_even_named_and_confirmed(self, tmp_path):
        ud = _one_app(tmp_path)
        res = detect_outcome(
            "no thanks on the polymarket follow-up, i'll wait", ud,
            turn_intent=_ti(application_event="outcome"),
        )
        assert res is None


class TestOutcomeGenuineRecallRegression:
    """The original recall cases must keep firing (named + no verdict =
    fail-open path; these are the spec's canonical shapes)."""

    @pytest.mark.parametrize("text,key", [
        ("glossier said no", "glossier"),
        ("topicals passed on me", "topicals"),
        ("didn't get the glossier role", "glossier"),
        ("glossier — moving forward with other candidates", "glossier"),
    ])
    def test_named_genuine_reports_still_fire(self, tmp_path, text, key):
        ud = _seed(tmp_path, [_apprec("glossier", "interviewed"),
                              _apprec("topicals", "submitted")])
        res = detect_outcome(text, ud)
        assert res is not None
        assert res["company"] == key
        assert res["result"] == "rejected"


class TestInterviewWordBoundary:
    """Interview signals must match whole words — bare substrings inside
    larger words fired on 'sunscreen'/'screenshot' (B-0722-02 fix ①)."""

    @pytest.mark.parametrize("text", [
        "sunscreen weather in boston finally",
        "let me send you a screenshot of the portal",
        "i'll screen-share the JD later",
    ])
    def test_substring_inside_word_does_not_fire(self, tmp_path, text):
        ud = _one_app(tmp_path)
        assert detect_interview(text, ud) is None

    def test_real_screen_report_still_fires(self, tmp_path):
        ud = _one_app(tmp_path)
        res = detect_interview("just got out of the polymarket screen", ud)
        assert res is not None and res["company"] == "polymarket"


class TestOutcomeWriteLogging:
    """Every ledger write must be attributable: detect_outcome exposes the
    matched signal + match path so the gateway log line can carry them
    (B-0722-02 fix ⑤)."""

    def test_result_carries_signal_and_match_path(self, tmp_path):
        ud = _one_app(tmp_path)
        res = detect_outcome(
            "polymarket turned me down", ud,
            turn_intent=_ti(application_event="outcome"),
        )
        assert res["signal"] == "turned me down"
        assert res["match"] == "named"

    def test_fallback_path_labelled(self, tmp_path):
        ud = _two_apps(tmp_path)
        res = detect_outcome(
            "they turned me down this morning", ud,
            turn_intent=_ti(application_event="outcome"),
        )
        assert res["match"] == "fallback"


class TestTurnIntentApplicationEventField:
    """The turn-intent detector emits `application_event` (B-0722-02 fix ④):
    strict whitelist, default "none", malformed LLM values degrade to "none"
    (a bad value must never enable a ledger write)."""

    def test_field_parsed_when_valid(self, monkeypatch):
        import agent.turn_intent_detector as tid
        from tests.agent.test_turn_intent_detector import _fake_response
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"application_event": "outcome", '
                '"confidence": "high", "reasoning": "rejection report"}'
            ),
            raising=False,
        )
        result = tid.detect_turn_intent("impiricus turned me down")
        assert result["application_event"] == "outcome"

    def test_malformed_value_degrades_to_none(self, monkeypatch):
        import agent.turn_intent_detector as tid
        from tests.agent.test_turn_intent_detector import _fake_response
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"application_event": "REJECTED!!", '
                '"confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        assert tid.detect_turn_intent("hello")["application_event"] == "none"

    def test_absent_field_defaults_to_none(self, monkeypatch):
        import agent.turn_intent_detector as tid
        from tests.agent.test_turn_intent_detector import _fake_response
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _fake_response(
                '{"dispatch_type": "none", "dispatches": [], '
                '"confidence": "high", "reasoning": "x"}'
            ),
            raising=False,
        )
        assert tid.detect_turn_intent("hello")["application_event"] == "none"
