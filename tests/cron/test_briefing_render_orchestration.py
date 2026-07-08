"""Orchestration tests for _run_briefing_render (S-0626-02 Plan C).

Plan C removed the name-strip repair LLM. _run_briefing_render now parses
step-0's JSON and returns `coaches_take` as the body, stashing the opener and
the verbatim response-window check-in on the capture dict for the scheduler to
render around the voice-scan. The check-in is NOT folded into the returned body
(the voice-scan judges only the take beat; the check-in is appended after the
scan by the scheduler — see _run_briefing_render docstring).

Parse-shape coverage lives in test_briefing_step0_parse.py.
"""
import pytest

from cron.scheduler import _run_briefing_render

pytestmark = pytest.mark.xdist_group("cron_scheduler")


# A parsed step-0 package (what _parse_step0_output returns). The orchestration
# tests patch _parse_step0_output directly, so the raw text only needs to be a
# non-empty string.
_RAW_PKG = {
    "coaches_take": "Day 1 of the wait — patience is the strategy. Want me to (A) keep monitoring, or (B) draft a nudge?",
    "opener": "Coming back to where we left off",
    "response_window_checkin": "Day 1 of your 48-hour window.",
}
_RAW_JSON = (
    '{"coaches_take": "Day 1 of the wait — patience is the strategy. '
    'Want me to (A) keep monitoring, or (B) draft a nudge?", '
    '"opener": "Coming back to where we left off", '
    '"response_window_checkin": "Day 1 of your 48-hour window."}'
)


def test_render_returns_take_body_and_captures_opener_and_checkin(monkeypatch):
    """Parse succeeds → returns coaches_take as the body, and captures the
    opener + the verbatim check-in (NOT folded into the returned body, so the
    voice-scan judges only the take)."""
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": dict(_RAW_PKG))
    capture: dict = {}
    result = _run_briefing_render(_RAW_JSON, "test-happy", capture=capture)
    assert result is not None
    assert "patience is the strategy" in result
    # the check-in is captured for post-scan append, NOT in the returned body
    assert "Day 1 of your 48-hour window." not in result
    assert capture["checkin"] == "Day 1 of your 48-hour window."
    assert capture["opener"] == "Coming back to where we left off"


def test_render_no_checkin_captures_none(monkeypatch):
    """A package with no response-window check-in captures checkin=None."""
    pkg = dict(_RAW_PKG)
    pkg["response_window_checkin"] = None
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": pkg)
    capture: dict = {}
    result = _run_briefing_render(_RAW_JSON, "test-no-checkin", capture=capture)
    assert result is not None
    assert capture["checkin"] is None


def test_render_parse_failure_returns_none(monkeypatch):
    """When step-0 output fails to parse, the orchestrator returns None so the
    caller falls back to the Phase-5 voice-scan path on the raw output."""
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": None)
    result = _run_briefing_render("not json", "test-parse-fail")
    assert result is None


def test_render_empty_take_returns_none(monkeypatch):
    """An empty coaches_take returns None (falls back to Phase-5 path) rather
    than delivering an empty body."""
    pkg = dict(_RAW_PKG)
    pkg["coaches_take"] = "   "
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": pkg)
    result = _run_briefing_render(_RAW_JSON, "test-empty-take")
    assert result is None


# ---------------------------------------------------------------------------
# B-0510-01 reopen (2026-07-08) — step-0 contract violation must FAIL CLOSED.
#
# Observed on dev: step-0 emitted its own intermediate (model preamble
# reasoning + labeled fields) instead of the JSON contract, and the scheduler
# delivered it raw to Slack, gated only by the log-only voice-scan. The raw
# non-JSON output is a pipeline intermediate, never a deliverable —
# _briefing_deliverable must substitute the deterministic quiet-day note.
# ---------------------------------------------------------------------------

# Tonight's real leak shape (abridged): reasoning prose + decide-field dump.
_DECIDE_DUMP = (
    "The skill instructions specify no web search for roles -- only for "
    "events. Given this is a briefing at 1:55 AM local time...\n\n"
    "I'll structure the coaches_take to address the Brigham submission "
    "nudge...\n\n"
    "coaches_take: The Brigham materials are still standing by from Tuesday...\n"
    "opener: Keeping eyes on healthcare and mission-driven data science roles.\n"
    "response_window_checkin: No reply yet from Wayfair -- the response window "
    "has passed. Would you like Publicist to draft a follow-up you can send?"
)


def test_decide_dump_fails_closed_to_fallback():
    """Non-JSON step-0 output (the leak shape) must yield the deterministic
    fallback note — never the raw intermediate."""
    from cron.scheduler import _briefing_deliverable, _quiet_day_fallback

    result = _briefing_deliverable(_DECIDE_DUMP, "test-b0510-leak")
    assert result == _quiet_day_fallback()
    assert "coaches_take:" not in result
    assert "skill instructions" not in result


def test_valid_json_still_renders_take():
    """Happy path unchanged: valid step-0 JSON renders the take body."""
    from cron.scheduler import _briefing_deliverable

    capture: dict = {}
    result = _briefing_deliverable(_RAW_JSON, "test-b0510-happy", capture=capture)
    assert result.startswith("Day 1 of the wait")
    assert capture["checkin"] == "Day 1 of your 48-hour window."


def test_empty_step0_output_fails_closed_to_fallback():
    """Empty/whitespace step-0 output also fails closed (same contract gate)."""
    from cron.scheduler import _briefing_deliverable, _quiet_day_fallback

    assert _briefing_deliverable("   ", "test-b0510-empty") == _quiet_day_fallback()
