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
