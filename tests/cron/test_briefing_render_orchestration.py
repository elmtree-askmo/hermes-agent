"""Orchestration tests for _run_briefing_render (S-0626-02 Plan B).

Migrated from the deleted test_two_step_briefing.py — only the three
orchestration cases survived the refactor (parse → name-strip-repair →
assemble). The decide-step and old-write-prompt contract tests were dropped;
their replacement coverage lives in test_briefing_step0_parse.py (parse) and
test_briefing_repair_assembly.py (assembly).
"""
import pytest

from cron.scheduler import _run_briefing_render

pytestmark = pytest.mark.xdist_group("cron_scheduler")


# A raw step-0 JSON output (what _parse_step0_output consumes). The orchestration
# tests below patch _parse_step0_output directly, so the raw text only needs to
# be a non-empty string for the fail-open case to have a take to fall back to.
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


def test_render_happy_path_returns_assembled_body(monkeypatch):
    """Parse + repair succeed → returns the assembled body containing the
    repaired take and the verbatim check-in."""
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": dict(_RAW_PKG))
    monkeypatch.setattr(
        "cron.scheduler._briefing_write_call",
        lambda take, opener, job_id="?": ("Day 1 of the wait — patience is the strategy.", "Coming back to where we left off"),
    )
    result = _run_briefing_render(_RAW_JSON, "test-happy")
    assert result is not None
    assert "patience is the strategy" in result
    # the response-window check-in is appended verbatim as its own beat
    assert "Day 1 of your 48-hour window." in result


def test_render_parse_failure_returns_none(monkeypatch):
    """When step-0 output fails to parse, the orchestrator returns None so the
    caller falls back to the Phase-5 voice-scan path on the raw output."""
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": None)
    result = _run_briefing_render("not json", "test-parse-fail")
    assert result is None


def test_render_write_repair_failure_fails_open_to_raw_take(monkeypatch):
    """When the name-strip repair call fails, the orchestrator FAILS OPEN to the
    raw step-0 take rather than returning None."""
    monkeypatch.setattr("cron.scheduler._parse_step0_output", lambda text, job_id="?": dict(_RAW_PKG))
    monkeypatch.setattr(
        "cron.scheduler._briefing_write_call",
        lambda take, opener, job_id="?": (None, None),
    )
    result = _run_briefing_render(_RAW_JSON, "test-write-fail")
    assert result is not None
    # the RAW (unrepaired) coaches_take still reaches the body
    assert "patience is the strategy" in result
