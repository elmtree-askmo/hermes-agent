"""Deterministic resume-solicitation guard for quiet-day briefings (Artemis B-0616-01).

The artemis-briefing quiet-day note re-asked a user for a resume already on file;
the SKILL.md prompt-only guard failed ~2/3 in prod (the model evades banned
literal phrases with synonyms). Fix: a deterministic pre-flight in the briefing
job path — when the user has a resume on file AND the day is genuinely empty
(no follow-up due today, no deadline within the window, no pending action), skip
the write-LLM entirely and emit the fixed quiet-day note, so the model never gets
a chance to solicit. This is the architecture-level enforcement the prompt rule
could not guarantee.

The "genuinely empty" predicate mirrors the server-side date-math in Artemis
`mcp-server/server.py` handle_get_strategy (todays_follow_ups / approaching_deadlines,
~lines 672-806) and `_resume_on_file` (~587-599). Re-implemented fork-local here
(stdlib, fail-open) rather than importing the Artemis server; keep the two in sync.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from cron.scheduler import _quiet_day_resume_short_circuit

pytestmark = pytest.mark.xdist_group("cron_scheduler")

_USER = "U0616TEST"


def _write_state(tmp_path, *, resume: bool, follow_ups=None, action_queue=None):
    """Build ~/.hermes/artemis/<user>/{strategy.json, resumes/} under tmp_path."""
    base = tmp_path / "artemis" / _USER
    base.mkdir(parents=True, exist_ok=True)
    strategy = {
        "follow_ups": follow_ups or [],
        "action_queue": action_queue or [],
        "archive": [],
    }
    (base / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")
    if resume:
        (base / "resumes").mkdir(exist_ok=True)
        (base / "resumes" / "general.json").write_text('{"name": "Test"}', encoding="utf-8")


def _today():
    return datetime.now(timezone.utc).date().isoformat()


def _in_n_days(n):
    return (datetime.now(timezone.utc).date() + timedelta(days=n)).isoformat()


@pytest.fixture
def patch_home(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.scheduler.get_hermes_home", lambda: tmp_path)
    return tmp_path


# --- the bug-exposed case: resume on file + genuinely empty day → short-circuit ---

def test_resume_on_file_and_empty_day_short_circuits(patch_home):
    _write_state(patch_home, resume=True, follow_ups=[], action_queue=[])
    assert _quiet_day_resume_short_circuit(_USER) is True


# --- no resume → must NOT short-circuit (resume nudge is by-design when absent) ---

def test_no_resume_does_not_short_circuit(patch_home):
    _write_state(patch_home, resume=False, follow_ups=[], action_queue=[])
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- resume on file but a follow-up is due today → real content, keep LLM render ---

def test_followup_due_today_does_not_short_circuit(patch_home):
    _write_state(
        patch_home,
        resume=True,
        follow_ups=[{"what": "Did you apply to Waymo?", "when": _today(), "channel": "briefing"}],
    )
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- resume on file but a deadline is within the 2-day window → keep LLM render ---

def test_approaching_deadline_does_not_short_circuit(patch_home):
    _write_state(
        patch_home,
        resume=True,
        action_queue=[{"id": "a1", "status": "pending", "deadline": _in_n_days(1)}],
    )
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- resume on file, a pending action exists (no deadline) → keep LLM render ---

def test_pending_action_does_not_short_circuit(patch_home):
    _write_state(
        patch_home,
        resume=True,
        action_queue=[{"id": "a1", "status": "pending", "deadline": None}],
    )
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- resume on file, only DONE actions + future follow-up → genuinely empty today ---

def test_done_actions_and_future_followup_short_circuits(patch_home):
    _write_state(
        patch_home,
        resume=True,
        follow_ups=[{"what": "future check", "when": _in_n_days(10), "channel": "briefing"}],
        action_queue=[{"id": "a1", "status": "done", "deadline": _in_n_days(1)}],
    )
    assert _quiet_day_resume_short_circuit(_USER) is True


# --- fail-open: missing strategy.json must not crash, returns False (no short-circuit) ---

def test_missing_strategy_fails_open_false(patch_home):
    (patch_home / "artemis" / _USER).mkdir(parents=True, exist_ok=True)
    # resumes present but no strategy.json
    (patch_home / "artemis" / _USER / "resumes").mkdir(exist_ok=True)
    (patch_home / "artemis" / _USER / "resumes" / "general.json").write_text("{}", encoding="utf-8")
    assert _quiet_day_resume_short_circuit(_USER) is False
