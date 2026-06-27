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


def _write_state(tmp_path, *, resume: bool, follow_ups=None, action_queue=None, applications=None,
                 onboarded=False, briefing_delivered=True):
    """Build ~/.hermes/artemis/<user>/{strategy.json, resumes/, applications.json} under tmp_path.

    onboarded — write onboarding_pushed.flag (user finished onboarding).
    briefing_delivered — write a prior briefings/*.json (a briefing has already been delivered).
        Default True so existing tests keep their "steady-state" semantics; the
        first-briefing exemption only fires when onboarded AND NOT briefing_delivered.
    """
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
    if applications is not None:
        (base / "applications.json").write_text(
            json.dumps({"applications": applications}), encoding="utf-8"
        )
    if onboarded:
        (base / "onboarding_pushed.flag").write_text("", encoding="utf-8")
    if briefing_delivered:
        bdir = base / "briefings"
        bdir.mkdir(exist_ok=True)
        (bdir / "2026-01-01T00-00-00Z.json").write_text(
            json.dumps({"user_id": _USER, "briefing_timestamp": "2026-01-01T00:00:00Z",
                        "formatted_output": "prior briefing"}), encoding="utf-8"
        )


def _ready_material_app(company="jerry", submitted_at=None, status="identified"):
    """An applications.json entry with a tailored resume artifact (ready material)."""
    return {
        "company": company,
        "status": status,
        "submitted_at": submitted_at,
        "artifacts": [{"kind": "resume", "name": f"{company}-data-scientist"}],
    }


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


# --- P-0627-01: the FIRST briefing after onboarding must NOT be suppressed ---
# Even on an otherwise-empty day, the first post-onboarding briefing should let the
# decide-step's real first-briefing framing through (the ranked scan is mid-flight,
# the market is active) — not the generic quiet-day note. Exemption = onboarded AND
# no briefing delivered yet.

def test_first_briefing_after_onboarding_does_not_short_circuit(patch_home):
    _write_state(patch_home, resume=True, follow_ups=[], action_queue=[],
                 onboarded=True, briefing_delivered=False)
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- but once a briefing HAS been delivered, an empty day reverts to short-circuit
# (the exemption is one-shot; it must not become a permanent resume-solicit loophole) ---

def test_empty_day_after_first_briefing_short_circuits(patch_home):
    _write_state(patch_home, resume=True, follow_ups=[], action_queue=[],
                 onboarded=True, briefing_delivered=True)
    assert _quiet_day_resume_short_circuit(_USER) is True


# --- exemption requires the onboarding flag; a pre-onboarding-flag user with no
# briefings yet (legacy/edge) still short-circuits on an empty day ---

def test_no_onboarding_flag_no_briefing_still_short_circuits(patch_home):
    _write_state(patch_home, resume=True, follow_ups=[], action_queue=[],
                 onboarded=False, briefing_delivered=False)
    assert _quiet_day_resume_short_circuit(_USER) is True


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


# --- ready-but-unsubmitted materials must NOT short-circuit (the Scene-2 bug) ---
# Regression: all tailor actions completed (action_queue empty) + future follow-up,
# BUT applications.json has tailored resumes ready and not yet submitted. The old
# guard saw "empty queue + future follow-up" and emitted "Nothing urgent" — wrong,
# the highest-value push is "your materials are ready, go submit". Surfaced live
# 2026-06-25 (Maya walkthrough) once B-0624-02 let distinct-company tailors all
# complete, first time the queue went genuinely empty with materials staged.

def test_ready_unsubmitted_materials_does_not_short_circuit(patch_home):
    _write_state(
        patch_home,
        resume=True,
        action_queue=[],  # all tailors done
        follow_ups=[{"what": "check submissions", "when": _in_n_days(7), "channel": "briefing"}],
        applications=[
            _ready_material_app("jerry", submitted_at=None),
            _ready_material_app("scale-jobs", submitted_at=None),
        ],
    )
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- once materials are submitted, the day IS quiet again → short-circuit ok ---

def test_all_materials_submitted_short_circuits(patch_home):
    _write_state(
        patch_home,
        resume=True,
        action_queue=[],
        follow_ups=[{"what": "check", "when": _in_n_days(7), "channel": "briefing"}],
        applications=[
            _ready_material_app("jerry", submitted_at=_today(), status="submitted"),
        ],
    )
    assert _quiet_day_resume_short_circuit(_USER) is True


# --- application without a resume artifact (identified only) is not "ready" → quiet ok ---

def test_identified_without_artifact_short_circuits(patch_home):
    _write_state(
        patch_home,
        resume=True,
        action_queue=[],
        follow_ups=[{"what": "check", "when": _in_n_days(7), "channel": "briefing"}],
        applications=[{"company": "jerry", "status": "identified", "submitted_at": None, "artifacts": []}],
    )
    assert _quiet_day_resume_short_circuit(_USER) is True


# --- fail-open: missing strategy.json must not crash, returns False (no short-circuit) ---

def test_missing_strategy_fails_open_false(patch_home):
    (patch_home / "artemis" / _USER).mkdir(parents=True, exist_ok=True)
    # resumes present but no strategy.json
    (patch_home / "artemis" / _USER / "resumes").mkdir(exist_ok=True)
    (patch_home / "artemis" / _USER / "resumes" / "general.json").write_text("{}", encoding="utf-8")
    assert _quiet_day_resume_short_circuit(_USER) is False


# --- Integration: drive tick() end to end and assert the short-circuit wires up ---
# The predicate tests above prove the boolean; these prove the tick() wiring
# actually swaps deliver_content to the fixed note AND skips the write-LLM,
# which the predicate-only tests cannot catch (the wiring bug surfaces only here).

from unittest.mock import patch

_RESUME_SOLICIT_BRIEFING = (
    "Quiet week on the board — graduation's close. When you're ready, "
    "drop your resume here and we'll get matching."
)


def _briefing_job():
    return {
        "id": "JTEST",
        "name": "daily-briefing",
        "skills": ["artemis-briefing"],
        "prompt": "Run artemis-briefing",
        "origin": {"platform": "slack", "chat_id": "D1", "user_id": _USER},
    }


def _run_tick_capturing(predicate_value):
    """Drive tick() with run_job stubbed to return a resume-soliciting briefing.
    Returns (delivered_content, two_step_called)."""
    captured = {}
    two_step_called = {"n": 0}

    def _fake_deliver(job, content, adapters=None, loop=None):
        captured["content"] = content
        return None

    def _fake_two_step(content, job_id, silence_tier=None, capture=None, user_id=None):
        two_step_called["n"] += 1
        return content  # pretend write-LLM passed it through unchanged

    with patch("cron.scheduler.get_due_jobs", return_value=[_briefing_job()]), \
         patch("cron.scheduler.advance_next_run"), \
         patch("cron.scheduler.save_job_output", return_value="/tmp/x"), \
         patch("cron.scheduler.mark_job_run"), \
         patch("cron.scheduler.run_job",
               return_value=(True, "doc", _RESUME_SOLICIT_BRIEFING, None)), \
         patch("cron.scheduler._quiet_day_resume_short_circuit",
               return_value=predicate_value), \
         patch("cron.scheduler._run_briefing_render", side_effect=_fake_two_step), \
         patch("cron.scheduler._voice_scan_check", return_value=(True, "")), \
         patch("cron.scheduler._deliver_result", side_effect=_fake_deliver):
        from cron.scheduler import tick
        tick(verbose=False)
    return captured.get("content"), two_step_called["n"]


def test_tick_short_circuits_to_fallback_when_predicate_true():
    from cron.scheduler import _quiet_day_fallback
    delivered, two_step_n = _run_tick_capturing(predicate_value=True)
    # the resume-soliciting briefing must NOT reach the user
    assert delivered == _quiet_day_fallback()
    assert "resume" not in (delivered or "").lower()
    # and the write-LLM (two-step) must be skipped entirely
    assert two_step_n == 0


def test_tick_keeps_llm_path_when_predicate_false():
    delivered, two_step_n = _run_tick_capturing(predicate_value=False)
    # predicate False → normal path: two-step runs, original content flows
    assert two_step_n == 1
    # The render output flows through unchanged; the server-side onboarding
    # footer (S-0626-02) is appended below it on the first 3 runs (this job has
    # no repeat block → completed=0). Assert the body survived rather than exact
    # equality — the footer is covered by TestBuildJobPromptArtemisFooter.
    assert delivered.startswith(_RESUME_SOLICIT_BRIEFING)
