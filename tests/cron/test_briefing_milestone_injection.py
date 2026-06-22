"""N-day milestone briefing block — server-side render + inject (Artemis S-0601-04).

At 30/60/90 days since signup, the scheduler renders a counts-only milestone
summary from strategy.json archive[] (application_submitted events, typed by
S-0601-02) and prepends it to the morning briefing, ahead of the team-attribution
block. Days-since-signup is read from the onboarding_pushed.flag mtime. Dedup is a
persisted milestones_emitted[] ledger so each N-day mark fires once.

This round is counts-only: the "when we started, your resume had no metrics"
contrast clause is deferred (no start-state capture exists yet — see
docs/specs/milestone-briefing.md § Out of Scope). Mirrors the team-attribution
render/inject structure in test_briefing_attribution_injection.py.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cron.scheduler import (
    _render_milestone_block,
    _inject_milestone_block,
)

pytestmark = pytest.mark.xdist_group("cron_scheduler")


def _setup_user(tmp_path, user_id, archive_entries, days_since_signup,
                milestones_emitted=None):
    """Write strategy.json + onboarding_pushed.flag for a user.

    The flag mtime is backdated `days_since_signup` days so the render path
    computes days-since-signup from it.
    """
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    strategy = {
        "user_id": user_id,
        "archive": archive_entries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if milestones_emitted is not None:
        strategy["milestones_emitted"] = milestones_emitted
    (user_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")

    # S-0622-04: the count source is now applications.json (status >= submitted),
    # not the archive event. Mirror the count of application_submitted archive
    # entries into a submitted ledger so each test's intended count holds.
    n_submitted = sum(
        1 for a in archive_entries
        if isinstance(a, dict) and a.get("event_type") == "application_submitted"
    )
    apps = [{"company": f"co{i}", "display_name": f"Co{i}", "status": "submitted",
             "artifacts": [], "outcome": None} for i in range(n_submitted)]
    (user_dir / "applications.json").write_text(
        json.dumps({"applications": apps,
                    "updated_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8")

    flag = user_dir / "onboarding_pushed.flag"
    flag.write_text("", encoding="utf-8")
    signup_ts = (datetime.now(timezone.utc) - timedelta(days=days_since_signup)).timestamp()
    os.utime(flag, (signup_ts, signup_ts))
    return user_dir


def _app(idx):
    return {
        "id": f"deliver-co{idx}-materials",
        "sub_agent": "publicist",
        "event_type": "application_submitted",
        "artifact_kind": "cover-letter",
        "summary": f"Co{idx} cover letter done",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---- _render_milestone_block ------------------------------------------------


def test_30_day_mark_renders_counts(tmp_path):
    user_id = "U30"
    _setup_user(tmp_path, user_id, [_app(i) for i in range(12)], days_since_signup=30)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert "30 days in" in out
    assert "12" in out
    # counts-only this round — no start-state contrast clause
    assert "when we started" not in out.lower()
    # crediting closer present
    assert "that's all you" in out.lower()


def test_singular_application_phrasing(tmp_path):
    user_id = "U1app"
    _setup_user(tmp_path, user_id, [_app(0)], days_since_signup=30)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert "1 tailored application" in out
    assert "tailored applications" not in out


def test_no_mark_due_before_30_days(tmp_path):
    user_id = "U10"
    _setup_user(tmp_path, user_id, [_app(i) for i in range(12)], days_since_signup=10)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert out == ""


def test_already_emitted_mark_does_not_refire(tmp_path):
    user_id = "Udone"
    _setup_user(tmp_path, user_id, [_app(i) for i in range(12)],
                days_since_signup=35, milestones_emitted=["30d"])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert out == ""


def test_render_marks_emitted_ledger(tmp_path):
    user_id = "Umark"
    user_dir = _setup_user(tmp_path, user_id, [_app(i) for i in range(12)],
                           days_since_signup=30)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        first = _render_milestone_block(user_id)
        assert first != ""
        # ledger persisted
        strategy = json.loads((user_dir / "strategy.json").read_text())
        assert "30d" in strategy.get("milestones_emitted", [])
        # second call same day → already emitted → empty
        second = _render_milestone_block(user_id)
    assert second == ""


def test_60_day_mark_fires_when_30_already_emitted(tmp_path):
    user_id = "U60"
    _setup_user(tmp_path, user_id, [_app(i) for i in range(20)],
                days_since_signup=62, milestones_emitted=["30d"])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert "60 days in" in out
    assert "20" in out


def test_highest_due_unemitted_mark_chosen(tmp_path):
    # A user past 90 days who never got any milestone (e.g. emitted ledger empty)
    # fires the highest due mark, not 30 — lower marks are implicitly past.
    user_id = "U90"
    _setup_user(tmp_path, user_id, [_app(i) for i in range(25)], days_since_signup=95)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert "90 days in" in out


def test_zero_applications_still_renders_at_mark(tmp_path):
    # Day 30 with no apps: the milestone still marks (the mark is time-based),
    # but with no apps the block is suppressed — a "0 apps" summary is not a
    # milestone worth voicing. Empty render, mark still recorded so it doesn't
    # nag later.
    user_id = "U0"
    user_dir = _setup_user(tmp_path, user_id, [], days_since_signup=30)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert out == ""
    strategy = json.loads((user_dir / "strategy.json").read_text())
    assert "30d" in strategy.get("milestones_emitted", [])


def test_fail_open_missing_strategy(tmp_path):
    user_id = "Umissing"
    # onboarding flag but no strategy.json
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "onboarding_pushed.flag").write_text("", encoding="utf-8")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert out == ""


def test_fail_open_corrupt_strategy(tmp_path):
    user_id = "Ucorrupt"
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "strategy.json").write_text("{not json", encoding="utf-8")
    (user_dir / "onboarding_pushed.flag").write_text("", encoding="utf-8")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert out == ""


def test_fail_open_no_onboarding_flag(tmp_path):
    # No onboarding flag → no signup date → cannot compute a mark → empty.
    user_id = "Unoflag"
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    strategy = {"user_id": user_id, "archive": [_app(i) for i in range(12)]}
    (user_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert out == ""


def test_non_application_events_not_counted(tmp_path):
    user_id = "Umixed"
    archive = [
        _app(0),
        {"id": "x", "sub_agent": "publicist", "event_type": None,
         "summary": "follow-up email", "completed_at": datetime.now(timezone.utc).isoformat()},
        {"id": "y", "sub_agent": "scout", "summary": "scan",
         "completed_at": datetime.now(timezone.utc).isoformat()},
    ]
    _setup_user(tmp_path, user_id, archive, days_since_signup=30)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_milestone_block(user_id)
    assert "1 tailored application" in out


# ---- _inject_milestone_block ------------------------------------------------


def test_inject_prepends_ahead_of_attribution(tmp_path):
    milestone = "30 days in. You've sent 12 tailored applications. That's all you — we just made sure nobody missed it."
    attribution = "🔍 *Scout* — found 3 roles."
    briefing = "Morning. Here's today."
    # milestone goes first, then attribution, then body
    combined = _inject_milestone_block(briefing, milestone)
    assert combined.startswith(milestone)
    assert briefing in combined


def test_inject_empty_milestone_unchanged(tmp_path):
    briefing = "Morning. Here's today."
    assert _inject_milestone_block(briefing, "") == briefing


def test_inject_empty_briefing_returns_milestone(tmp_path):
    milestone = "30 days in. You've sent 8 tailored applications. That's all you — we just made sure nobody missed it."
    assert _inject_milestone_block("", milestone) == milestone
