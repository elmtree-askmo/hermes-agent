"""S-0626-02 — silence_tier gates the deterministic render layer.

The silence-awareness spec (S-0525-02) prescribes that day8/day5 check-ins push
NO content — "No content push, no roles sections, no attribution lines". But the
silence_tier only gated the LLM write-prompt copy branch; the server-side
attribution + job-card render paths fired unconditionally, so a day8 check-in was
delivered with the full team-activity + job cards prepended (walkthrough Finding
6-1). These tests assert the render layer itself suppresses on a silence tier.

The attribution + job-card render functions take a `silence_tier` argument and
return empty / None for any non-engaged tier (day1/day5/day8), regardless of
qualifying archive entries / job-match artifact.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cron.scheduler import (
    _render_team_attribution_for_briefing,
    _render_job_cards_for_briefing,
    _render_opener,
)

pytestmark = pytest.mark.xdist_group("cron_scheduler")

USER = "U0SIL1"


def _iso(offset_hours):
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).isoformat()


def _setup_strategy(tmp_path, user_id, archive_entries):
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "strategy.json").write_text(json.dumps({
        "user_id": user_id,
        "archive": archive_entries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")


def _setup_jobs(tmp_path, user_id):
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    d = tmp_path / "artemis" / user_id / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"job-match-{today}.json").write_text(json.dumps({
        "scanned_at": _iso(-2),
        "jobs": [{"job_id": "j1", "title": "Senior PM", "company": "Acme",
                  "location": "SF", "url": "https://x/j1", "why": "fit",
                  "match_pct": 78}],
    }), encoding="utf-8")


# A full set of qualifying overnight results — would normally render content.
_RESULTS = [
    {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-3),
     "summary": "Surfaced 3 consumer brand fits"},
    {"id": "a1", "sub_agent": "analyst", "completed_at": _iso(-2),
     "summary": "Rewrote internship bullet to lead with 40% engagement"},
    {"id": "p1", "sub_agent": "publicist", "completed_at": _iso(-1),
     "summary": "Drafted Glossier cover letter"},
]


# ---- attribution gate -------------------------------------------------------


@pytest.mark.parametrize("tier", ["day1", "day5", "day8"])
def test_attribution_suppressed_on_silence_tier(tmp_path, tier):
    """Even with 3 qualifying results, a silence tier renders no attribution."""
    _setup_strategy(tmp_path, USER, _RESULTS)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(USER, silence_tier=tier)
    assert out == ""


def test_attribution_renders_when_engaged(tmp_path):
    """engaged tier (or None) renders the full attribution block — unchanged."""
    _setup_strategy(tmp_path, USER, _RESULTS)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out_engaged = _render_team_attribution_for_briefing(USER, silence_tier="engaged")
        out_none = _render_team_attribution_for_briefing(USER, silence_tier=None)
    assert out_engaged.count("\n") == 2   # 3 lines
    assert out_none.count("\n") == 2


# ---- job-cards gate ---------------------------------------------------------


@pytest.mark.parametrize("tier", ["day1", "day5", "day8"])
def test_job_cards_suppressed_on_silence_tier(tmp_path, tier):
    """Even with a valid job-match artifact, a silence tier renders no cards."""
    _setup_jobs(tmp_path, USER)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_job_cards_for_briefing(USER, silence_tier=tier)
    assert out is None


def test_job_cards_render_when_engaged(tmp_path):
    """engaged tier (or None) renders cards — unchanged."""
    _setup_jobs(tmp_path, USER)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out_engaged = _render_job_cards_for_briefing(USER, silence_tier="engaged")
        out_none = _render_job_cards_for_briefing(USER, silence_tier=None)
    assert out_engaged and len(out_engaged) == 5


# ---- opener gate (S-0626-02 — the render path Phase 1 missed) ----------------


@pytest.mark.parametrize("tier", ["day1", "day5", "day8"])
def test_opener_suppressed_on_silence_tier(tmp_path, tier):
    """Even with 3 qualifying overnight results (which would normally fall back
    to "Morning. Your team ran N things overnight."), a silence tier renders no
    opener — a quiet user's re-entry carries no team-activity greeting."""
    _setup_strategy(tmp_path, USER, _RESULTS)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_opener(USER, llm_opener=None, silence_tier=tier)
    assert out == ""


def test_opener_renders_when_engaged(tmp_path):
    """engaged tier (or None) renders the server-fallback opener — unchanged."""
    _setup_strategy(tmp_path, USER, _RESULTS)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out_engaged = _render_opener(USER, llm_opener=None, silence_tier="engaged")
        out_none = _render_opener(USER, llm_opener=None, silence_tier=None)
    assert out_engaged.startswith("Morning. Your team ran")
    assert out_none.startswith("Morning. Your team ran")
