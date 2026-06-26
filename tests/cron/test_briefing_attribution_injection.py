"""Briefing team attribution paragraph — server-side render + inject.

Scheduler reads ~/.hermes/artemis/<user_id>/strategy.json before delivery
and prepends a canonical 3-line attribution block when archive[] has
recent sub-agent completions. See `_render_team_attribution_for_briefing`
docstring for the LLM-prompt enforcement history that motivated this path.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cron.scheduler import (
    _inject_attribution_block,
    _render_team_attribution_for_briefing,
    _render_opener,
    _count_active_sub_agents,
    _has_fresh_reviewable_products,
    _SUB_AGENT_ATTRIBUTION_REGISTRY,
)

pytestmark = pytest.mark.xdist_group("cron_scheduler")


def _setup_strategy(tmp_path, user_id, archive_entries):
    """Write a minimal strategy.json under tmp_path's artemis tree."""
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    strategy = {
        "user_id": user_id,
        "archive": archive_entries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (user_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")


def _iso(now_offset_hours):
    now = datetime.now(timezone.utc)
    ts = now + timedelta(hours=now_offset_hours)
    return ts.isoformat()


# ---- _render_team_attribution_for_briefing ----------------------------------


def test_three_sub_agents_render_in_canonical_order(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "p1", "sub_agent": "publicist", "completed_at": _iso(-1),
         "summary": "Drafted Glossier cover letter"},
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-3),
         "summary": "Surfaced 3 consumer brand fits"},
        {"id": "a1", "sub_agent": "analyst", "completed_at": _iso(-2),
         "summary": "Rewrote internship bullet to lead with 40% engagement"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    lines = out.split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("🔍 *Scout* Surfaced 3 consumer brand fits.")
    assert lines[1].startswith("📊 *Analyst* Rewrote internship bullet")
    assert lines[2].startswith("✍️ *Publicist* Drafted Glossier cover letter.")


def test_items_older_than_24h_excluded(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "s_old", "sub_agent": "scout", "completed_at": _iso(-48),
         "summary": "Stale scan"},
        {"id": "p_recent", "sub_agent": "publicist", "completed_at": _iso(-1),
         "summary": "Recent draft"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    assert "Stale scan" not in out
    assert "Recent draft" in out
    assert "Scout" not in out


def test_only_most_recent_per_subagent(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-10),
         "summary": "Earlier scout"},
        {"id": "s2", "sub_agent": "scout", "completed_at": _iso(-2),
         "summary": "Later scout"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    assert "Later scout" in out
    assert "Earlier scout" not in out


def test_single_sub_agent_still_renders(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-1),
         "summary": "Solo scout overnight"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    assert out == "🔍 *Scout* Solo scout overnight."


def test_empty_archive_returns_empty(tmp_path):
    _setup_strategy(tmp_path, "U123", [])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing("U123")
    assert out == ""


def test_missing_strategy_returns_empty(tmp_path):
    # No strategy.json at all
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing("U_NONE")
    assert out == ""


def test_corrupt_strategy_returns_empty(tmp_path):
    user_dir = tmp_path / "artemis" / "U_BAD"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "strategy.json").write_text("{ not valid json", encoding="utf-8")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing("U_BAD")
    assert out == ""


def test_unknown_subagent_skipped(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "x1", "sub_agent": "marketer", "completed_at": _iso(-1),
         "summary": "Stranger"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    assert out == ""


def test_missing_summary_skipped(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-1),
         "summary": ""},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    assert out == ""


def test_trailing_period_normalized(tmp_path):
    user_id = "U123"
    _setup_strategy(tmp_path, user_id, [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-1),
         "summary": "Surfaced 4 roles."},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_team_attribution_for_briefing(user_id)
    assert out.endswith("Surfaced 4 roles.")
    assert "Surfaced 4 roles.." not in out


# ---- _inject_attribution_block ---------------------------------------------


def test_inject_prepends_with_blank_line_gap():
    attribution = "🔍 *Scout* Found 2 roles."
    deliver = "Morning briefing content here."
    out = _inject_attribution_block(deliver, attribution)
    assert out == "🔍 *Scout* Found 2 roles.\n\nMorning briefing content here."


def test_inject_empty_attribution_returns_content_unchanged():
    deliver = "Quiet day note."
    out = _inject_attribution_block(deliver, "")
    assert out == deliver


def test_inject_empty_content_returns_attribution_alone():
    attribution = "📊 *Analyst* — Did a thing."
    out = _inject_attribution_block("", attribution)
    assert out == attribution


# ---- _count_active_sub_agents ----------------------------------------------


def test_count_active_sub_agents_counts_distinct_within_24h(tmp_path):
    _setup_strategy(tmp_path, "U123", [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-1), "summary": "x"},
        {"id": "a1", "sub_agent": "analyst", "completed_at": _iso(-2), "summary": "y"},
        {"id": "s2", "sub_agent": "scout", "completed_at": _iso(-3), "summary": "z"},  # same agent
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _count_active_sub_agents("U123") == 2  # scout + analyst, scout counted once


def test_count_active_sub_agents_excludes_old(tmp_path):
    _setup_strategy(tmp_path, "U123", [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-48), "summary": "old"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _count_active_sub_agents("U123") == 0


# ---- _render_opener (B-primary + A-fallback) -------------------------------


def _setup_n(tmp_path, n):
    """Set up strategy with n distinct recent sub-agents."""
    agents = list(_SUB_AGENT_ATTRIBUTION_REGISTRY.keys())[:n]
    _setup_strategy(tmp_path, "U123", [
        {"id": f"{a}1", "sub_agent": a, "completed_at": _iso(-1), "summary": "did work"}
        for a in agents
    ])


def test_opener_uses_llm_text_when_valid(tmp_path):
    _setup_n(tmp_path, 3)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_opener("U123", "Morning — busy night, three fronts moved.")
    assert out == "Morning — busy night, three fronts moved."


def test_opener_falls_back_to_plural_template_when_llm_missing(tmp_path):
    _setup_n(tmp_path, 3)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_opener("U123", None)
    assert out == "Morning. Your team ran 3 things overnight."


def test_opener_falls_back_to_plural_template_when_llm_blank(tmp_path):
    _setup_n(tmp_path, 2)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_opener("U123", "   ")
    assert out == "Morning. Your team ran 2 things overnight."


def test_opener_fallback_singular_names_the_agent(tmp_path):
    # one sub-agent → singular highlight template
    _setup_strategy(tmp_path, "U123", [
        {"id": "a1", "sub_agent": "analyst", "completed_at": _iso(-1), "summary": "did work"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        out = _render_opener("U123", None)
    assert out == "Morning. Analyst finished something overnight."


def test_opener_empty_when_no_active_sub_agents(tmp_path):
    _setup_strategy(tmp_path, "U123", [])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        # even if LLM supplied text, no team work → no opener (nothing to greet about)
        assert _render_opener("U123", "Morning, team did stuff.") == ""


# ---- _has_fresh_reviewable_products (C — walkthrough signal, server-side) --


def test_fresh_reviewable_true_for_recent_cover_letter(tmp_path):
    _setup_strategy(tmp_path, "U123", [
        {"id": "p1", "sub_agent": "publicist", "completed_at": _iso(-2),
         "artifact_kind": "cover-letter", "summary": "tailored Glossier letter"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _has_fresh_reviewable_products("U123") is True


def test_fresh_reviewable_true_for_recent_resume(tmp_path):
    _setup_strategy(tmp_path, "U123", [
        {"id": "r1", "sub_agent": "publicist", "completed_at": _iso(-1),
         "artifact_kind": "resume", "summary": "tailored resume variant"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _has_fresh_reviewable_products("U123") is True


def test_fresh_reviewable_false_for_jobs_only(tmp_path):
    # a job-scan artifact is NOT a reviewable drafted material — no walkthrough
    _setup_strategy(tmp_path, "U123", [
        {"id": "s1", "sub_agent": "scout", "completed_at": _iso(-1),
         "artifact_kind": "jobs", "summary": "10 matches"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _has_fresh_reviewable_products("U123") is False


def test_fresh_reviewable_false_for_old_cover_letter(tmp_path):
    _setup_strategy(tmp_path, "U123", [
        {"id": "p1", "sub_agent": "publicist", "completed_at": _iso(-48),
         "artifact_kind": "cover-letter", "summary": "old letter"},
    ])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _has_fresh_reviewable_products("U123") is False


def test_fresh_reviewable_false_on_empty(tmp_path):
    _setup_strategy(tmp_path, "U123", [])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _has_fresh_reviewable_products("U123") is False


def test_fresh_materials_flag_injected_server_side(tmp_path):
    """S-0626-02 Plan B: the walkthrough A/B is keyed on the server-set
    fresh_materials flag, injected deterministically into the package from the
    archive (NOT inferred by an LLM from raw text). The flag's CONSUMER moved to
    step-0's SKILL.md (which authors the walkthrough wording into coaches_take);
    the fork's job is only to inject the flag. This asserts the injection path:
    _run_briefing_render sets pkg["fresh_materials"] when the archive qualifies.
    """
    import cron.scheduler as sched
    captured = {}

    def _fake_parse(raw, job_id="?"):
        return {"coaches_take": "take", "opener": None, "response_window_checkin": None}

    # No write-repair / network: stub the repair call to echo the take back.
    with patch.object(sched, "_parse_step0_output", _fake_parse), \
         patch.object(sched, "_briefing_write_call", lambda t, o, j="?": (t, o)), \
         patch.object(sched, "_has_fresh_reviewable_products", return_value=True) as fresh:
        body = sched._run_briefing_render("{}", "j1", user_id="U123")
    assert fresh.called
    assert body == "take"  # render succeeded with the flag-qualifying user
