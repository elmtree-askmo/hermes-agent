"""Artemis S-0604-01 Phase B — New Roles Block Kit card render + read.

Scheduler reads TODAY's job-match artifact (`jobs/job-match-<date>.json`, a
per-day overwrite file) and renders send_jobs-shaped Block Kit cards, posted
as a separate message after the briefing text (bypassing Phase 6). Reading
today's date (not a rolling window) means a stale prior-day artifact is never
surfaced.
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import cron.scheduler as sch
from cron.scheduler import (
    _build_job_card_blocks,
    _deliver_job_cards,
    _match_bar,
    _render_job_cards_for_briefing,
)

pytestmark = pytest.mark.xdist_group("cron_scheduler")

USER = "U0CARD1"
VALID_JOB = {
    "job_id": "j1", "title": "Senior PM", "company": "Acme",
    "location": "SF", "url": "https://x/j1", "why": "fit", "match_pct": 78,
}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _write_jobs(tmp_path, jobs, user=USER, date=None):
    date = date or _today()
    d = tmp_path / "artemis" / user / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"job-match-{date}.json").write_text(
        json.dumps({"scanned_at": "2026-06-11T00:00:00+00:00", "jobs": jobs})
    )


# ---- _match_bar -------------------------------------------------------------


def test_match_bar_bounds():
    assert _match_bar(0) == "░" * 10
    assert _match_bar(100) == "▓" * 10
    assert _match_bar(78) == "▓" * 8 + "░" * 2   # round(7.8) = 8


# ---- _build_job_card_blocks -------------------------------------------------


def test_card_blocks_structure_and_actions():
    blocks = _build_job_card_blocks([VALID_JOB])
    # title section + company context + match section + why section + actions
    assert len(blocks) == 5
    assert blocks[0]["text"]["text"] == "*Senior PM*"
    actions = blocks[-1]["elements"]
    aids = [e.get("action_id") for e in actions]
    assert "job_save" in aids and "job_skip" in aids       # reuse global handlers
    view = [e for e in actions if e["text"]["text"] == "View posting"][0]
    assert view["url"] == "https://x/j1"
    skip = [e for e in actions if e.get("action_id") == "job_skip"][0]
    assert skip["value"] == "j1"
    save = [e for e in actions if e.get("action_id") == "job_save"][0]
    assert json.loads(save["value"])["job_id"] == "j1"     # handler-expected payload


def test_card_blocks_no_match_pct_skips_bar():
    j = dict(VALID_JOB)
    j.pop("match_pct")
    blocks = _build_job_card_blocks([j])
    assert len(blocks) == 4                                  # match section dropped
    assert not any("Match:" in b.get("text", {}).get("text", "") for b in blocks)


def test_card_blocks_divider_between_jobs():
    blocks = _build_job_card_blocks([VALID_JOB, dict(VALID_JOB, job_id="j2")])
    assert any(b["type"] == "divider" for b in blocks)


# ---- _render_job_cards_for_briefing ----------------------------------------


def test_render_todays_artifact(tmp_path):
    _write_jobs(tmp_path, [VALID_JOB])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        blocks = _render_job_cards_for_briefing(USER)
    assert blocks and len(blocks) == 5


def test_render_none_when_no_today_artifact(tmp_path):
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _render_job_cards_for_briefing(USER) is None


def test_render_ignores_prior_day_artifact(tmp_path):
    # only a stale prior-day file exists → today's read finds nothing
    _write_jobs(tmp_path, [VALID_JOB], date="20200101")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _render_job_cards_for_briefing(USER) is None


def test_render_none_when_empty_jobs(tmp_path):
    _write_jobs(tmp_path, [])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        assert _render_job_cards_for_briefing(USER) is None


def test_render_skips_malformed_entries(tmp_path):
    _write_jobs(tmp_path, [{"title": "no url/company"}, VALID_JOB])
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        blocks = _render_job_cards_for_briefing(USER)
    assert blocks and len(blocks) == 5                       # only the 1 valid job


# ---- cap at 7 jobs / Slack 50-block limit ----------------------------------


def test_render_caps_at_7_jobs_under_50_blocks(tmp_path):
    # 10 worst-case jobs (every optional block present: match_pct + salary)
    jobs = [dict(VALID_JOB, job_id=f"j{i}", salary="$100-200k") for i in range(10)]
    _write_jobs(tmp_path, jobs)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        blocks = _render_job_cards_for_briefing(USER)
    assert blocks is not None
    # Slack rejects a message with >50 blocks; cap 7 keeps it under.
    assert len(blocks) <= 50
    # exactly 7 job cards rendered (title sections start with "*")
    titles = [b for b in blocks
              if b.get("type") == "section"
              and b.get("text", {}).get("text", "").startswith("*")]
    assert len(titles) == 7


# ---- thread under the briefing message -------------------------------------


def test_deliver_job_cards_threads_under_briefing(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, token):
            pass

        async def chat_postMessage(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True, "ts": "1.1"}

    monkeypatch.setattr(sch, "_resolve_delivery_target",
                        lambda job: {"platform": "slack", "chat_id": "D1", "thread_id": None})
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr("slack_sdk.web.async_client.AsyncWebClient", _FakeClient)

    job = {"id": "daily-briefing", "_briefing_msg_ts": "1781172067.053159"}
    blocks = _build_job_card_blocks([VALID_JOB])
    _deliver_job_cards(job, blocks, loop=None)
    # card posts as a reply under the briefing message
    assert captured.get("thread_ts") == "1781172067.053159"
    assert captured.get("channel") == "D1"
    assert captured.get("blocks") == blocks


def test_deliver_job_cards_falls_back_to_origin_thread(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, token):
            pass

        async def chat_postMessage(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True, "ts": "1.1"}

    monkeypatch.setattr(sch, "_resolve_delivery_target",
                        lambda job: {"platform": "slack", "chat_id": "D1", "thread_id": "origin.ts"})
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr("slack_sdk.web.async_client.AsyncWebClient", _FakeClient)

    # no briefing ts captured → fall back to the origin thread_id
    job = {"id": "daily-briefing"}
    _deliver_job_cards(job, _build_job_card_blocks([VALID_JOB]), loop=None)
    assert captured.get("thread_ts") == "origin.ts"


# ---- B-0723-01: cross-renderer button contract ------------------------------
# The briefing-path card renderer (this module) and the artemis send-jobs hook
# build the same card independently; P-0721-03 wired the View click recording
# in the hook only, and this renderer's View button silently stayed a bare url
# button (random Slack action_id -> 404 -> click dropped). These pins keep the
# two copies from drifting again: every button carries an explicit action_id,
# and View's value is the same self-contained payload Save carries.


def _card_buttons():
    blocks = _build_job_card_blocks([VALID_JOB])
    return [
        el
        for b in blocks
        if b.get("type") == "actions"
        for el in b.get("elements", [])
    ]


def test_every_card_button_has_explicit_action_id():
    btns = _card_buttons()
    assert btns, "card rendered no action buttons"
    for el in btns:
        label = (el.get("text") or {}).get("text")
        assert el.get("action_id"), f"button {label!r} has no explicit action_id"


def test_view_button_wired_like_save():
    btns = {(el.get("text") or {}).get("text"): el for el in _card_buttons()}
    view, save = btns["View posting"], btns["Save"]
    assert view.get("action_id") == "job_view"
    view_payload = json.loads(view["value"])
    save_payload = json.loads(save["value"])
    assert view_payload == save_payload
    assert set(view_payload) == {"job_id", "title", "company", "location", "url"}
    # the client-side open must survive the wiring
    assert view.get("url") == VALID_JOB["url"]
