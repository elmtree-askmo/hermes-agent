"""Artemis S-0715-01 — the briefing job card resolves "today" on the USER's
local calendar, matching the basis the Artemis-side seeder dates the artifact on.

Under UTC dating the read was unsatisfiable for UTC+9 (a 9am-local briefing
fires at 00:00 UTC and looked for a date no prior firing could have written) and
left UTC+8 only the hour between UTC midnight and the briefing to produce it.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cron.scheduler import _render_job_cards_for_briefing, _user_local_now

pytestmark = pytest.mark.xdist_group("cron_scheduler")

USER = "U0LOC1"


def _write_artifact(tmp_path, user_id, datestr):
    d = tmp_path / "artemis" / user_id / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"job-match-{datestr}.json").write_text(json.dumps({
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [{"job_id": "j1", "title": "Senior PM", "company": "Acme",
                  "location": "SF", "url": "https://x/j1", "why": "fit",
                  "match_pct": 78}],
    }), encoding="utf-8")


def _set_tz(tmp_path, user_id, tz):
    d = tmp_path / "artemis" / user_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "slack_tz.txt").write_text(tz, encoding="utf-8")


# ---- _user_local_now --------------------------------------------------------


def test_user_local_now_uses_slack_tz(tmp_path):
    _set_tz(tmp_path, USER, "Asia/Chongqing")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        local = _user_local_now(USER)
    assert local.utcoffset() == timedelta(hours=8)


def test_user_local_now_falls_back_to_utc_without_tz(tmp_path):
    (tmp_path / "artemis" / USER).mkdir(parents=True, exist_ok=True)
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        local = _user_local_now(USER)
    assert local.utcoffset() == timedelta(0)


def test_user_local_now_falls_back_to_utc_on_bad_tz(tmp_path):
    _set_tz(tmp_path, USER, "Not/AZone")
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
        local = _user_local_now(USER)
    assert local.utcoffset() == timedelta(0)


# ---- the card read ----------------------------------------------------------


def test_card_reads_the_users_local_date(tmp_path):
    """UTC+8 at 16:30 UTC is already the next local day — the card must look for
    that local date, which is what the seeder wrote."""
    fixed_local = datetime(2026, 7, 16, 0, 30, tzinfo=timezone(timedelta(hours=8)))
    _set_tz(tmp_path, USER, "Asia/Chongqing")
    _write_artifact(tmp_path, USER, "20260716")          # local date
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler._user_local_now", return_value=fixed_local):
        out = _render_job_cards_for_briefing(USER, silence_tier=None)
    assert out, "local-dated artifact must be found"


def test_card_ignores_a_utc_dated_artifact_when_local_day_differs(tmp_path):
    """The previous basis' file (UTC date 07-15) is not this user's local today
    (07-16) — reading it would show yesterday's roles as today's."""
    fixed_local = datetime(2026, 7, 16, 0, 30, tzinfo=timezone(timedelta(hours=8)))
    _set_tz(tmp_path, USER, "Asia/Chongqing")
    _write_artifact(tmp_path, USER, "20260715")          # stale UTC-dated file
    with patch("cron.scheduler.get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler._user_local_now", return_value=fixed_local):
        out = _render_job_cards_for_briefing(USER, silence_tier=None)
    assert out is None
