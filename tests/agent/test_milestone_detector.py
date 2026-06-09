"""Milestone affirm injection — server-side detect + render + dedup mark.

Artemis S-0601-03 layer 1 (fork side). The gateway injects a milestone
state-reminder block into Coach's system prompt each turn when the user has
crossed an un-affirmed application-count tier, so the affirm does not depend on
Coach choosing to call get_strategy. Counts are deterministic over
strategy.json archive[] application_submitted events (S-0601-02).
"""

import json
from datetime import datetime, timezone

import pytest

from agent.milestone_detector import (
    detect_milestone,
    render_milestone_block,
    mark_milestone_affirmed,
)


def _app(i):
    return {
        "id": f"deliver-company{i}-materials",
        "status": "done",
        "event_type": "application_submitted",
        "artifact_kind": "cover-letter",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _setup(tmp_path, user_id="U123", n_apps=0, affirmed=None, extra=None):
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    archive = [_app(i) for i in range(1, n_apps + 1)]
    if extra:
        archive.extend(extra)
    strategy = {
        "user_id": user_id,
        "archive": archive,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if affirmed is not None:
        strategy["milestones_affirmed"] = affirmed
    (user_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")
    return user_dir


class TestDetect:
    def test_crossing_apps_3_with_apps_2_affirmed(self, tmp_path):
        ud = _setup(tmp_path, n_apps=3, affirmed=["apps_2"])
        m = detect_milestone(ud)
        assert m is not None
        assert m["tier"] == "apps_3"
        assert m["count"] == 3

    def test_already_affirmed_returns_none(self, tmp_path):
        ud = _setup(tmp_path, n_apps=3, affirmed=["apps_2", "apps_3"])
        assert detect_milestone(ud) is None

    def test_lower_tier_not_refired_when_higher_already_affirmed(self, tmp_path):
        # 3 apps, only apps_3 marked (apps_2 was implicitly crossed at the same
        # time). apps_2 must NOT re-fire — a higher affirmed tier covers every
        # lower tier. Regression for the dev 2026-06-09 double-inject.
        ud = _setup(tmp_path, n_apps=3, affirmed=["apps_3"])
        assert detect_milestone(ud) is None

    def test_fourth_app_no_tier_returns_none(self, tmp_path):
        ud = _setup(tmp_path, n_apps=4, affirmed=["apps_2", "apps_3"])
        assert detect_milestone(ud) is None

    def test_below_first_threshold_returns_none(self, tmp_path):
        ud = _setup(tmp_path, n_apps=1)
        assert detect_milestone(ud) is None

    def test_highest_unaffirmed_tier_chosen(self, tmp_path):
        ud = _setup(tmp_path, n_apps=5, affirmed=[])
        m = detect_milestone(ud)
        assert m["tier"] == "apps_5"
        assert m["count"] == 5

    def test_non_application_items_not_counted(self, tmp_path):
        extra = [{"id": "scan-x", "status": "done"},
                 {"id": "fu", "event_type": None, "artifact_kind": "inbox"}]
        ud = _setup(tmp_path, n_apps=2, affirmed=[], extra=extra)
        m = detect_milestone(ud)
        assert m["tier"] == "apps_2"
        assert m["count"] == 2

    def test_missing_strategy_returns_none(self, tmp_path):
        ud = tmp_path / "artemis" / "Unobody"
        ud.mkdir(parents=True, exist_ok=True)
        assert detect_milestone(ud) is None

    def test_corrupt_strategy_returns_none(self, tmp_path):
        ud = tmp_path / "artemis" / "Ubad"
        ud.mkdir(parents=True, exist_ok=True)
        (ud / "strategy.json").write_text("{not json", encoding="utf-8")
        assert detect_milestone(ud) is None


class TestRender:
    def test_block_names_count_and_credits_user(self):
        block = render_milestone_block({"tier": "apps_3", "count": 3,
                                        "label": "3 applications submitted"})
        assert "3" in block
        assert "milestone" in block.lower()
        assert block.strip() != ""

    def test_render_none_returns_empty(self):
        assert render_milestone_block(None) == ""


class TestMark:
    def test_mark_appends_tier_and_persists(self, tmp_path):
        ud = _setup(tmp_path, n_apps=3, affirmed=["apps_2"])
        mark_milestone_affirmed(ud, "apps_3")
        saved = json.loads((ud / "strategy.json").read_text())
        assert "apps_3" in saved["milestones_affirmed"]
        assert "apps_2" in saved["milestones_affirmed"]

    def test_mark_does_not_truncate_archive(self, tmp_path):
        ud = _setup(tmp_path, n_apps=10, affirmed=[])
        mark_milestone_affirmed(ud, "apps_10")
        saved = json.loads((ud / "strategy.json").read_text())
        apps = [a for a in saved["archive"]
                if a.get("event_type") == "application_submitted"]
        assert len(apps) == 10

    def test_mark_idempotent(self, tmp_path):
        ud = _setup(tmp_path, n_apps=3, affirmed=["apps_3"])
        mark_milestone_affirmed(ud, "apps_3")
        saved = json.loads((ud / "strategy.json").read_text())
        assert saved["milestones_affirmed"].count("apps_3") == 1

    def test_detect_then_mark_then_detect_is_none(self, tmp_path):
        ud = _setup(tmp_path, n_apps=3, affirmed=["apps_2"])
        m = detect_milestone(ud)
        assert m["tier"] == "apps_3"
        mark_milestone_affirmed(ud, "apps_3")
        assert detect_milestone(ud) is None
