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
    user_reported_completion,
    count_submitted_applications,
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
    # S-0622-04: the count source is now applications.json (status >= submitted),
    # not the archive event. Seed n_apps submitted records to match n_apps.
    if n_apps:
        _write_applications(
            user_dir,
            [_apprec(f"company{i}", "submitted") for i in range(1, n_apps + 1)],
        )
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


def _write_applications(user_dir, records):
    """Write applications.json (the S-0622-04 ledger) into an existing user_dir."""
    (user_dir / "applications.json").write_text(
        json.dumps({"applications": records,
                    "updated_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )


def _apprec(company, status):
    return {"company": company, "display_name": company, "status": status,
            "artifacts": [], "outcome": None}


class TestCountSubmittedApplications:
    """S-0622-04 Phase 1: the milestone count migrates from the archive
    application_submitted event to the applications.json ledger — count records
    with status >= submitted (submitted or interviewed)."""

    def test_counts_submitted_records(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        _write_applications(ud, [
            _apprec("loreal", "submitted"),
            _apprec("coca-cola", "submitted"),
        ])
        assert count_submitted_applications(ud) == 2

    def test_materials_ready_not_counted(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        _write_applications(ud, [
            _apprec("loreal", "submitted"),
            _apprec("acme", "materials_ready"),
            _apprec("widget", "identified"),
        ])
        assert count_submitted_applications(ud) == 1

    def test_interviewed_counts_as_submitted_or_higher(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        _write_applications(ud, [
            _apprec("loreal", "submitted"),
            _apprec("acme", "interviewed"),
        ])
        assert count_submitted_applications(ud) == 2

    def test_missing_applications_returns_zero(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        assert count_submitted_applications(ud) == 0

    def test_corrupt_applications_returns_zero(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        (ud / "applications.json").write_text("{not json", encoding="utf-8")
        assert count_submitted_applications(ud) == 0


class TestDetectFromApplications:
    """detect_milestone now reads applications.json, not the archive count."""

    def test_detect_counts_from_applications_ledger(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        (ud / "strategy.json").write_text(
            json.dumps({"milestones_affirmed": ["apps_2"]}), encoding="utf-8")
        _write_applications(ud, [
            _apprec("a", "submitted"), _apprec("b", "submitted"),
            _apprec("c", "submitted"),
        ])
        m = detect_milestone(ud)
        assert m is not None
        assert m["tier"] == "apps_3"
        assert m["count"] == 3

    def test_scene3_regression_archive_undercounts_ledger_correct(self, tmp_path):
        """SIM-JOURNEY scene 3: the user submitted 4 applications but only 2 rode
        the archive application_submitted event (Estée/Target cover letters used
        the save_cover_letter side path). The archive count was wrong at 2; the
        applications ledger is the correct 4. Migrated detect must see 4."""
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        # Archive has only 2 typed events (the old, wrong source).
        archive = [_app(1), _app(2)]
        (ud / "strategy.json").write_text(
            json.dumps({"archive": archive, "milestones_affirmed": []}),
            encoding="utf-8")
        # The ledger has the correct 4 submitted applications.
        _write_applications(ud, [
            _apprec("loreal", "submitted"), _apprec("coca-cola", "submitted"),
            _apprec("estee-lauder", "submitted"), _apprec("target", "submitted"),
        ])
        m = detect_milestone(ud)
        assert m is not None
        assert m["count"] == 4, "must count the ledger (4), not the archive (2)"


class TestUserReportedCompletion:
    """The affirm injection is gated on the user's turn reporting a completion,
    so a generic 'what's next' turn never burns an un-voiced tier."""

    @pytest.mark.parametrize("text", [
        "just submitted the warby parker one",
        "ok submitted it",
        "I applied to glossier",
        "sent it",
        "sent the topicals app",
        "just sent off the linnea one",
        "ok cool, hit submit on that",
        "got it in before the deadline",
        "SUBMITTED THE OAKWELL ROLE",  # case-insensitive
    ])
    def test_completion_reports_detected(self, text):
        assert user_reported_completion(text) is True

    @pytest.mark.parametrize("text", [
        "ok what's next",
        "cool. what should I focus on now",
        "show me the pipeline",
        "how's it looking",
        "let's do the widening scan",
        "thanks",
        "",
        None,
    ])
    def test_generic_turns_not_detected(self, text):
        assert user_reported_completion(text) is False
