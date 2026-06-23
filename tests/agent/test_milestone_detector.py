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


def _apprec(company, status, submitted_at=None):
    return {"company": company, "display_name": company, "status": status,
            "artifacts": [], "outcome": None, "submitted_at": submitted_at}


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


# ---- S-0622-04 Phase 2: interview / outcome detection + advance ----
#
# Company mapping strategy: the detector does NOT do open entity extraction. It
# canonical-folds each KNOWN active application's display_name and the user text
# to the same form (lowercase, ascii-fold, hyphenate) and looks for a hyphen-
# segment-bounded match. So variants fold and match (L'Oréal / loreal / Coca-Cola
# / coca cola), and a company not in the ledger can't be invented. When the text
# names no known application, it falls back to the spec's heuristic: the most-
# recent active application (the one genuinely-probabilistic edge, spec § Known
# limitation).

from agent.milestone_detector import (  # noqa: E402
    detect_interview,
    detect_outcome,
    detect_submit,
    advance_interview,
    advance_outcome,
    advance_submitted,
)


def _seed(tmp_path, records):
    ud = tmp_path / "artemis" / "U1"
    ud.mkdir(parents=True)
    _write_applications(ud, records)
    return ud


class TestDetectInterview:
    @pytest.mark.parametrize("display,text,key", [
        ("Glossier", "just got out of the glossier screen", "glossier"),
        ("Topicals", "had my phone screen with Topicals today", "topicals"),
        ("Warby Parker", "just interviewed at Warby Parker", "warby-parker"),
        ("L'Oréal", "first round with loreal went ok", "loreal"),
        ("Coca-Cola", "had the coca cola interview", "coca-cola"),
    ])
    def test_interview_phrasing_maps_to_known_record(self, tmp_path, display, text, key):
        ud = _seed(tmp_path, [_apprec(key, "submitted")])
        # seed display_name so folding can match the user's surface form
        recs = json.loads((ud / "applications.json").read_text())["applications"]
        recs[0]["display_name"] = display
        _write_applications(ud, recs)
        res = detect_interview(text, ud)
        assert res is not None
        assert res["company"] == key

    @pytest.mark.parametrize("text", [
        "what's next", "send the warby parker materials", "thanks", "", None,
        "i submitted the glossier one",  # submit phrasing, not interview
    ])
    def test_non_interview_turns_not_detected(self, tmp_path, text):
        ud = _seed(tmp_path, [_apprec("glossier", "submitted")])
        assert detect_interview(text, ud) is None

    def test_unknown_company_with_interview_phrasing_falls_back_to_recent(self, tmp_path):
        """Interview phrasing but no named company -> most-recent active record."""
        g = _apprec("glossier", "submitted"); g["submitted_at"] = "2026-06-18"
        t = _apprec("topicals", "submitted"); t["submitted_at"] = "2026-06-20"
        ud = _seed(tmp_path, [g, t])
        res = detect_interview("just got out of the screen", ud)
        assert res is not None
        assert res["company"] == "topicals"  # most recent submitted_at


class TestDetectOutcome:
    @pytest.mark.parametrize("text,key,result", [
        ("glossier said no", "glossier", "rejected"),
        ("topicals passed on me", "topicals", "rejected"),
        ("didn't get the glossier role", "glossier", "rejected"),
        ("glossier — moving forward with other candidates", "glossier", "rejected"),
    ])
    def test_rejection_maps_to_known_record(self, tmp_path, text, key, result):
        ud = _seed(tmp_path, [_apprec("glossier", "interviewed"),
                              _apprec("topicals", "submitted")])
        res = detect_outcome(text, ud)
        assert res is not None
        assert res["company"] == key
        assert res["result"] == result

    @pytest.mark.parametrize("text", [
        "what's next", "just got out of the glossier screen", "thanks", "", None,
    ])
    def test_non_outcome_turns_not_detected(self, tmp_path, text):
        ud = _seed(tmp_path, [_apprec("glossier", "submitted")])
        assert detect_outcome(text, ud) is None


class TestAdvanceInterview:
    def test_advances_submitted_to_interviewed(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("glossier", "submitted")])
        assert advance_interview(ud, "glossier") is True
        recs = json.loads((ud / "applications.json").read_text())["applications"]
        assert recs[0]["status"] == "interviewed"

    def test_no_matching_record_is_noop(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("glossier", "submitted")])
        assert advance_interview(ud, "nonexistent") is False
        recs = json.loads((ud / "applications.json").read_text())["applications"]
        assert recs[0]["status"] == "submitted"

    def test_missing_ledger_is_noop(self, tmp_path):
        ud = tmp_path / "artemis" / "U1"
        ud.mkdir(parents=True)
        assert advance_interview(ud, "glossier") is False


class TestAdvanceOutcome:
    def test_sets_outcome_keeps_status(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("glossier", "interviewed")])
        assert advance_outcome(ud, "glossier", "rejected",
                               "moving forward with other candidates") is True
        rec = json.loads((ud / "applications.json").read_text())["applications"][0]
        assert rec["status"] == "interviewed"  # terminal lives in outcome
        assert rec["outcome"]["result"] == "rejected"
        assert rec["outcome"]["note"] == "moving forward with other candidates"

    def test_company_canonicalization_matches(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("warby-parker", "submitted")])
        assert advance_outcome(ud, "Warby Parker", "rejected", None) is True
        rec = json.loads((ud / "applications.json").read_text())["applications"][0]
        assert rec["outcome"]["result"] == "rejected"


class TestDetectSubmit:
    """User-report submit class (spec line 134, moved into Phase 2). Reuses the
    submit word-list (user_reported_completion) for the signal and maps the
    company against the known materials_ready records."""

    @pytest.mark.parametrize("text,key", [
        ("just sent the glossier application", "glossier"),
        ("submitted warby parker", "warby-parker"),
    ])
    def test_submit_phrasing_maps_to_known_materials_ready(self, tmp_path, text, key):
        ud = _seed(tmp_path, [_apprec(key, "materials_ready")])
        recs = json.loads((ud / "applications.json").read_text())["applications"]
        recs[0]["display_name"] = key.replace("-", " ").title()
        _write_applications(ud, recs)
        res = detect_submit(text, ud)
        assert res is not None and res["company"] == key

    @pytest.mark.parametrize("text", ["what's next", "thanks", "", None])
    def test_non_submit_turns_not_detected(self, tmp_path, text):
        ud = _seed(tmp_path, [_apprec("glossier", "materials_ready")])
        assert detect_submit(text, ud) is None

    def test_named_submit_maps_to_resume_only_identified(self, tmp_path):
        """A user can submit an application that has only a resume (no cover
        letter), so it sits at `identified` rather than `materials_ready`. A
        NAMED submit report must still map to it — the resume-ready record is a
        real, submit-trackable application."""
        rec = _apprec("acme-analytics", "identified")
        rec["display_name"] = "Acme Analytics"
        rec["artifacts"] = [{"kind": "resume", "name": "acme-analytics-ds"}]
        ud = _seed(tmp_path, [rec])
        res = detect_submit("just submitted the Acme Analytics application", ud)
        assert res is not None and res["company"] == "acme-analytics"

    def test_named_submit_skips_identified_with_no_resume(self, tmp_path):
        """An `identified` record with no resume artifact is not yet a prepared
        application (nothing tailored). A named submit must NOT map to it."""
        rec = _apprec("placeholder-co", "identified")
        rec["display_name"] = "Placeholder Co"
        rec["artifacts"] = []
        ud = _seed(tmp_path, [rec])
        assert detect_submit("just submitted the Placeholder Co application", ud) is None

    def test_unnamed_submit_does_not_fallback_to_identified(self, tmp_path):
        """The no-named-company fallback must NOT reach into `identified` records
        — only materials_ready/active ones. A resume-only `identified` record the
        user merely had drafted should not be silently marked submitted by a
        generic 'submitted it' with no company named."""
        rec = _apprec("acme-analytics", "identified")
        rec["display_name"] = "Acme Analytics"
        rec["artifacts"] = [{"kind": "resume", "name": "acme-analytics-ds"}]
        ud = _seed(tmp_path, [rec])
        assert detect_submit("ok submitted it", ud) is None


class TestAdvanceSubmitted:
    def test_advances_materials_ready_to_submitted(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("glossier", "materials_ready")])
        assert advance_submitted(ud, "glossier") is True
        rec = json.loads((ud / "applications.json").read_text())["applications"][0]
        assert rec["status"] == "submitted"
        assert rec["submitted_at"]  # stamped

    def test_no_matching_record_is_noop(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("glossier", "materials_ready")])
        assert advance_submitted(ud, "nonexistent") is False

    def test_idempotent_on_already_submitted(self, tmp_path):
        ud = _seed(tmp_path, [_apprec("glossier", "submitted", submitted_at="2026-06-18")])
        assert advance_submitted(ud, "glossier") is True
        rec = json.loads((ud / "applications.json").read_text())["applications"][0]
        assert rec["status"] == "submitted"
        assert rec["submitted_at"] == "2026-06-18"  # not overwritten


@pytest.mark.parametrize("raw,expected", [
    ("L'Oréal", "loreal"), ("loreal", "loreal"), ("L'Oreal", "loreal"),
    ("Coca-Cola", "coca-cola"), ("coca-cola", "coca-cola"),
    ("Estée Lauder", "estee-lauder"), ("Warby Parker", "warby-parker"),
    ("", ""), (None, ""),
])
def test_canonical_company_mirrors_artemis(raw, expected):
    """The fork's narrow _canonical_company MUST fold identically to Artemis's
    (the two are duplicated across the repo boundary — this locks them in sync so
    a record written by one side is found by the other; no fracture)."""
    from agent.milestone_detector import _canonical_company
    assert _canonical_company(raw) == expected
