"""Post-submission bridge injection — server-side detect + render.

Artemis P-0609-01 (fork side). After the user reports submitting an application,
the gateway injects a bridge state-reminder so Coach offers the next ready role
(one pivot sentence + A/B), instead of depending on Coach choosing to call
get_strategy.

S-0622-04 Phase 2: detection migrated from scanning strategy.json action_queue[]
for a Strategist-staged deliver-<company>-materials item to reading the
applications.json ledger for the next `materials_ready` application (drafted,
not yet submitted). The ledger is the deterministic single source — the original
P-0609-01 false-green was the Strategist *missing* the deliver-* emit; reading
the ledger removes that unreliable source. The just-submitted application is
already advanced to `submitted` (gateway submit-detect) by the time the bridge
runs, so the remaining materials_ready record is the genuine next role.
"""

import json

from agent.post_submission_bridge import (
    detect_next_queued_role,
    render_post_submission_bridge_block,
)


def _setup(tmp_path, user_id="U123", apps=None):
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "applications.json").write_text(
        json.dumps({"applications": apps or []}), encoding="utf-8")
    return user_dir


def _apprec(company, display, status, submitted_at=None, outcome=None):
    return {"company": company, "display_name": display, "status": status,
            "artifacts": [], "outcome": outcome, "submitted_at": submitted_at}


class TestDetect:
    def test_materials_ready_app_detected(self, tmp_path):
        ud = _setup(tmp_path, apps=[_apprec("oakwell", "Oakwell", "materials_ready")])
        r = detect_next_queued_role(ud)
        assert r is not None
        assert r["company_label"] == "Oakwell"

    def test_no_materials_ready_returns_none(self, tmp_path):
        ud = _setup(tmp_path, apps=[_apprec("oakwell", "Oakwell", "submitted")])
        assert detect_next_queued_role(ud) is None

    def test_empty_ledger_returns_none(self, tmp_path):
        ud = _setup(tmp_path, apps=[])
        assert detect_next_queued_role(ud) is None

    def test_submitted_app_not_detected(self, tmp_path):
        # An already-submitted role is not a *next* role to bridge to.
        ud = _setup(tmp_path, apps=[_apprec("oakwell", "Oakwell", "submitted")])
        assert detect_next_queued_role(ud) is None

    def test_outcome_set_app_not_detected(self, tmp_path):
        ud = _setup(tmp_path, apps=[_apprec("oakwell", "Oakwell", "materials_ready",
                    outcome={"result": "rejected", "at": "x", "note": "no"})])
        assert detect_next_queued_role(ud) is None

    def test_first_materials_ready_chosen_when_multiple(self, tmp_path):
        ud = _setup(tmp_path, apps=[
            _apprec("oakwell", "Oakwell", "materials_ready"),
            _apprec("brightline", "Brightline", "materials_ready"),
        ])
        r = detect_next_queued_role(ud)
        assert r["company_label"] == "Oakwell"

    def test_missing_ledger_returns_none(self, tmp_path):
        # Fail-safe: no applications.json must never raise inside the gateway turn.
        user_dir = tmp_path / "artemis" / "U404"
        user_dir.mkdir(parents=True, exist_ok=True)
        assert detect_next_queued_role(user_dir) is None

    def test_corrupt_ledger_returns_none(self, tmp_path):
        user_dir = tmp_path / "artemis" / "U500"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "applications.json").write_text("{not json", encoding="utf-8")
        assert detect_next_queued_role(user_dir) is None

    def test_company_label_from_display_name(self, tmp_path):
        ud = _setup(tmp_path, apps=[_apprec("brightline-health", "Brightline Health",
                                            "materials_ready")])
        r = detect_next_queued_role(ud)
        assert r["company_label"] == "Brightline Health"


class TestRender:
    def test_none_renders_empty(self):
        assert render_post_submission_bridge_block(None) == ""

    def test_block_names_the_role_and_ab_choice(self, tmp_path):
        ud = _setup(tmp_path, apps=[_apprec("oakwell", "Oakwell", "materials_ready")])
        r = detect_next_queued_role(ud)
        block = render_post_submission_bridge_block(r)
        assert block
        assert "Oakwell" in block
        low = block.lower()
        assert "look" in low and "hold" in low

    def test_block_forbids_celebration_tone(self, tmp_path):
        ud = _setup(tmp_path, apps=[_apprec("oakwell", "Oakwell", "materials_ready")])
        r = detect_next_queued_role(ud)
        block = render_post_submission_bridge_block(r)
        assert "🎉" not in block
        assert "no celebration" in block.lower()
        assert "no streak" in block.lower()
