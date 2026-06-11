"""Post-submission bridge injection — server-side detect + render.

Artemis P-0609-01 (fork side). After the user reports submitting an application,
the gateway injects a bridge state-reminder so Coach offers the next queued role
(one pivot sentence + A/B), instead of depending on Coach choosing to call
get_strategy. Detection reads strategy.json action_queue[] for a pending
deliver-<company>-materials role-decision item. Gated on the same
user_reported_completion signal as the milestone affirm (agent/milestone_detector).
"""

import json

from agent.post_submission_bridge import (
    detect_next_queued_role,
    render_post_submission_bridge_block,
)


def _setup(tmp_path, user_id="U123", queue=None):
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    strategy = {"user_id": user_id, "action_queue": queue or []}
    (user_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")
    return user_dir


def _role(id_, status="pending", action="Surface the role for review"):
    return {"id": id_, "status": status, "action": action, "surface": "coach_surface"}


class TestDetect:
    def test_pending_deliver_item_detected(self, tmp_path):
        ud = _setup(tmp_path, queue=[_role("deliver-oakwell-materials")])
        r = detect_next_queued_role(ud)
        assert r is not None
        assert r["id"] == "deliver-oakwell-materials"

    def test_no_deliver_item_returns_none(self, tmp_path):
        ud = _setup(tmp_path, queue=[_role("tailor-resume-oakwell")])
        assert detect_next_queued_role(ud) is None

    def test_empty_queue_returns_none(self, tmp_path):
        ud = _setup(tmp_path, queue=[])
        assert detect_next_queued_role(ud) is None

    def test_done_deliver_item_not_detected(self, tmp_path):
        # An already-delivered role is not a *next* role to bridge to.
        ud = _setup(tmp_path, queue=[_role("deliver-oakwell-materials", status="done")])
        assert detect_next_queued_role(ud) is None

    def test_blocked_deliver_item_not_detected(self, tmp_path):
        # Only an actionable (pending) role is offerable; blocked is parked.
        ud = _setup(tmp_path, queue=[_role("deliver-oakwell-materials", status="blocked")])
        assert detect_next_queued_role(ud) is None

    def test_first_pending_deliver_chosen_when_multiple(self, tmp_path):
        # Stage rule queues one at a time, but be deterministic if two exist:
        # the earliest-ordered pending deliver item is the immediate next role.
        ud = _setup(tmp_path, queue=[
            _role("deliver-oakwell-materials"),
            _role("deliver-brightline-materials"),
        ])
        r = detect_next_queued_role(ud)
        assert r["id"] == "deliver-oakwell-materials"

    def test_missing_strategy_returns_none(self, tmp_path):
        # Fail-safe: no strategy.json must never raise inside the gateway turn.
        user_dir = tmp_path / "artemis" / "U404"
        user_dir.mkdir(parents=True, exist_ok=True)
        assert detect_next_queued_role(user_dir) is None

    def test_corrupt_strategy_returns_none(self, tmp_path):
        user_dir = tmp_path / "artemis" / "U500"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "strategy.json").write_text("{not json", encoding="utf-8")
        assert detect_next_queued_role(user_dir) is None

    def test_company_label_derived_from_id(self, tmp_path):
        ud = _setup(tmp_path, queue=[_role("deliver-brightline-health-materials")])
        r = detect_next_queued_role(ud)
        assert r["company_label"] == "Brightline Health"


class TestRender:
    def test_none_renders_empty(self):
        assert render_post_submission_bridge_block(None) == ""

    def test_block_names_the_role_and_ab_choice(self, tmp_path):
        ud = _setup(tmp_path, queue=[_role("deliver-oakwell-materials")])
        r = detect_next_queued_role(ud)
        block = render_post_submission_bridge_block(r)
        assert block
        assert "Oakwell" in block
        # One pivot sentence + A/B — look now / hold it. No celebration tone.
        low = block.lower()
        assert "look" in low and "hold" in low

    def test_block_forbids_celebration_tone(self, tmp_path):
        ud = _setup(tmp_path, queue=[_role("deliver-oakwell-materials")])
        r = detect_next_queued_role(ud)
        block = render_post_submission_bridge_block(r)
        # The bridge is Rule 1 surface-and-invite, not a milestone celebration.
        # No emoji, and it must instruct against celebration/streak tone.
        assert "🎉" not in block
        assert "no celebration" in block.lower()
        assert "no streak" in block.lower()
