"""S-0617-01: the cold-start block appends the onboarding sharpening
instruction branched on dispatch_type, and degrades safely when the
detector did not run (dispatch_type unbound/None)."""

from agent.turn_intent_detector import render_onboarding_sharpening_block


def _compose(cs_block: str, dispatch_type) -> str:
    """Mirror of the wiring in gateway/run.py cold-start block: append the
    sharpening block iff the helper returns one. Kept in sync with run.py."""
    extra = render_onboarding_sharpening_block(dispatch_type)
    return cs_block + extra if extra else cs_block


def test_multi_appends_nothing():
    # S-0617-01 v3: the non-blocking (single/multi) preference path moved out of
    # the cold-start block into onboarding_preference_detector, so the cold-start
    # helper now appends nothing for multi/single. The preference question is
    # injected next turn by the milestone-affirm-style site instead.
    composed = _compose("BASE_MIRROR_TONE", "multi")
    assert composed == "BASE_MIRROR_TONE"


def test_single_appends_nothing():
    composed = _compose("BASE_MIRROR_TONE", "single")
    assert composed == "BASE_MIRROR_TONE"


def test_none_appends_blocking_sharpening():
    composed = _compose("BASE_MIRROR_TONE", "none")
    assert "direction not yet clear" in composed


def test_unbound_dispatch_type_appends_nothing():
    composed = _compose("BASE_MIRROR_TONE", None)
    assert composed == "BASE_MIRROR_TONE"


from agent.onboarding_preference_detector import (
    detect_onboarding_preference_pending,
    mark_onboarding_preference_pending,
    mark_onboarding_preference_asked,
)


def _should_mark_pending(spawn_ok: bool, direction_present: bool) -> bool:
    """New gate: drop preference-pending when onboarding-complete spawned the
    team AND this onboarding arrived via the non-blocking path (direction was
    marked at the goal turn). No longer reads this-turn dispatch_type."""
    return bool(spawn_ok) and bool(direction_present)


def test_mark_pending_only_on_nonblocking_successful_spawn():
    assert _should_mark_pending(True, True) is True       # non-blocking + spawned
    assert _should_mark_pending(True, False) is False      # blocking path (no direction flag)
    assert _should_mark_pending(False, True) is False      # spawn failed


def test_pending_then_asked_lifecycle(tmp_path):
    mark_onboarding_preference_pending(tmp_path)
    assert detect_onboarding_preference_pending(tmp_path) is not None
    mark_onboarding_preference_asked(tmp_path)
    assert detect_onboarding_preference_pending(tmp_path) is None


from agent.onboarding_preference_detector import render_onboarding_preference_block


def _compose_preference(context_prompt: str, pending) -> str:
    """Mirror of the gateway preference injection: append the block iff pending."""
    block = render_onboarding_preference_block(pending)
    return context_prompt + block if block else context_prompt


def test_preference_pending_appends_block():
    composed = _compose_preference("BASE", {"kind": "preference"})
    assert composed.startswith("BASE")
    assert "refine the scan's filters" in composed


def test_no_pending_appends_nothing():
    assert _compose_preference("BASE", None) == "BASE"
