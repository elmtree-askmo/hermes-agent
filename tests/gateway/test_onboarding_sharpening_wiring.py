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


def _should_mark_pending(spawn_ok: bool, dispatch_type) -> bool:
    """Mirror of the gateway gate: mark preference-pending only when the team
    actually dispatched (non-blocking path) and the spawn succeeded."""
    return bool(spawn_ok) and dispatch_type in ("single", "multi")


def test_mark_pending_only_on_nonblocking_successful_spawn():
    assert _should_mark_pending(True, "multi") is True
    assert _should_mark_pending(True, "single") is True
    assert _should_mark_pending(True, "none") is False      # blocking path: no preference marker
    assert _should_mark_pending(False, "multi") is False    # dispatch failed: no team, no refinement
    assert _should_mark_pending(True, None) is False         # detector off/unbound


def test_pending_then_asked_lifecycle(tmp_path):
    mark_onboarding_preference_pending(tmp_path)
    assert detect_onboarding_preference_pending(tmp_path) is not None
    mark_onboarding_preference_asked(tmp_path)
    assert detect_onboarding_preference_pending(tmp_path) is None
