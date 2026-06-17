"""S-0617-01: the cold-start block appends the onboarding sharpening
instruction branched on dispatch_type, and degrades safely when the
detector did not run (dispatch_type unbound/None)."""

from agent.turn_intent_detector import render_onboarding_sharpening_block


def _compose(cs_block: str, dispatch_type) -> str:
    """Mirror of the wiring in gateway/run.py cold-start block: append the
    sharpening block iff the helper returns one. Kept in sync with run.py."""
    extra = render_onboarding_sharpening_block(dispatch_type)
    return cs_block + extra if extra else cs_block


def test_multi_appends_preference_sharpening():
    composed = _compose("BASE_MIRROR_TONE", "multi")
    assert composed.startswith("BASE_MIRROR_TONE")
    assert "refine the scan's filters" in composed


def test_none_appends_blocking_sharpening():
    composed = _compose("BASE_MIRROR_TONE", "none")
    assert "direction not yet clear" in composed


def test_unbound_dispatch_type_appends_nothing():
    composed = _compose("BASE_MIRROR_TONE", None)
    assert composed == "BASE_MIRROR_TONE"
