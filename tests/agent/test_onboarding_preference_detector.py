"""S-0617-01: onboarding direction-flag tracking.

The module tracks whether a single/multi dispatch occurred this onboarding
(the direction flag), read later to gate the proactive sharpening invite.
Fail-safe: missing/corrupt state returns False / no-op, never raises. The
reactive preference-injection this module once held was removed in the v3
rewrite (see docs/specs/sharpening-questions.md § Amendment v3)."""

from pathlib import Path

from agent.onboarding_preference_detector import (
    mark_onboarding_direction_present,
    has_onboarding_direction_present,
)

DIRECTION = "onboarding_direction_present.flag"


def test_direction_absent_by_default(tmp_path):
    assert has_onboarding_direction_present(tmp_path) is False


def test_mark_direction_then_present(tmp_path):
    mark_onboarding_direction_present(tmp_path)
    assert (tmp_path / DIRECTION).exists()
    assert has_onboarding_direction_present(tmp_path) is True


def test_mark_direction_idempotent(tmp_path):
    mark_onboarding_direction_present(tmp_path)
    mark_onboarding_direction_present(tmp_path)  # no raise
    assert has_onboarding_direction_present(tmp_path) is True


def test_has_direction_on_nonexistent_dir_is_false():
    assert has_onboarding_direction_present(Path("/nonexistent/xyz")) is False


# B-0624-03: cold-start onboarding-block direction gate.
# The gate must treat a goal stated this turn (direction_present=True) as
# "direction present" even when no sub-agent dispatch fired (dispatch_type
# "none") and the profile has no goal yet — otherwise the contradictory
# "user hasn't told you their goal / don't brief the team" block fires and
# Coach leaks its planning monologue.
from agent.onboarding_preference_detector import cold_start_no_direction


def test_goal_as_question_routes_to_handoff():
    # Maya's opener: goal stated, but framed as a question → dispatch none,
    # profile empty. Must NOT be "no direction".
    assert cold_start_no_direction("none", has_goal=False, direction_present=True) is False


def test_bare_greeting_is_no_direction():
    # "hi" — no goal, no dispatch, no direction → invite-the-goal block.
    assert cold_start_no_direction("none", has_goal=False, direction_present=False) is True


def test_dispatch_is_direction():
    assert cold_start_no_direction("single", has_goal=False, direction_present=False) is False
    assert cold_start_no_direction("multi", has_goal=False, direction_present=False) is False


def test_existing_profile_goal_is_direction():
    assert cold_start_no_direction("none", has_goal=True, direction_present=False) is False
