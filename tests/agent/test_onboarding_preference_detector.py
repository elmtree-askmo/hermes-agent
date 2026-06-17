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
