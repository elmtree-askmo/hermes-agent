"""S-0617-01 non-blocking path: onboarding preference-sharpening injection.

Mirrors the milestone_detector test shape. The detector is onboarding-only and
ask-once: it fires when a preference-pending marker is present and the asked
marker is absent, injects ONE preference question, then marks asked so it never
re-fires. Fail-safe: missing/corrupt state returns None / no-op, never raises."""

from pathlib import Path

from agent.onboarding_preference_detector import (
    detect_onboarding_preference_pending,
    render_onboarding_preference_block,
    mark_onboarding_preference_asked,
    mark_onboarding_preference_pending,
)

PENDING = "onboarding_preference_pending.flag"
ASKED = "onboarding_preference_asked.flag"


def test_pending_marker_absent_returns_none(tmp_path):
    assert detect_onboarding_preference_pending(tmp_path) is None


def test_pending_set_asked_absent_returns_pending(tmp_path):
    (tmp_path / PENDING).write_text("1")
    result = detect_onboarding_preference_pending(tmp_path)
    assert result is not None


def test_asked_marker_suppresses_even_when_pending(tmp_path):
    (tmp_path / PENDING).write_text("1")
    (tmp_path / ASKED).write_text("1")
    assert detect_onboarding_preference_pending(tmp_path) is None


def test_render_block_contains_one_axis_preference_instruction():
    block = render_onboarding_preference_block({"kind": "preference"})
    assert "one" in block.lower()
    assert "prestige" in block.lower() or "exclude" in block.lower()
    # must NOT tell Coach to defer the team (team already dispatched)
    assert "do not say 'briefing the team'" not in block.lower()


def test_render_block_none_returns_empty_string():
    assert render_onboarding_preference_block(None) == ""


def test_mark_pending_then_asked_makes_detect_return_none(tmp_path):
    mark_onboarding_preference_pending(tmp_path)
    assert detect_onboarding_preference_pending(tmp_path) is not None
    mark_onboarding_preference_asked(tmp_path)
    assert detect_onboarding_preference_pending(tmp_path) is None


def test_mark_pending_idempotent(tmp_path):
    mark_onboarding_preference_pending(tmp_path)
    mark_onboarding_preference_pending(tmp_path)  # no raise
    assert (tmp_path / PENDING).exists()


def test_detect_on_nonexistent_dir_returns_none():
    assert detect_onboarding_preference_pending(Path("/nonexistent/xyz")) is None
