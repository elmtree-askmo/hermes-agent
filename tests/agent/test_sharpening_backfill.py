"""Unit tests for agent.sharpening_backfill gating (B-0624-04 Layer 2)."""

from __future__ import annotations

import pytest

from agent import sharpening_backfill as sb


class TestPreferencesAreEmpty:
    @pytest.mark.parametrize("profile", [
        None,
        {},
        {"goal": "x"},                       # preferences absent
        {"goal": "x", "preferences": None},  # preferences null (the bug state)
        {"preferences": {}},                 # empty dict
    ])
    def test_empty_cases(self, profile):
        assert sb.preferences_are_empty(profile) is True

    @pytest.mark.parametrize("profile", [
        {"preferences": {"location": "Boston"}},
        {"preferences": {"location": "Boston", "avoid": "reporting"}},
    ])
    def test_nonempty_cases(self, profile):
        assert sb.preferences_are_empty(profile) is False


class TestShouldBackfill:
    def test_fires_when_flag_set_and_prefs_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        uid = "U1"
        d = tmp_path / "artemis" / uid
        d.mkdir(parents=True)
        (d / "onboarding_pushed.flag").write_text("")
        assert sb.should_backfill(uid, {"goal": "x", "preferences": None}) is True

    def test_skips_when_no_flag(self, tmp_path, monkeypatch):
        """Cold-start / mid-onboarding (flag absent) -> never fire."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        uid = "U1"
        (tmp_path / "artemis" / uid).mkdir(parents=True)
        assert sb.should_backfill(uid, {"preferences": None}) is False

    def test_skips_when_prefs_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        uid = "U1"
        d = tmp_path / "artemis" / uid
        d.mkdir(parents=True)
        (d / "onboarding_pushed.flag").write_text("")
        assert sb.should_backfill(uid, {"preferences": {"location": "Boston"}}) is False

    def test_skips_when_no_user_id(self):
        assert sb.should_backfill("", {"preferences": None}) is False


class TestSpawnBackfill:
    def test_missing_helper_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no scripts/ helper here
        out = sb.spawn_backfill("U1", "sess1")
        assert out["ok"] is False
        assert "helper not found" in out["error"]

    def test_no_user_id(self):
        out = sb.spawn_backfill("", "sess1")
        assert out["ok"] is False
