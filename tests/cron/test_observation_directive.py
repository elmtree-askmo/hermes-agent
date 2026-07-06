"""Tests for _observation_directive (Artemis S-0601-05 / S-0702-01 / S-0707-01).

The scheduler shells out to the Artemis helper `compute-observation-surface.py`
and frames the selected observation as a briefing beat: name-only for plain
observations, a scan-steering OFFER for actionable ones, and — S-0707-01 M4 —
an offer that carries the deterministic `direction_history_note` when the
selected direction was applied-then-displaced before (re-offer with memory,
never "fresh").

Tests monkeypatch the subprocess boundary — no real Artemis install.
"""
import json
from types import SimpleNamespace

import pytest

import cron.scheduler as scheduler


JOB = {"origin": {"user_id": "U_TEST"}}


@pytest.fixture
def helper_script(tmp_path, monkeypatch):
    """Point get_hermes_home at tmp_path with the helper script present."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "compute-observation-surface.py").write_text("# stub\n")
    monkeypatch.setattr(scheduler, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _mock_surface(monkeypatch, observation):
    payload = json.dumps({"observation": observation, "reason": "selected"})

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)


def test_actionable_observation_renders_offer(helper_script, monkeypatch):
    _mock_surface(monkeypatch, {
        "id": "obs-x",
        "text": "keeps coming back to mission-driven work",
        "direction": {"actionable": True, "scan_dimension": "domain",
                      "value": "healthcare-mission-driven"},
    })
    directive = scheduler._observation_directive(JOB)
    assert "scan-steering OFFER" in directive
    assert "healthcare-mission-driven" in directive
    assert "RE-OFFER HISTORY" not in directive


def test_history_note_spliced_into_offer(helper_script, monkeypatch):
    """S-0707-01 M4: a re-offered direction carries its deterministic history —
    the directive must instruct the voice to acknowledge the earlier tilt and
    the switch instead of presenting the pattern as new."""
    note = (
        "The scan was tilted toward 'healthcare-mission-driven' on 2026-07-04; "
        "it moved to 'end-to-end-model-ownership' when the user confirmed that "
        "direction on 2026-07-06."
    )
    _mock_surface(monkeypatch, {
        "id": "obs-x",
        "text": "keeps coming back to mission-driven work",
        "direction": {"actionable": True, "scan_dimension": "domain",
                      "value": "healthcare-mission-driven"},
        "direction_history_note": note,
    })
    directive = scheduler._observation_directive(JOB)
    assert "scan-steering OFFER" in directive
    assert note in directive
    assert "RE-OFFER HISTORY" in directive
    assert "as new" in directive  # do-not-present-as-new instruction


def test_history_note_ignored_on_non_actionable(helper_script, monkeypatch):
    """A stray note on a non-actionable observation must not turn the name-only
    beat into an offer."""
    _mock_surface(monkeypatch, {
        "id": "obs-y",
        "text": "takes breaks when overwhelmed, then re-engages",
        "direction_history_note": "should never surface",
    })
    directive = scheduler._observation_directive(JOB)
    assert "OFFER" not in directive
    assert "should never surface" not in directive
