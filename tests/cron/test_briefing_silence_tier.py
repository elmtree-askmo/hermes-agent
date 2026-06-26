"""Briefing silence-tier injection (Artemis S-0525-02 Domain 6).

`_build_job_prompt` appends a SILENCE_AWARENESS directive to artemis-briefing
jobs based on how long the user has been silent, computed by the Artemis helper
`scripts/compute-silence-tier.py` (run via subprocess, mirroring the gateway's
compute-pending-announcements.py). Fail-open: any error → no directive.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import (
    _briefing_silence_directive,
    _build_job_prompt,
    _run_briefing_render,
)

pytestmark = pytest.mark.xdist_group("cron_scheduler")


def _briefing_job(user_id="U123"):
    return {
        "skills": ["artemis-briefing"],
        "prompt": "Run artemis-briefing",
        "origin": {"user_id": user_id},
        "repeat": {"completed": 5},  # past the FOOTER_REQUIRED window
    }


def _patch_script(tmp_path, payload):
    """Make the helper script 'exist' and stub its subprocess output to `payload`."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "compute-silence-tier.py").write_text("# stub")
    proc = MagicMock(returncode=0, stdout=json.dumps(payload))
    return (
        patch("cron.scheduler.get_hermes_home", return_value=tmp_path),
        patch("cron.scheduler.subprocess.run", return_value=proc),
    )


class TestSilenceDirective:
    @pytest.mark.parametrize("tier,speak,needle", [
        ("day1", True, "low-key"),
        ("day5", True, "pause"),
        ("day8", True, "lowest-bar"),
    ])
    def test_speak_tier_directives(self, tmp_path, tier, speak, needle):
        gh, sr = _patch_script(tmp_path, {"tier": tier, "speak": speak, "reason": "x"})
        with gh, sr:
            out = _briefing_silence_directive(_briefing_job())
        assert "SILENCE_AWARENESS" in out
        assert needle in out

    def test_silent_gap_emits_silent_directive(self, tmp_path):
        gh, sr = _patch_script(tmp_path, {"tier": "day1", "speak": False, "reason": "x"})
        with gh, sr:
            out = _briefing_silence_directive(_briefing_job())
        assert "[SILENT]" in out

    def test_engaged_yields_no_directive(self, tmp_path):
        gh, sr = _patch_script(tmp_path, {"tier": "engaged", "speak": True, "reason": "x"})
        with gh, sr:
            assert _briefing_silence_directive(_briefing_job()) == ""

    def test_no_user_id_yields_no_directive(self, tmp_path):
        gh, sr = _patch_script(tmp_path, {"tier": "day5", "speak": True, "reason": "x"})
        with gh, sr:
            assert _briefing_silence_directive({"skills": ["artemis-briefing"]}) == ""

    def test_missing_script_fails_open(self, tmp_path):
        # script not created → does not exist
        with patch("cron.scheduler.get_hermes_home", return_value=tmp_path):
            assert _briefing_silence_directive(_briefing_job()) == ""

    def test_subprocess_error_fails_open(self, tmp_path):
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "compute-silence-tier.py").write_text("# stub")
        with patch("cron.scheduler.get_hermes_home", return_value=tmp_path), \
             patch("cron.scheduler.subprocess.run", side_effect=OSError("boom")):
            assert _briefing_silence_directive(_briefing_job()) == ""

    def test_bad_json_fails_open(self, tmp_path):
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "compute-silence-tier.py").write_text("# stub")
        proc = MagicMock(returncode=0, stdout="not json")
        with patch("cron.scheduler.get_hermes_home", return_value=tmp_path), \
             patch("cron.scheduler.subprocess.run", return_value=proc):
            assert _briefing_silence_directive(_briefing_job()) == ""


class TestSilenceTierRender:
    """S-0626-02 Plan B: the deterministic silence tier is injected onto the
    package inside _run_briefing_render. The package is now internal, so we
    assert the observable behavior — the silence path renders a body without
    crashing — rather than reaching into the (now private) package."""

    _PKG = {
        "coaches_take": "Day 1 of the wait — patience is the strategy.",
        "opener": None,
        "response_window_checkin": None,
    }

    def test_silence_tier_renders_body(self):
        with patch("cron.scheduler._parse_step0_output", return_value=dict(self._PKG)):
            out = _run_briefing_render("<json>", "j", silence_tier="day5")
        assert out is not None
        assert "patience is the strategy" in out

    def test_no_silence_tier_renders_body(self):
        with patch("cron.scheduler._parse_step0_output", return_value=dict(self._PKG)):
            out = _run_briefing_render("<json>", "j")
        assert out is not None
        assert "patience is the strategy" in out


class TestBuildJobPromptInjection:
    def test_directive_appended_for_briefing(self):
        with patch("cron.scheduler._briefing_silence_directive",
                   return_value="SILENCE_AWARENESS: test directive"):
            out = _build_job_prompt(_briefing_job())
        assert "SILENCE_AWARENESS: test directive" in out

    def test_no_directive_for_non_briefing_job(self):
        # Non-briefing jobs never call the silence path → no SILENCE_AWARENESS.
        out = _build_job_prompt({"prompt": "some other cron job"})
        assert "SILENCE_AWARENESS" not in out

    def test_silent_directive_present_for_new_user(self, tmp_path):
        """S-0626-02: the onboarding pause-reminder footer moved server-side
        (step-0 emits JSON, so a footer line in the prompt would break
        json.loads) — it's no longer a prompt directive. The silence-tier
        [SILENT] directive is still appended to the built prompt."""
        gh, sr = _patch_script(tmp_path, {"tier": "day8", "speak": False, "reason": "x"})
        job = _briefing_job()
        job["repeat"] = {"completed": 0}  # would have been inside the old footer window
        with gh, sr, patch("tools.skills_tool.skill_view",
                           return_value=json.dumps({"success": True, "content": "briefing skill"})):
            out = _build_job_prompt(job)
        # The footer is no longer injected into the prompt.
        assert "FOOTER_REQUIRED" not in out
        # The silent-gap silence directive still rides the trailing-directives path.
        assert "SILENCE_AWARENESS" in out
        assert "do not send a briefing today" in out
