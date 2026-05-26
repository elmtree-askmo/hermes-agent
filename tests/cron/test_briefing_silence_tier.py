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
    _run_two_step_briefing,
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


class TestTwoStepSilenceTier:
    """Phase 2: the tier flows into the decide package deterministically (code
    sets it), so the write call can branch copy — not inferred from raw output."""

    _PKG = {
        "briefing_type": "quiet_day",
        "follow_ups": [],
        "coaches_take": "x",
        "tone_signal": "low_pressure",
        "team_work": {},
    }

    def test_silence_tier_injected_into_write_package(self):
        captured = {}

        def fake_write(pkg, job_id="?"):
            captured["pkg"] = pkg
            return "rendered"

        with patch("cron.scheduler._briefing_decide_call", return_value=dict(self._PKG)), \
             patch("cron.scheduler._briefing_write_call", side_effect=fake_write):
            out = _run_two_step_briefing("raw", "j", silence_tier="day5")
        assert out == "rendered"
        assert captured["pkg"].get("silence_tier") == "day5"

    def test_no_silence_tier_leaves_package_clean(self):
        captured = {}

        def fake_write(pkg, job_id="?"):
            captured["pkg"] = pkg
            return "rendered"

        with patch("cron.scheduler._briefing_decide_call", return_value=dict(self._PKG)), \
             patch("cron.scheduler._briefing_write_call", side_effect=fake_write):
            _run_two_step_briefing("raw", "j")
        assert "silence_tier" not in captured["pkg"]


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

    def test_silent_directive_follows_footer(self, tmp_path):
        """When a new user (footer window) is also silent, [SILENT] wins by
        coming last."""
        gh, sr = _patch_script(tmp_path, {"tier": "day8", "speak": False, "reason": "x"})
        job = _briefing_job()
        job["repeat"] = {"completed": 0}  # inside FOOTER_REQUIRED window
        with gh, sr, patch("tools.skills_tool.skill_view",
                           return_value=json.dumps({"success": True, "content": "briefing skill"})):
            out = _build_job_prompt(job)
        assert "FOOTER_REQUIRED" in out
        assert "do not send a briefing today" in out  # the silent-gap directive
        # silence directive comes after the footer so its [SILENT] instruction
        # dominates. (Anchor on SILENCE_AWARENESS — the cron hint also mentions
        # [SILENT] near the top of the prompt.)
        assert out.index("FOOTER_REQUIRED") < out.index("SILENCE_AWARENESS")
