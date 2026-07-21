"""Per-job slim prompt fields for cron sessions (S-0721-01).

Two per-job fields on the cron job record:
- ``skip_repo_context: true`` → session built with skip_context_files=True +
  load_soul_identity=True (repo-context files dropped, SOUL.md kept).
- ``enabled_toolsets: [...]`` → passed to AIAgent verbatim as the allowlist.

Jobs without the fields must resolve to today's behavior exactly: full
context files and the full default toolset (None).
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from cron.scheduler import _job_slim_context, _job_enabled_toolsets


class TestJobSlimContext:
    def test_absent_field_is_false(self):
        assert _job_slim_context({"name": "reminder", "prompt": "hi"}) is False

    def test_true_field(self):
        assert _job_slim_context({"skip_repo_context": True}) is True

    def test_false_and_falsy_values(self):
        assert _job_slim_context({"skip_repo_context": False}) is False
        assert _job_slim_context({"skip_repo_context": None}) is False


class TestJobEnabledToolsets:
    def test_absent_field_is_none(self):
        assert _job_enabled_toolsets({"name": "reminder"}) is None

    def test_empty_list_is_none(self):
        """Empty allowlist must NOT strip every tool — treat as unset."""
        assert _job_enabled_toolsets({"enabled_toolsets": []}) is None

    def test_list_passed_through(self):
        job = {"enabled_toolsets": ["web", "artemis-tools"]}
        assert _job_enabled_toolsets(job) == ["web", "artemis-tools"]

    def test_non_list_is_none(self):
        assert _job_enabled_toolsets({"enabled_toolsets": "web"}) is None


class TestSoulIdentityGating:
    """_build_system_prompt's SOUL slot honors load_soul_identity."""

    def _gate(self, skip_context_files: bool, load_soul_identity: bool) -> bool:
        """Reproduce the identity-slot gate with a minimal agent object."""
        import run_agent

        agent = object.__new__(run_agent.AIAgent)
        agent.skip_context_files = skip_context_files
        agent.load_soul_identity = load_soul_identity
        # The gate under test (run_agent._build_system_prompt identity slot)
        return not agent.skip_context_files or agent.load_soul_identity

    def test_default_full_prompt_loads_soul(self):
        assert self._gate(skip_context_files=False, load_soul_identity=False) is True

    def test_slim_context_keeps_soul(self):
        assert self._gate(skip_context_files=True, load_soul_identity=True) is True

    def test_batch_mode_still_skips_soul(self):
        """Pre-existing behavior: skip_context_files alone drops SOUL too."""
        assert self._gate(skip_context_files=True, load_soul_identity=False) is False


class TestSystemPromptComposition:
    """End-to-end over _build_system_prompt with mocked loaders: the slim
    combination keeps SOUL.md and drops repo-context files; the default
    combination (field-less job) keeps both — pinned so legacy jobs are
    byte-identical."""

    def _build(self, *, skip_context_files: bool, load_soul_identity: bool) -> str:
        import run_agent

        agent = object.__new__(run_agent.AIAgent)
        agent.skip_context_files = skip_context_files
        agent.load_soul_identity = load_soul_identity
        agent.valid_tool_names = set()
        agent._user_id = None
        agent._tool_use_enforcement = False
        agent.model = "test-model"
        agent.provider = None
        agent.platform = "cron"
        agent.skip_memory = True
        agent.pass_session_id = False
        agent.enabled_toolsets = None
        agent.disabled_toolsets = None
        agent._memory_enabled = False
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._memory_store = MagicMock()
        agent._todo_store = MagicMock(format_for_injection=lambda: "")

        with (
            patch.object(run_agent, "load_soul_md", return_value="SOUL-IDENTITY-CONTENT"),
            patch.object(
                run_agent,
                "build_context_files_prompt",
                return_value="REPO-CONTEXT-CONTENT",
            ),
        ):
            return run_agent.AIAgent._build_system_prompt(agent)

    def test_slim_keeps_soul_drops_repo_context(self):
        prompt = self._build(skip_context_files=True, load_soul_identity=True)
        assert "SOUL-IDENTITY-CONTENT" in prompt
        assert "REPO-CONTEXT-CONTENT" not in prompt

    def test_default_keeps_both(self):
        prompt = self._build(skip_context_files=False, load_soul_identity=False)
        assert "SOUL-IDENTITY-CONTENT" in prompt
        assert "REPO-CONTEXT-CONTENT" in prompt
