"""Skip the pre-reset memory flush when every flush target is disabled.

The flush agent has exactly three write paths: memory (MEMORY.md),
user profile (USER.md), and skill_manage. Multi-user deployments disable
all three (memory lives in an external provider, the skills dir is global)
— in that configuration the flush run reads the full session transcript
into an LLM call and is guaranteed to write nothing (skill_manage errors
at runtime; the prompt forbids the memory tool). Skip the run entirely.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    """gateway.run imports dotenv at module level; stub it so tests run without the package."""
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.session_store = MagicMock()
    return runner


_TRANSCRIPT_4_MSGS = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "remember my name is Alice"},
    {"role": "assistant", "content": "Got it, Alice!"},
]


def _run_flush(monkeypatch, config):
    """Run the flush with a controlled config; return the mocked AIAgent class."""
    ai_agent_cls = MagicMock()
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ai_agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    runner = _make_runner()
    runner.session_store.load_transcript.return_value = _TRANSCRIPT_4_MSGS

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("hermes_cli.config.load_config", **config),
        patch.dict("sys.modules", {"tools.memory_tool": MagicMock()}),
    ):
        runner._flush_memories_for_session("session_abc")
    return ai_agent_cls


class TestAllTargetsDisabled:
    def test_all_disabled_skips_flush_run(self, monkeypatch):
        """memory + user_profile + skill_manage all off → no LLM run at all."""
        cfg = {
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "skills": {"skill_manage_enabled": False},
        }
        ai_agent_cls = _run_flush(monkeypatch, {"return_value": cfg})
        ai_agent_cls.assert_not_called()

    def test_skills_section_missing_defaults_disabled(self, monkeypatch):
        """skill_manage defaults to disabled (mirrors skill_manager_tool) —
        memory off + user off + no skills section → skip."""
        cfg = {"memory": {"memory_enabled": False, "user_profile_enabled": False}}
        ai_agent_cls = _run_flush(monkeypatch, {"return_value": cfg})
        ai_agent_cls.assert_not_called()


class TestAnyTargetEnabledProceeds:
    def test_memory_enabled_proceeds(self, monkeypatch):
        cfg = {
            "memory": {"memory_enabled": True, "user_profile_enabled": False},
            "skills": {"skill_manage_enabled": False},
        }
        ai_agent_cls = _run_flush(monkeypatch, {"return_value": cfg})
        ai_agent_cls.assert_called_once()

    def test_user_profile_enabled_proceeds(self, monkeypatch):
        cfg = {
            "memory": {"memory_enabled": False, "user_profile_enabled": True},
            "skills": {"skill_manage_enabled": False},
        }
        ai_agent_cls = _run_flush(monkeypatch, {"return_value": cfg})
        ai_agent_cls.assert_called_once()

    def test_skill_manage_enabled_proceeds(self, monkeypatch):
        cfg = {
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "skills": {"skill_manage_enabled": True},
        }
        ai_agent_cls = _run_flush(monkeypatch, {"return_value": cfg})
        ai_agent_cls.assert_called_once()

    def test_config_load_failure_fails_open(self, monkeypatch):
        """load_config raising keeps today's behavior: flush proceeds."""
        ai_agent_cls = _run_flush(
            monkeypatch, {"side_effect": RuntimeError("no config")}
        )
        ai_agent_cls.assert_called_once()
