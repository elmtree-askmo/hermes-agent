"""Tests for gateway-side plugin command dispatch.

Covers:
- identity passing: the dispatch passes ``source`` when the handler
  accepts it (signature-inspected), stays backward compatible when not;
- error containment: a raising handler returns an error string instead of
  falling through to the LLM (silent-LLM-fallback trap);
- plugin discovery at gateway boot (a restart must not leave the registry
  empty for the first slash command);
- running-agent bypass: a plugin command arriving while an agent runs is
  dispatched inline, not injected into the LLM conversation as interrupt
  payload.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import commands as commands_mod
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager


@pytest.fixture()
def plugin_ctx(monkeypatch):
    registry_snapshot = list(commands_mod.COMMAND_REGISTRY)
    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    ctx = PluginContext(PluginManifest(name="test-plugin"), manager)
    yield ctx
    commands_mod.COMMAND_REGISTRY[:] = registry_snapshot
    commands_mod.rebuild_lookups()


@pytest.fixture()
def runner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # _handle_message checks user authorization before any dispatch. Grant
    # it explicitly — on dev machines the flag leaks in from ~/.hermes/.env
    # (loaded override=True at runner init), which masked this dependency;
    # clean CI has no such env and would deny U777 before the code under
    # test is ever reached (codex r5).
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    config = GatewayConfig(platforms={}, sessions_dir=tmp_path / "sessions")
    return GatewayRunner(config)


def _source():
    return SessionSource(
        platform=Platform.SLACK, chat_id="D123", chat_type="dm", user_id="U777"
    )


def _event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=_source(),
    )


class TestDispatchIdentity:
    @pytest.mark.asyncio
    async def test_source_passed_when_handler_accepts_it(self, runner):
        seen = {}

        def handler(args, source=None):
            seen["args"] = args
            seen["source"] = source
            return "ok"

        result = await runner._dispatch_plugin_command(handler, _event("/testdbg foo bar"), _source())
        assert result == "ok"
        assert seen["args"] == "foo bar"
        assert seen["source"].user_id == "U777"

    @pytest.mark.asyncio
    async def test_var_keyword_handler_receives_source(self, runner):
        seen = {}

        def handler(args, **kwargs):
            seen.update(kwargs)
            return "ok"

        await runner._dispatch_plugin_command(handler, _event("/testdbg"), _source())
        assert seen["source"].user_id == "U777"

    @pytest.mark.asyncio
    async def test_legacy_handler_without_source_still_works(self, runner):
        seen = {}

        def handler(args):
            seen["args"] = args
            return "legacy"

        result = await runner._dispatch_plugin_command(handler, _event("/testdbg x"), _source())
        assert result == "legacy"
        assert seen["args"] == "x"

    @pytest.mark.asyncio
    async def test_async_handler_awaited(self, runner):
        async def handler(args, source=None):
            await asyncio.sleep(0)
            return "async-ok"

        result = await runner._dispatch_plugin_command(handler, _event("/testdbg"), _source())
        assert result == "async-ok"


class TestDispatchErrorContainment:
    @pytest.mark.asyncio
    async def test_raising_handler_returns_error_string(self, runner):
        def handler(args, source=None):
            raise RuntimeError("boom")

        result = await runner._dispatch_plugin_command(handler, _event("/testdbg"), _source())
        assert result is not None
        assert "boom" in result
        assert result.startswith("⚠")

    @pytest.mark.asyncio
    async def test_empty_result_returns_none(self, runner):
        result = await runner._dispatch_plugin_command(
            lambda args: "", _event("/testdbg"), _source()
        )
        assert result is None


class TestBootDiscovery:
    @pytest.mark.asyncio
    async def test_start_calls_discover_plugins(self, runner, monkeypatch):
        called = {"n": 0}

        def fake_discover():
            called["n"] += 1

        monkeypatch.setattr(plugins_mod, "discover_plugins", fake_discover)
        await runner.start()
        assert called["n"] == 1

    @pytest.mark.asyncio
    async def test_start_survives_discovery_failure(self, runner, monkeypatch):
        def broken_discover():
            raise RuntimeError("plugin dir unreadable")

        monkeypatch.setattr(plugins_mod, "discover_plugins", broken_discover)
        # Must not raise — discovery failure is logged, boot continues.
        await runner.start()


class TestRunningAgentBypass:
    @pytest.mark.asyncio
    async def test_plugin_command_dispatched_while_agent_running(self, runner, plugin_ctx):
        """/testdbg while an agent runs must answer inline — not interrupt
        the agent with the command text as conversation payload."""
        plugin_ctx.register_gateway_command(
            "testdbg", "Test debug", lambda args, source=None: f"state:{source.user_id}"
        )

        event = _event("/testdbg")
        session_key = runner._session_key_for_source(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent

        result = await runner._handle_message(event)

        assert result == "state:U777"
        running_agent.interrupt.assert_not_called()
