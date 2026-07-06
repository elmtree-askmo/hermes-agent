"""Tests for Slack slash-command plugin dispatch, ephemeral delivery, and
strict-subcommand mode.

- Plugin gateway commands (e.g. /hermes debug) are dispatched directly to
  the message handler and the reply is POSTed to the slash command's
  ``response_url`` as an *ephemeral* message — never chat.postMessage,
  which would leak personal state into public channels and pollute the
  Coach DM under test.
- Strict-subcommand mode (``SLACK_STRICT_SUBCOMMANDS``, default off): an
  unmatched first token gets a deterministic ephemeral rejection (with a
  did-you-mean hint at edit distance 1) and never reaches the LLM. With
  the flag off, upstream ask-the-agent fallthrough is unchanged.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.slack import SlackAdapter, _edit_distance_leq1
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
def adapter(monkeypatch):
    monkeypatch.delenv("SLACK_STRICT_SUBCOMMANDS", raising=False)
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a.handle_message = AsyncMock()
    a._post_response_url = AsyncMock(return_value=True)
    return a


def _slash_payload(text):
    return {
        "text": text,
        "user_id": "U777",
        "channel_id": "D123",
        "team_id": "T1",
        "response_url": "https://hooks.slack.com/commands/T1/123/abc",
    }


class TestPluginCommandEphemeralDelivery:
    @pytest.mark.asyncio
    async def test_plugin_command_replies_via_response_url(self, adapter, plugin_ctx):
        plugin_ctx.register_gateway_command(
            "testdbg", "Test debug", lambda args, source=None: "digest-output"
        )
        handler = AsyncMock(return_value="digest-output")
        adapter._message_handler = handler

        await adapter._handle_slash_command(_slash_payload("testdbg"))

        handler.assert_awaited_once()
        event = handler.await_args.args[0]
        assert event.text == "/testdbg"
        assert event.source.user_id == "U777"
        adapter._post_response_url.assert_awaited_once_with(
            "https://hooks.slack.com/commands/T1/123/abc", "digest-output"
        )
        # Never routed through the normal channel-post path.
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_plugin_command_args_preserved(self, adapter, plugin_ctx):
        plugin_ctx.register_gateway_command(
            "testdbg", "Test debug", lambda args, source=None: "ok"
        )
        handler = AsyncMock(return_value="ok")
        adapter._message_handler = handler

        await adapter._handle_slash_command(_slash_payload("testdbg raw strategy.json"))

        event = handler.await_args.args[0]
        assert event.text == "/testdbg raw strategy.json"

    @pytest.mark.asyncio
    async def test_handler_exception_reported_ephemerally(self, adapter, plugin_ctx):
        plugin_ctx.register_gateway_command(
            "testdbg", "Test debug", lambda args, source=None: "unused"
        )
        adapter._message_handler = AsyncMock(side_effect=RuntimeError("boom"))

        await adapter._handle_slash_command(_slash_payload("testdbg"))

        posted = adapter._post_response_url.await_args.args[1]
        assert "boom" in posted
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_builtin_command_unaffected(self, adapter, plugin_ctx):
        await adapter._handle_slash_command(_slash_payload("status"))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/status"
        adapter._post_response_url.assert_not_awaited()


class TestStrictSubcommandMode:
    @pytest.mark.asyncio
    async def test_flag_off_falls_through_to_llm(self, adapter):
        """Upstream default: unmatched text is a regular question."""
        await adapter._handle_slash_command(_slash_payload("what is my status?"))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "what is my status?"
        adapter._post_response_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_on_rejects_unknown_token(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")

        await adapter._handle_slash_command(_slash_payload("frobnicate now"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `frobnicate`" in posted
        assert "Available:" in posted

    @pytest.mark.asyncio
    async def test_flag_on_did_you_mean_hint(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")

        await adapter._handle_slash_command(_slash_payload("statu"))

        posted = adapter._post_response_url.await_args.args[1]
        assert "Did you mean `status`?" in posted

    @pytest.mark.asyncio
    async def test_flag_on_typoed_plugin_command_rejected(self, adapter, plugin_ctx, monkeypatch):
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")
        plugin_ctx.register_gateway_command(
            "debug", "Debug digest", lambda args, source=None: "digest"
        )

        await adapter._handle_slash_command(_slash_payload("debg"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Did you mean `debug`?" in posted

    @pytest.mark.asyncio
    async def test_flag_on_registered_subcommand_unaffected(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")

        await adapter._handle_slash_command(_slash_payload("status"))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/status"

    @pytest.mark.asyncio
    async def test_flag_on_empty_text_still_helps(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")

        await adapter._handle_slash_command(_slash_payload(""))

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/help"


class TestPostResponseUrl:
    @pytest.mark.asyncio
    async def test_payload_is_ephemeral(self, monkeypatch):
        config = PlatformConfig(enabled=True, token="xoxb-fake-token")
        adapter = SlackAdapter(config)
        captured = {}

        class _FakeResponse:
            status_code = 200
            text = ""

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None):
                captured["url"] = url
                captured["json"] = json
                return _FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

        ok = await adapter._post_response_url("https://hooks.slack.com/x", "hello")
        assert ok is True
        assert captured["json"] == {"response_type": "ephemeral", "text": "hello"}

    @pytest.mark.asyncio
    async def test_network_failure_returns_false(self, monkeypatch):
        config = PlatformConfig(enabled=True, token="xoxb-fake-token")
        adapter = SlackAdapter(config)

        class _BrokenClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                raise ConnectionError("no network")

            async def __aexit__(self, *exc):
                return False

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _BrokenClient)

        ok = await adapter._post_response_url("https://hooks.slack.com/x", "hello")
        assert ok is False


class TestEditDistance:
    def test_substitution(self):
        assert _edit_distance_leq1("debag", "debug")

    def test_deletion(self):
        assert _edit_distance_leq1("debg", "debug")

    def test_insertion(self):
        assert _edit_distance_leq1("debugg", "debug")

    def test_exact_match_is_false(self):
        assert not _edit_distance_leq1("debug", "debug")

    def test_distance_two_is_false(self):
        assert not _edit_distance_leq1("dbg", "debug")
        assert not _edit_distance_leq1("status", "debug")
