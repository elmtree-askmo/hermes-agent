"""Tests for the plugin gateway-command registration API.

``PluginContext.register_gateway_command()`` completes the half-built
upstream plugin-command interface: the gateway dispatch in
``gateway/run.py`` already looks up handlers via
``get_plugin_command_handler``, but the registration side did not exist.
Registration must do BOTH halves:

1. store the handler so gateway dispatch can invoke it, and
2. register a ``CommandDef`` so the command appears in
   ``slack_subcommand_map()`` — without this, ``/hermes <cmd>`` is treated
   as a regular question and silently sent to the LLM.
"""

import pytest

from hermes_cli import commands as commands_mod
from hermes_cli import plugins as plugins_mod
from hermes_cli.commands import slack_subcommand_map
from hermes_cli.plugins import (
    PluginContext,
    PluginManifest,
    PluginManager,
    get_plugin_command_handler,
    get_plugin_gateway_commands,
)


@pytest.fixture()
def plugin_ctx(monkeypatch):
    """Fresh plugin manager + command-registry snapshot/restore."""
    registry_snapshot = list(commands_mod.COMMAND_REGISTRY)
    # Purge CommandDefs a prior test's real plugin discovery may have left
    # behind (locally-installed artemis-debug registers "debug") so this
    # fixture's own registrations don't hit the collision guard.
    commands_mod.COMMAND_REGISTRY[:] = [
        c for c in commands_mod.COMMAND_REGISTRY if c.name not in ("debug", "testdbg")
    ]
    commands_mod.rebuild_lookups()
    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    ctx = PluginContext(PluginManifest(name="test-plugin"), manager)
    yield ctx
    commands_mod.COMMAND_REGISTRY[:] = registry_snapshot
    commands_mod.rebuild_lookups()


def _handler(args):
    return f"echo:{args}"


class TestRegistration:
    def test_handler_retrievable(self, plugin_ctx):
        plugin_ctx.register_gateway_command("testdbg", "Test debug", _handler)
        assert get_plugin_command_handler("testdbg") is _handler

    def test_lookup_normalizes_slash_and_case(self, plugin_ctx):
        plugin_ctx.register_gateway_command("testdbg", "Test debug", _handler)
        assert get_plugin_command_handler("/testdbg") is _handler
        assert get_plugin_command_handler("TESTDBG") is _handler

    def test_unknown_command_returns_none(self, plugin_ctx):
        assert get_plugin_command_handler("no-such-command") is None

    def test_appears_in_slack_subcommand_map(self, plugin_ctx):
        """The CommandDef half: without it, /hermes testdbg falls through
        to the LLM as a regular question."""
        plugin_ctx.register_gateway_command("testdbg", "Test debug", _handler)
        mapping = slack_subcommand_map()
        assert mapping.get("testdbg") == "/testdbg"

    def test_metadata_recorded(self, plugin_ctx):
        plugin_ctx.register_gateway_command("testdbg", "Test debug", _handler)
        meta = get_plugin_gateway_commands()["testdbg"]
        assert meta["plugin"] == "test-plugin"
        assert meta["help"] == "Test debug"


class TestGuards:
    def test_builtin_collision_refused(self, plugin_ctx):
        plugin_ctx.register_gateway_command("status", "Shadow status", _handler)
        assert get_plugin_command_handler("status") is None

    def test_builtin_alias_collision_refused(self, plugin_ctx):
        plugin_ctx.register_gateway_command("reset", "Shadow reset alias", _handler)
        assert get_plugin_command_handler("reset") is None

    def test_empty_name_refused(self, plugin_ctx):
        plugin_ctx.register_gateway_command("  ", "Empty", _handler)
        assert get_plugin_gateway_commands() == {}

    def test_duplicate_plugin_registration_refused(self, plugin_ctx):
        plugin_ctx.register_gateway_command("testdbg", "First", _handler)

        def other(args):
            return "other"

        plugin_ctx.register_gateway_command("testdbg", "Second", other)
        # First registration wins; the second is refused as a collision.
        assert get_plugin_command_handler("testdbg") is _handler


class TestNativeCommandSync:
    """Gateway commands must reach the shared platform command collection
    (Discord native slash sync, Telegram menu) — codex round-2 P2. The
    collected ``cmd_key`` must be the dispatchable ``/<name>`` text: Discord
    dispatches ``f"{cmd_key} {args}"``, so an empty placeholder key would
    register a slash command that dispatches an empty string."""

    def test_collection_includes_gateway_command(self, plugin_ctx):
        plugin_ctx.register_gateway_command("testdbg", "Test debug output", _handler)
        from hermes_cli.commands import _collect_gateway_skill_entries

        entries, _hidden = _collect_gateway_skill_entries(
            platform="discord", max_slots=50, reserved_names=set()
        )
        match = [e for e in entries if e[0] == "testdbg"]
        assert match, f"gateway command missing from collection: {entries}"
        name, desc, cmd_key = match[0]
        assert desc == "Test debug output"
        assert cmd_key == "/testdbg"

    def test_discord_skill_commands_offers_gateway_command(self, plugin_ctx):
        plugin_ctx.register_gateway_command("testdbg", "Test debug output", _handler)
        from hermes_cli.commands import discord_skill_commands

        entries, _hidden = discord_skill_commands(max_slots=50, reserved_names=set())
        assert any(
            name == "testdbg" and cmd_key == "/testdbg"
            for name, _desc, cmd_key in entries
        )
