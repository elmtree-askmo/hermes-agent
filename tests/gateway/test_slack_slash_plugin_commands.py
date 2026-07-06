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
    # An earlier test may have run real plugin discovery (gateway boot) and
    # registered a locally-installed plugin's CommandDef (e.g. artemis-debug's
    # "debug") into the global registry — which would make this fixture's own
    # registrations collide. Purge those names; the snapshot restores them.
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


@pytest.fixture()
def adapter(monkeypatch):
    monkeypatch.delenv("SLACK_STRICT_SUBCOMMANDS", raising=False)
    monkeypatch.delenv("HERMES_ARTEMIS_ENABLED", raising=False)
    monkeypatch.delenv("SLACK_SUBCOMMAND_ALLOWLIST", raising=False)
    monkeypatch.delenv("SLACK_SLASH_COMMANDS", raising=False)
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a.handle_message = AsyncMock()
    a._post_response_url = AsyncMock(return_value=True)
    return a


def _slash_payload(text, command="/artemis"):
    return {
        "command": command,
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
    async def test_flag_on_case_slip_gets_did_you_mean(self, adapter, monkeypatch):
        """`DEBUG` is distance 0 from `debug` after lowering — the edit-
        distance-1 check alone would miss it and reject with no hint."""
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")

        await adapter._handle_slash_command(_slash_payload("STATUS"))

        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `STATUS`" in posted
        assert "Did you mean `status`?" in posted

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

    @pytest.mark.asyncio
    async def test_artemis_deployment_defaults_to_strict(self, adapter, monkeypatch):
        """HERMES_ARTEMIS_ENABLED alone turns strict mode on — no separate
        deploy-time knob to forget."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")

        await adapter._handle_slash_command(_slash_payload("frobnicate now"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `frobnicate`" in posted

    @pytest.mark.asyncio
    async def test_explicit_false_overrides_artemis_default(self, adapter, monkeypatch):
        """SLACK_STRICT_SUBCOMMANDS=false restores upstream fallthrough even
        on an Artemis deployment (debug escape hatch)."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "false")

        await adapter._handle_slash_command(_slash_payload("what is my status?"))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "what is my status?"
        adapter._post_response_url.assert_not_awaited()


class TestSubcommandAllowlist:
    @pytest.mark.asyncio
    async def test_artemis_default_blocks_builtin_operator_commands(self, adapter, plugin_ctx, monkeypatch):
        """Slash commands are workspace-scoped: every allowlisted Slack user
        can invoke them. The Artemis default exposes ONLY debug — operator
        commands (yolo/model/update/...) must reject."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        plugin_ctx.register_gateway_command(
            "debug", "Debug digest", lambda args, source=None: "digest"
        )

        await adapter._handle_slash_command(_slash_payload("yolo"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `yolo`" in posted
        assert "Available: `debug`" in posted
        assert "`status`" not in posted

    @pytest.mark.asyncio
    async def test_artemis_default_still_dispatches_debug(self, adapter, plugin_ctx, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        plugin_ctx.register_gateway_command(
            "debug", "Debug digest", lambda args, source=None: "digest"
        )
        handler = AsyncMock(return_value="digest")
        adapter._message_handler = handler

        await adapter._handle_slash_command(_slash_payload("debug"))

        handler.assert_awaited_once()
        adapter._post_response_url.assert_awaited_once_with(
            "https://hooks.slack.com/commands/T1/123/abc", "digest"
        )

    @pytest.mark.asyncio
    async def test_alias_of_blocked_command_also_rejected(self, adapter, monkeypatch):
        """`reset` is an alias of `new` — allowlist filtering must cover
        aliases, not just canonical names."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")

        await adapter._handle_slash_command(_slash_payload("reset"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `reset`" in posted

    @pytest.mark.asyncio
    async def test_env_allowlist_widens_exposure(self, adapter, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "debug,status")

        await adapter._handle_slash_command(_slash_payload("status"))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/status"

    @pytest.mark.asyncio
    async def test_allowlist_all_restores_upstream_surface(self, adapter, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "all")

        await adapter._handle_slash_command(_slash_payload("status"))

        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text == "/status"

    @pytest.mark.asyncio
    async def test_bare_invocation_shows_command_overview(self, adapter, plugin_ctx, monkeypatch):
        """/help is not in the Artemis allowlist — a bare invocation must
        answer with the exposed-command overview instead of dispatching /help."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        plugin_ctx.register_gateway_command(
            "debug", "Artemis per-user state digest (zero-LLM)", lambda args, source=None: "digest",
            args_hint="[apps|mem0|raw <file>|snapshot|help]",
        )

        await adapter._handle_slash_command(_slash_payload(""))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Available commands:" in posted
        assert "/artemis debug [apps|mem0|raw <file>|snapshot|help] — Artemis per-user state digest (zero-LLM)" in posted

    @pytest.mark.asyncio
    async def test_help_token_shows_same_overview(self, adapter, plugin_ctx, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        plugin_ctx.register_gateway_command(
            "debug", "Artemis per-user state digest (zero-LLM)", lambda args, source=None: "digest",
        )

        await adapter._handle_slash_command(_slash_payload("help"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Available commands:" in posted
        assert "/artemis debug" in posted
        assert "`/artemis <command> help`" in posted

    @pytest.mark.asyncio
    async def test_help_dispatches_upstream_when_allowlisted(self, adapter, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "debug,help")

        await adapter._handle_slash_command(_slash_payload("help"))

        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text == "/help"

    @pytest.mark.asyncio
    async def test_upstream_without_flags_unfiltered(self, adapter):
        await adapter._handle_slash_command(_slash_payload("status"))

        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text == "/status"

    @pytest.mark.asyncio
    async def test_slash_prefixed_bypass_rejected_when_strict_off(self, adapter, monkeypatch):
        """Codex P1: `/hermes /yolo` misses the subcommand map (leading
        slash), and with strict off would fall through as free text — but
        `text.startswith("/")` makes it a COMMAND event, so the gateway
        would execute the built-in and sidestep the allowlist. Must reject
        whenever an allowlist is active, strict or not."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "false")

        await adapter._handle_slash_command(_slash_payload("/yolo"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `/yolo`" in posted

    @pytest.mark.asyncio
    async def test_slash_prefixed_bypass_rejected_upstream_allowlist(self, adapter, monkeypatch):
        """Same bypass on a plain upstream deployment that opted into an
        allowlist without strict mode."""
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "debug")

        await adapter._handle_slash_command(_slash_payload("/yolo", command="/hermes"))

        adapter.handle_message.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "Unknown command: `/yolo`" in posted

    @pytest.mark.asyncio
    async def test_slash_prefixed_falls_through_without_allowlist(self, adapter):
        """No allowlist = upstream semantics unchanged: slash-prefixed text
        still reaches the gateway as a COMMAND event (every registry command
        is legitimately invocable, so no boundary is crossed)."""
        await adapter._handle_slash_command(_slash_payload("/status", command="/hermes"))

        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text == "/status"


class TestSlashCommandNames:
    def test_upstream_default(self, monkeypatch):
        from gateway.platforms.slack import _slash_command_names

        monkeypatch.delenv("SLACK_SLASH_COMMANDS", raising=False)
        monkeypatch.delenv("HERMES_ARTEMIS_ENABLED", raising=False)
        assert _slash_command_names() == ["/hermes"]

    def test_artemis_listens_on_artemis_only(self, monkeypatch):
        """Artemis never registers a /hermes handler — the full upstream
        command surface is unreachable even if the app manifest lags."""
        from gateway.platforms.slack import _slash_command_names

        monkeypatch.delenv("SLACK_SLASH_COMMANDS", raising=False)
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        assert _slash_command_names() == ["/artemis"]

    def test_env_override_normalizes_slashes(self, monkeypatch):
        from gateway.platforms.slack import _slash_command_names

        monkeypatch.setenv("SLACK_SLASH_COMMANDS", "artemis, /coach")
        assert _slash_command_names() == ["/artemis", "/coach"]


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


class TestSlashAuthorizationGate:
    """Pre-dispatch replies (help overview, unknown-subcommand rejection)
    expose the internal command surface, and slash commands are
    workspace-scoped — codex round-2 P2. With the gateway's authorization
    check wired, unauthorized invokers get the same private-beta reply the
    gateway gives unauthorized DMs, and nothing else."""

    @pytest.mark.asyncio
    async def test_unauthorized_bare_invocation_gets_no_command_surface(
        self, adapter, monkeypatch
    ):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        adapter.set_authorization_check(lambda source: False)

        await adapter._handle_slash_command(_slash_payload(""))

        posted = adapter._post_response_url.await_args.args[1]
        assert "private beta" in posted
        assert "Available" not in posted
        assert "debug" not in posted
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_typo_gets_no_available_list(
        self, adapter, monkeypatch
    ):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        adapter.set_authorization_check(lambda source: False)

        await adapter._handle_slash_command(_slash_payload("debg"))

        posted = adapter._post_response_url.await_args.args[1]
        assert "private beta" in posted
        assert "Unknown command" not in posted
        assert "Available" not in posted

    @pytest.mark.asyncio
    async def test_unauthorized_plugin_dispatch_blocked(
        self, adapter, plugin_ctx, monkeypatch
    ):
        plugin_ctx.register_gateway_command(
            "testdbg", "Test debug", lambda args, source=None: "digest-output"
        )
        handler = AsyncMock(return_value="digest-output")
        adapter._message_handler = handler
        adapter.set_authorization_check(lambda source: False)

        await adapter._handle_slash_command(_slash_payload("testdbg"))

        handler.assert_not_awaited()
        posted = adapter._post_response_url.await_args.args[1]
        assert "private beta" in posted

    @pytest.mark.asyncio
    async def test_authorized_invoker_unchanged(self, adapter, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        seen = []

        def check(source):
            seen.append(source)
            return True

        adapter.set_authorization_check(check)

        await adapter._handle_slash_command(_slash_payload(""))

        posted = adapter._post_response_url.await_args.args[1]
        assert "Available commands:" in posted
        assert seen and seen[0].user_id == "U777"

    @pytest.mark.asyncio
    async def test_check_exception_fails_closed(self, adapter, monkeypatch):
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")

        def check(source):
            raise RuntimeError("auth backend down")

        adapter.set_authorization_check(check)

        await adapter._handle_slash_command(_slash_payload(""))

        posted = adapter._post_response_url.await_args.args[1]
        assert "private beta" in posted
        assert "Available" not in posted

    @pytest.mark.asyncio
    async def test_no_check_wired_keeps_legacy_behavior(self, adapter, monkeypatch):
        """Standalone adapter (no gateway runner): pre-dispatch replies work
        as before; dispatch paths still hit the gateway's own auth check."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")

        await adapter._handle_slash_command(_slash_payload(""))

        posted = adapter._post_response_url.await_args.args[1]
        assert "Available commands:" in posted


class TestAliasAllowlistCanonicalization:
    """Slack-only aliases ("compact" -> /compress) don't resolve via
    resolve_command — codex round-2 P3. Allowlisting the canonical command
    must keep its aliases working, and the rejection's Available list must
    not show aliases as pseudo-canonicals."""

    @pytest.mark.asyncio
    async def test_compact_survives_compress_allowlist(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "compress")

        await adapter._handle_slash_command(
            _slash_payload("compact keep it short", command="/hermes")
        )

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/compress keep it short"

    @pytest.mark.asyncio
    async def test_alias_name_in_allowlist_keeps_alias(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "compact")

        await adapter._handle_slash_command(
            _slash_payload("compact keep it short", command="/hermes")
        )

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/compress keep it short"

    @pytest.mark.asyncio
    async def test_available_list_hides_slack_only_alias(self, adapter, monkeypatch):
        monkeypatch.setenv("SLACK_SUBCOMMAND_ALLOWLIST", "compress")
        monkeypatch.setenv("SLACK_STRICT_SUBCOMMANDS", "true")

        await adapter._handle_slash_command(_slash_payload("zzz", command="/hermes"))

        posted = adapter._post_response_url.await_args.args[1]
        assert "`compress`" in posted
        assert "compact" not in posted


class TestSlashPrefixedPluginDispatch:
    """`/hermes /debug` must behave like `/hermes debug` — codex r4 P2. A
    slash-prefixed plugin command that missed the subcommand map was built
    as a COMMAND event and delivered via the public channel path instead of
    the ephemeral response_url."""

    @pytest.mark.asyncio
    async def test_slash_prefixed_plugin_command_stays_ephemeral(
        self, adapter, plugin_ctx
    ):
        """Upstream (no allowlist, strict off): the exact reported case."""
        plugin_ctx.register_gateway_command(
            "testdbg", "Test debug", lambda args, source=None: "digest-output"
        )
        handler = AsyncMock(return_value="digest-output")
        adapter._message_handler = handler

        await adapter._handle_slash_command(
            _slash_payload("/testdbg raw strategy.json", command="/hermes")
        )

        handler.assert_awaited_once()
        event = handler.await_args.args[0]
        assert event.text == "/testdbg raw strategy.json"
        adapter._post_response_url.assert_awaited_once_with(
            "https://hooks.slack.com/commands/T1/123/abc", "digest-output"
        )
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slash_prefixed_allowlisted_command_dispatches(
        self, adapter, plugin_ctx, monkeypatch
    ):
        """Artemis: `/artemis /debug` matches the allowlisted subcommand."""
        monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")
        plugin_ctx.register_gateway_command(
            "debug", "Debug digest", lambda args, source=None: "digest"
        )
        handler = AsyncMock(return_value="digest")
        adapter._message_handler = handler

        await adapter._handle_slash_command(_slash_payload("/debug"))

        handler.assert_awaited_once()
        assert handler.await_args.args[0].text == "/debug"

    @pytest.mark.asyncio
    async def test_slash_prefixed_builtin_unchanged(self, adapter):
        """`/hermes /status` keeps dispatching the built-in as before."""
        await adapter._handle_slash_command(
            _slash_payload("/status", command="/hermes")
        )

        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text == "/status"


class TestResponseUrlLengthGuard:
    """response_url shares chat.postMessage's 40K per-message cap; Slack
    allows up to 5 responses per response_url in 30 min. Oversized output
    is chunked into up to 5 ephemeral posts; beyond that it truncates with
    an explicit notice (local review finding)."""

    @staticmethod
    def _fake_httpx(monkeypatch, posted):
        class FakeResponse:
            status_code = 200
            text = "ok"

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None):
                posted.append(json)
                return FakeResponse()

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    @pytest.mark.asyncio
    async def test_oversized_output_chunked(self, adapter, monkeypatch):
        import gateway.platforms.slack as slack_mod

        posted = []
        self._fake_httpx(monkeypatch, posted)
        # Use the real _post_response_url (fixture replaces it with a mock).
        real = slack_mod.SlackAdapter._post_response_url
        big = "x" * (adapter.MAX_MESSAGE_LENGTH + 5000)

        ok = await real(adapter, "https://hooks.slack.com/commands/T1/1/a", big)

        assert ok is True
        assert len(posted) == 2
        assert all(len(p["text"]) <= adapter.MAX_MESSAGE_LENGTH for p in posted)
        assert all(p["response_type"] == "ephemeral" for p in posted)
        # No content lost within the 5-post budget.
        joined = "".join(p["text"] for p in posted)
        assert joined.count("x") == len(big)

    @pytest.mark.asyncio
    async def test_beyond_five_chunks_truncated_with_notice(self, adapter, monkeypatch):
        import gateway.platforms.slack as slack_mod

        posted = []
        self._fake_httpx(monkeypatch, posted)
        real = slack_mod.SlackAdapter._post_response_url
        huge = "x" * (adapter.MAX_MESSAGE_LENGTH * 6)

        ok = await real(adapter, "https://hooks.slack.com/commands/T1/1/a", huge)

        assert ok is True
        assert len(posted) == 5
        assert "more part(s) dropped" in posted[-1]["text"]
        # The notice append must not push the final post over the budget.
        assert all(len(p["text"]) <= adapter.MAX_MESSAGE_LENGTH for p in posted)

    @pytest.mark.asyncio
    async def test_normal_output_single_untouched_post(self, adapter, monkeypatch):
        import gateway.platforms.slack as slack_mod

        posted = []
        self._fake_httpx(monkeypatch, posted)
        real = slack_mod.SlackAdapter._post_response_url

        await real(adapter, "https://hooks.slack.com/commands/T1/1/a", "digest")

        assert len(posted) == 1
        assert posted[0]["text"] == "digest"
