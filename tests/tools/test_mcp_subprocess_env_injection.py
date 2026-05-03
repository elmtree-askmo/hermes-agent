"""G1 (S-0429-01) — MCP stdio subprocess inherits session-bound user_id via env.

ContextVars are in-process, asyncio-task-local. The MCP server runs as a
separate subprocess and can't see them. ``_run_stdio`` materializes the
session ContextVar values into the spawn env so the Artemis MCP server can
read ``HERMES_SESSION_USER_ID`` and bind every handler to the right user.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import session_context as sc


def _make_mcp_tool(name="echo"):
    tool = SimpleNamespace()
    tool.name = name
    tool.description = "test"
    tool.inputSchema = {"type": "object", "properties": {}, "required": []}
    return tool


def _mock_stdio_and_session(session):
    mock_read, mock_write = MagicMock(), MagicMock()
    mock_stdio_cm = MagicMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)
    mock_cs_cm = MagicMock()
    mock_cs_cm.__aenter__ = AsyncMock(return_value=session)
    mock_cs_cm.__aexit__ = AsyncMock(return_value=False)
    return (
        patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm),
        patch("tools.mcp_tool.ClientSession", return_value=mock_cs_cm),
    )


class TestSpawnEnvSessionUserId:
    """``_run_stdio`` injects ContextVar-sourced session env into the spawn
    StdioServerParameters."""

    def _run_with_session(self, *, user_id, chat_id="D1", platform="slack"):
        """Spawn an MCP subprocess with the given session ContextVars set;
        return the StdioServerParameters env that was passed to the SDK."""
        from tools.mcp_tool import MCPServerTask

        sc.clear_session()
        if user_id is not None or chat_id is not None or platform is not None:
            sc.set_session(
                platform=platform,
                chat_id=chat_id,
                user_id=user_id,
            )

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mcp_tool()])
        )
        p_stdio, p_cs = _mock_stdio_and_session(mock_session)

        async def _go():
            with patch("tools.mcp_tool.StdioServerParameters") as mock_params, \
                 p_stdio, p_cs:
                server = MCPServerTask("artemis-tools-test")
                await server.start({"command": "node", "args": []})
                # Capture the env dict passed to the SDK
                env_arg = mock_params.call_args.kwargs.get("env")
                await server.shutdown()
                return env_arg

        return asyncio.run(_go())

    def test_user_id_injected_when_contextvar_set(self):
        env = self._run_with_session(user_id="U0AQW54L1UN")
        assert env is not None
        assert env.get("HERMES_SESSION_USER_ID") == "U0AQW54L1UN"

    def test_user_id_absent_when_contextvar_unset(self):
        sc.clear_session()
        env = self._run_with_session(user_id=None, chat_id=None, platform=None)
        # When no session is bound (CLI / cron-without-origin), env must NOT
        # carry a stale user_id leaked from a prior session or os.environ.
        assert env is not None
        assert "HERMES_SESSION_USER_ID" not in env

    def test_chat_id_and_platform_also_propagated(self):
        """Sanity: the chat_id / platform ContextVars are propagated too —
        prevents a regression where one var works but the others don't."""
        env = self._run_with_session(
            user_id="U0AQW54L1UN", chat_id="D1ABC", platform="slack"
        )
        assert env.get("HERMES_SESSION_CHAT_ID") == "D1ABC"
        assert env.get("HERMES_SESSION_PLATFORM") == "slack"

    def test_stale_os_environ_value_does_not_leak(self):
        """If os.environ has a leftover HERMES_SESSION_USER_ID from a prior
        cron job's finally-cleanup race, ContextVar=None must still produce
        an env without that key — not silently inherit."""
        sc.clear_session()
        with patch.dict(
            "os.environ",
            {"HERMES_SESSION_USER_ID": "U0LEAKED"},
            clear=False,
        ):
            env = self._run_with_session(
                user_id=None, chat_id=None, platform=None
            )
        assert "HERMES_SESSION_USER_ID" not in env, (
            "spawn env leaked HERMES_SESSION_USER_ID from os.environ when "
            "ContextVar was unset; this is the cross-session leak vector "
            "G1 closes"
        )
