"""G1 (S-0429-01) — user_id ContextVar wiring in tools.session_context.

The MCP subprocess spawn path (``tools/mcp_tool._run_stdio``) and the cron
scheduler both read this ContextVar to materialize ``HERMES_SESSION_USER_ID``
in subprocess env. Without the ContextVar, the spawn site has no way to
know which user the gateway is currently serving — the asyncio-task-local
scope is what keeps concurrent Slack handlers from clobbering each other.
"""
from __future__ import annotations

import pytest

from tools import session_context as sc


def test_user_id_default_none():
    sc.clear_session()
    assert sc.get_user_id() is None


def test_user_id_set_and_get():
    sc.clear_session()
    sc.set_session(platform="slack", chat_id="D1", user_id="U0AQW54L1UN")
    assert sc.get_user_id() == "U0AQW54L1UN"


def test_user_id_cleared():
    sc.set_session(platform="slack", chat_id="D1", user_id="U0X")
    sc.clear_session()
    assert sc.get_user_id() is None


def test_user_id_optional_kwarg_defaults_to_none():
    """Backward compat: existing callers that don't pass user_id keep working."""
    sc.clear_session()
    sc.set_session(platform="slack", chat_id="D1")
    assert sc.get_user_id() is None
