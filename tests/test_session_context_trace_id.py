"""Trace-id ContextVar wiring in tools.session_context.

Mirrors ``session_user_id`` (S-0429-01): an asyncio-task-local run id so
every log line and every spawned subagent (Strategist / Executor) can be
joined back to the one Coach run that produced them. The gateway generates
a fresh id per inbound turn; the MCP subprocess spawn path and the cron
scheduler materialize it into ``HERMES_TRACE_ID`` subprocess env (slice 2).
"""
from __future__ import annotations

from tools import session_context as sc


def test_trace_id_default_none():
    sc.clear_session()
    assert sc.get_trace_id() is None


def test_trace_id_set_and_get():
    sc.clear_session()
    sc.set_session(platform="slack", chat_id="D1", trace_id="abc123def456")
    assert sc.get_trace_id() == "abc123def456"


def test_trace_id_cleared():
    sc.set_session(platform="slack", chat_id="D1", trace_id="abc")
    sc.clear_session()
    assert sc.get_trace_id() is None


def test_trace_id_optional_kwarg_defaults_to_none():
    """Backward compat: existing callers that don't pass trace_id keep working."""
    sc.clear_session()
    sc.set_session(platform="slack", chat_id="D1")
    assert sc.get_trace_id() is None


def test_set_trace_id_standalone():
    """Subprocess startup reads HERMES_TRACE_ID env then sets the ContextVar
    directly (without a full set_session)."""
    sc.clear_session()
    sc.set_trace_id("fromenv01")
    assert sc.get_trace_id() == "fromenv01"


def test_new_trace_id_is_short_hex_and_unique():
    a = sc.new_trace_id()
    b = sc.new_trace_id()
    assert a != b
    assert len(a) == 12
    int(a, 16)  # raises if not hex
