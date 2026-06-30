"""Log records carry the run-scoped trace id.

``hermes_logging`` installs a LogRecord factory that stamps every record
with ``trace_tag`` (``" [<trace_id>]"`` during a run, ``""`` outside one) from
the session ContextVar, so ``%(trace_tag)s`` in the format never KeyErrors,
background lines stay clean, and a whole run greps by ``[<trace_id>]``.
"""
from __future__ import annotations

import logging

import hermes_logging
from tools import session_context as sc


def _make_record() -> logging.LogRecord:
    factory = logging.getLogRecordFactory()
    return factory("test", logging.INFO, __file__, 1, "msg", None, None)


def test_record_factory_injects_active_trace_id():
    hermes_logging._install_trace_record_factory()
    sc.set_session(platform="slack", chat_id="D1", trace_id="run123abc")
    try:
        assert _make_record().trace_tag == " [run123abc]"
    finally:
        sc.clear_session()


def test_record_factory_empty_tag_when_no_run():
    """Outside a run the tag is empty (line stays clean) — not a 'trace=-'."""
    hermes_logging._install_trace_record_factory()
    sc.clear_session()
    assert _make_record().trace_tag == ""


def test_install_is_idempotent():
    """Calling the installer twice must not double-wrap the factory."""
    hermes_logging._install_trace_record_factory()
    first = logging.getLogRecordFactory()
    hermes_logging._install_trace_record_factory()
    assert logging.getLogRecordFactory() is first


def test_format_string_includes_trace_field():
    assert "%(trace_tag)s" in hermes_logging._LOG_FORMAT
    assert "%(trace_tag)s" in hermes_logging._LOG_FORMAT_VERBOSE


def test_record_factory_falls_back_to_env_in_subprocess(monkeypatch):
    """Spawned Strategist/Executor have no ContextVar but inherit
    HERMES_TRACE_ID env — the factory must read it so their logs join the run."""
    hermes_logging._install_trace_record_factory()
    sc.clear_session()  # ContextVar unset, mimicking a fresh subprocess
    monkeypatch.setenv("HERMES_TRACE_ID", "envtrace99")
    assert _make_record().trace_tag == " [envtrace99]"


def test_contextvar_wins_over_env(monkeypatch):
    """In the gateway both could be set; the per-turn ContextVar must win."""
    hermes_logging._install_trace_record_factory()
    monkeypatch.setenv("HERMES_TRACE_ID", "envtrace99")
    sc.set_session(platform="slack", chat_id="D1", trace_id="ctxwins01")
    try:
        assert _make_record().trace_tag == " [ctxwins01]"
    finally:
        sc.clear_session()
