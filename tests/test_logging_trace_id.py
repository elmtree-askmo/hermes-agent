"""Log records carry the run-scoped trace id.

``hermes_logging`` installs a LogRecord factory that stamps every record
with ``trace_id`` from the session ContextVar (falling back to ``"-"`` when
no run is active), so ``%(trace_id)s`` in the format never KeyErrors and a
whole Coach→Strategist→Executor run can be grepped by one id.
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
        assert _make_record().trace_id == "run123abc"
    finally:
        sc.clear_session()


def test_record_factory_defaults_to_dash_when_no_run():
    hermes_logging._install_trace_record_factory()
    sc.clear_session()
    assert _make_record().trace_id == "-"


def test_install_is_idempotent():
    """Calling the installer twice must not double-wrap the factory."""
    hermes_logging._install_trace_record_factory()
    first = logging.getLogRecordFactory()
    hermes_logging._install_trace_record_factory()
    assert logging.getLogRecordFactory() is first


def test_format_string_includes_trace_field():
    assert "%(trace_id)s" in hermes_logging._LOG_FORMAT
    assert "%(trace_id)s" in hermes_logging._LOG_FORMAT_VERBOSE


def test_record_factory_falls_back_to_env_in_subprocess(monkeypatch):
    """Spawned Strategist/Executor have no ContextVar but inherit
    HERMES_TRACE_ID env — the factory must read it so their logs join the run."""
    hermes_logging._install_trace_record_factory()
    sc.clear_session()  # ContextVar unset, mimicking a fresh subprocess
    monkeypatch.setenv("HERMES_TRACE_ID", "envtrace99")
    assert _make_record().trace_id == "envtrace99"


def test_contextvar_wins_over_env(monkeypatch):
    """In the gateway both could be set; the per-turn ContextVar must win."""
    hermes_logging._install_trace_record_factory()
    monkeypatch.setenv("HERMES_TRACE_ID", "envtrace99")
    sc.set_session(platform="slack", chat_id="D1", trace_id="ctxwins01")
    try:
        assert _make_record().trace_id == "ctxwins01"
    finally:
        sc.clear_session()
