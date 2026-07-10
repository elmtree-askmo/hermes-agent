"""Asyncio-task-local session context.

Replaces the process-wide ``os.environ["HERMES_SESSION_*"]`` pattern where
cross-task isolation matters (e.g. filtering a tool result by the calling
user). ``ContextVar`` values follow each asyncio task independently, so
concurrent message handlers can't clobber each other's session context.

Callers should treat env vars as a fallback for non-async contexts (CLI,
scripts, legacy tests). When running inside the gateway, read from the
ContextVars here first.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

session_platform: ContextVar[Optional[str]] = ContextVar("hermes_session_platform", default=None)
session_chat_id: ContextVar[Optional[str]] = ContextVar("hermes_session_chat_id", default=None)
session_chat_name: ContextVar[Optional[str]] = ContextVar("hermes_session_chat_name", default=None)
session_thread_id: ContextVar[Optional[str]] = ContextVar("hermes_session_thread_id", default=None)
# G1 (S-0429-01): asyncio-task-local Slack/Grid user_id. The MCP subprocess
# spawn path materializes this into ``HERMES_SESSION_USER_ID`` env so the
# Artemis MCP server can bind every handler's user_id to the calling
# session — never to LLM-supplied args.
session_user_id: ContextVar[Optional[str]] = ContextVar("hermes_session_user_id", default=None)

# S-0518-01 (thread consistency): asyncio-task-local Slack thread ts that
# server-side direct pushes (announce_subagent, post_activity_log,
# complete_action's auto activity-log push) should bind to. Differs from
# ``session_thread_id`` (which is set only when the inbound message
# ORIGINATES inside a thread). ``session_thread_ts`` ALWAYS holds the ts
# Coach's main reply will use as thread_ts — equal to
# ``event.source.thread_id or event.message_id`` — so out-of-band server
# pushes (MCP tool side effects) can join the same Slack thread and the
# user sees one cohesive turn instead of split across main + thread.
session_thread_ts: ContextVar[Optional[str]] = ContextVar("hermes_session_thread_ts", default=None)

# Run-scoped trace id. The gateway generates a fresh id per inbound turn (and
# the cron scheduler per job); it follows the asyncio task so every log line
# in the run carries it. The MCP subprocess spawn path materializes it into
# ``HERMES_TRACE_ID`` env so spawned subagents (Strategist / Executor) inherit
# the same id and the whole Coach→Strategist→Executor run joins on one key.
session_trace_id: ContextVar[Optional[str]] = ContextVar("hermes_session_trace_id", default=None)

# Optional per-run trace-NAME override. A normal turn leaves this None and the
# observability plugin names the trace by process role (coach-turn) or by cron
# detection (scheduled). Background maintenance runs in the coach process
# (session-expiry memory flush) set it so their span is named distinctly
# instead of masquerading as a real user coach-turn. Reset by set_session /
# clear_session so a reused pooled thread never leaks it into a later run.
session_trace_name: ContextVar[Optional[str]] = ContextVar("hermes_session_trace_name", default=None)


def new_trace_id() -> str:
    """Generate a short run-scoped trace id (12 hex chars — enough for grep
    uniqueness while keeping log lines readable)."""
    return uuid.uuid4().hex[:12]


def set_session(
    *,
    platform: Optional[str],
    chat_id: Optional[str],
    chat_name: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    trace_name: Optional[str] = None,
) -> None:
    session_platform.set(platform)
    session_chat_id.set(chat_id)
    session_chat_name.set(chat_name)
    session_thread_id.set(thread_id)
    session_user_id.set(user_id)
    session_trace_id.set(trace_id)
    # Defaults to None: a normal turn's set_session wipes any leftover
    # maintenance trace name from a reused thread.
    session_trace_name.set(trace_name)


def clear_session() -> None:
    session_platform.set(None)
    session_chat_id.set(None)
    session_chat_name.set(None)
    session_thread_id.set(None)
    session_user_id.set(None)
    session_thread_ts.set(None)
    session_trace_id.set(None)
    session_trace_name.set(None)


def get_chat_id() -> Optional[str]:
    return session_chat_id.get()


def get_platform() -> Optional[str]:
    return session_platform.get()


def get_chat_name() -> Optional[str]:
    return session_chat_name.get()


def get_thread_id() -> Optional[str]:
    return session_thread_id.get()


def get_user_id() -> Optional[str]:
    return session_user_id.get()


def get_thread_ts() -> Optional[str]:
    return session_thread_ts.get()


def set_thread_ts(ts: Optional[str]) -> None:
    session_thread_ts.set(ts)


def get_trace_id() -> Optional[str]:
    return session_trace_id.get()


def set_trace_id(trace_id: Optional[str]) -> None:
    """Set the trace id directly — used by subprocess startup, which reads
    ``HERMES_TRACE_ID`` env then seeds the ContextVar without a full
    ``set_session`` (no inbound Slack context in that path)."""
    session_trace_id.set(trace_id)


def get_trace_name() -> Optional[str]:
    return session_trace_name.get()


def set_trace_name(trace_name: Optional[str]) -> None:
    session_trace_name.set(trace_name)


@contextmanager
def maintenance_run(
    *, user_id: Optional[str], trace_name: str, platform: Optional[str] = None,
    chat_id: Optional[str] = None,
):
    """Scope a background coach-process AIAgent run (not a user message) with
    its own minted trace id, the owning user_id, and a distinct trace name.

    ``platform`` defaults to None on purpose: a maintenance run is headless, it
    has no interactive origin platform. Claiming "slack" would masquerade it as
    a user turn (the very mislabel this scope exists to prevent) and mislead any
    platform-reading tool (tts/cronjob/terminal).

    Runs in a pooled executor thread (run_in_executor doesn't copy context in),
    so this SETs a fresh identity on entry and CLEARs on exit — the pooled
    thread never leaks the maintenance identity into a later run reusing it.
    Yields the minted trace id.
    """
    tid = new_trace_id()
    set_session(
        platform=platform, chat_id=chat_id, user_id=user_id,
        trace_id=tid, trace_name=trace_name,
    )
    try:
        yield tid
    finally:
        clear_session()
