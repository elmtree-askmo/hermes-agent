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


def set_session(
    *,
    platform: Optional[str],
    chat_id: Optional[str],
    chat_name: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    session_platform.set(platform)
    session_chat_id.set(chat_id)
    session_chat_name.set(chat_name)
    session_thread_id.set(thread_id)
    session_user_id.set(user_id)


def clear_session() -> None:
    session_platform.set(None)
    session_chat_id.set(None)
    session_chat_name.set(None)
    session_thread_id.set(None)
    session_user_id.set(None)
    session_thread_ts.set(None)


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
