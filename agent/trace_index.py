"""Per-turn trace anchor + correlation index (observability Tier 0 + Tier 1).

At each user-turn start the agent records one anchor binding the three
otherwise-disconnected identities of a run:

  trace_id    run-scoped id that already tags every log line and spans the
              Coach->Strategist->Executor chain (separate processes/profiles).
  session_id  the Hermes session = ``sessions/<session_id>.jsonl`` filename and
              the per-profile ``state.db`` primary key.
  user_id     the Slack user the run serves (``~/.hermes/artemis/<user_id>/``).

Why this exists: Coach / Strategist / Executor run in three isolated stores
(separate ``state.db`` + ``sessions/`` under different profiles, no
``parent_session_id`` link). ``trace_id`` is the only thing tying them
together, but it lives only in log lines — not bound to the session/user
persistence. This records that binding.

Two outputs:
  * Tier 0 — an INFO log line (``turn-start session=.. user=..``) which the
    ``hermes_logging`` factory prefixes with ``trace=<id>``, so the logs become
    self-correlating (a trace -> its session + user, and the reverse).
  * Tier 1 — an append to a SHARED ``trace_index.jsonl`` at the ROOT hermes
    home (so Coach plus the profile-scoped Strategist/Executor all append to
    ONE file). Each ``trace_id`` thus maps to the 1..N ``session_id`` of one
    run — the only bridge across the three isolated session stores.

Fail-open: any error here is swallowed; observability must never break a turn.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger("trace")


def _profile_name() -> str:
    """Which profile this run is under — the role indicator and the locator for
    the session file. ``~/.hermes/profiles/<name>`` -> ``<name>``
    (e.g. ``strategist``/``executor``); the gateway/Coach root home -> ``main``.

    With this + ``session_id`` the session file is at
    ``<root>/profiles/<profile>/sessions/<session_id>.jsonl`` (or, for
    ``main``, ``<root>/sessions/<session_id>.jsonl``).
    """
    home = get_hermes_home()
    return home.name if home.parent.name == "profiles" else "main"


def _trace_index_path() -> Path:
    """Shared index at the ROOT hermes home so every profile writes one file.

    A profile home is ``~/.hermes/profiles/<name>``; strip that so the
    Strategist/Executor subprocesses append to the same file as the Coach.
    """
    home = get_hermes_home()
    if home.parent.name == "profiles":
        home = home.parent.parent
    return home / "logs" / "trace_index.jsonl"


def _append(rec: dict) -> None:
    """Append one JSON record to the shared index (append-only; concurrent
    small-line appends are atomic under O_APPEND)."""
    path = _trace_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record_turn(
    session_id: str,
    *,
    platform: str | None = None,
    parent_session_id: str | None = None,
) -> None:
    """Emit the turn-start anchor (Tier 0) + index record (Tier 1).

    ``parent_session_id`` links a session that was rotated out of a *prior*
    one — context compression ends the old session and mints a new one
    mid-run (``run_agent._compress_context``), so one logical run spans
    several session_ids. Recording the parent lets a consumer nest those
    compression slices under the run's root session instead of counting them
    as separate runs. ``None`` for a fresh (uncompressed) session.
    """
    try:
        from tools.session_context import get_trace_id, get_user_id
        trace_id = get_trace_id() or os.environ.get("HERMES_TRACE_ID") or "-"
        user_id = get_user_id() or os.environ.get("HERMES_SESSION_USER_ID") or "-"
        plat = platform or "-"

        # Tier 0 — anchor log line. The hermes_logging factory prepends
        # ``trace=<trace_id>``, so this single line binds trace->session->user.
        _par = f" parent={parent_session_id}" if parent_session_id else ""
        logger.info(
            "turn-start session=%s user=%s platform=%s%s",
            session_id, user_id, plat, _par,
        )

        # Tier 1 — one record per session: the correlation map only. Run
        # completeness / end-reason is NOT duplicated here — it already lives in
        # state.db ``sessions.end_reason`` (e.g. "compression" vs a clean close),
        # located via this session_id + profile. Keeping the index single-event
        # avoids two-rows-per-turn + a start/end join.
        rec = {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "platform": plat,
            # Role + session-file locator: main | strategist | executor | ...
            "profile": _profile_name(),
            "started_at": datetime.datetime.now().astimezone().isoformat(),
        }
        if parent_session_id:
            rec["parent_session_id"] = parent_session_id
        _append(rec)
    except Exception:
        logger.debug("trace_index.record_turn failed", exc_info=True)
