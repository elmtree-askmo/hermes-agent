"""Onboarding direction-flag tracking.

Artemis S-0617-01 (fork side). This module tracks one thing: whether a
single/multi dispatch occurred this onboarding — the "direction present"
flag. It is read later to gate the proactive sharpening invite.

The reactive preference-injection this module once held (a server-side
state-reminder injected into context_prompt, gated by pending/asked markers)
was removed in the v3 rewrite: the non-blocking sharpening is now a
helper-posted invite + the existing hermes.md machinery, not a reactive
injection (see docs/specs/sharpening-questions.md § Amendment v3).

State lives as a flag file under <user_dir>:
  onboarding_direction_present.flag — dropped on any single/multi dispatch
      turn; marks that this onboarding arrived via the non-blocking path, so
      the briefing turn — which is classified `none` — can still detect it.

This module reads/writes per-user artemis files directly (like
milestone_detector) — it does not import the Artemis MCP server. All disk
errors fail safe (False / no-op), never raised inside the gateway turn.
"""

from __future__ import annotations

from pathlib import Path

_DIRECTION_FLAG = "onboarding_direction_present.flag"


def mark_onboarding_direction_present(user_dir: Path) -> None:
    """Mark that a single/multi dispatch occurred this onboarding (the goal
    turn). Session-level: read later at onboarding-complete to decide the
    non-blocking path, since the briefing turn itself classifies as 'none'.
    Idempotent, best-effort (swallows write errors).
    """
    try:
        d = Path(user_dir)
        d.mkdir(parents=True, exist_ok=True)
        flag = d / _DIRECTION_FLAG
        if not flag.exists():
            flag.write_text("1", encoding="utf-8")
    except OSError:
        return


def has_onboarding_direction_present(user_dir: Path) -> bool:
    """True iff a single/multi dispatch was marked this onboarding. Fail-safe:
    any error returns False.
    """
    try:
        return (Path(user_dir) / _DIRECTION_FLAG).exists()
    except OSError:
        return False
