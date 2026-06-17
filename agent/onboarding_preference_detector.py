"""Non-blocking onboarding preference-sharpening injection.

Artemis S-0617-01 (fork side). For the NON-BLOCKING onboarding path (the user
arrived with a direction, so the detector returned single/multi and the team
auto-dispatched on the first turn), Coach must ask one filter-preference
question AFTER the self-intros — i.e. on the turn after onboarding_pushed.flag
was written. That flag retires the cold-start block, so this injection cannot
live there (see docs/specs/sharpening-questions.md § Amendment v3). Instead it
mirrors milestone_detector (S-0601-03): a server-side state-reminder injected
into context_prompt regardless of Coach's tool calls, gated by its own markers
so it is onboarding-only and ask-once.

State lives as two flag files under <user_dir>:
  onboarding_preference_pending.flag — dropped at onboarding-complete on the
      non-blocking path; means "a preference question is due next turn".
  onboarding_preference_asked.flag — dropped when the question is injected;
      ask-once dedup. Once present, the detector never fires again.

This module reads/writes per-user artemis files directly (like
milestone_detector) — it does not import the Artemis MCP server. All disk
errors fail safe (None / no-op), never raised inside the gateway turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_PENDING_FLAG = "onboarding_preference_pending.flag"
_ASKED_FLAG = "onboarding_preference_asked.flag"


def detect_onboarding_preference_pending(user_dir: Path) -> dict[str, Any] | None:
    """Return a pending-preference descriptor, or None.

    Fires only when the pending marker is present and the asked marker is
    absent (onboarding-only, ask-once). Fail-safe: any error returns None.
    """
    try:
        d = Path(user_dir)
        if not (d / _PENDING_FLAG).exists():
            return None
        if (d / _ASKED_FLAG).exists():
            return None
    except OSError:
        return None
    return {"kind": "preference"}


def render_onboarding_preference_block(pending: dict[str, Any] | None) -> str:
    """Render the system-prompt injection for a due preference question.

    Returns "" when pending is None. Tells Coach to ask ONE short preference
    question — the filter axes the scan needs but the résumé/conversation
    didn't supply (prestige-vs-fit, exclusion) — and not to re-ask known facts.
    The team has already dispatched, so this is additive refinement, not a
    block; it does NOT tell Coach to defer "briefing the team".
    """
    if not pending:
        return ""
    return (
        "\n**Onboarding sharpening — refine the scan's filters.** "
        "The team is already underway (the self-intros have landed). "
        "Ask ONE short preference question this turn — only a filter axis the scan "
        "needs but the résumé/conversation hasn't given you (how to weigh prestige "
        "vs fit, and any role shapes to exclude). Just one axis, then let it rest — "
        "do not stack questions and do not re-ask anything already known from the "
        "résumé (e.g. location). If they don't engage it, move on; this is a "
        "refinement, not a gate."
    )


def mark_onboarding_preference_pending(user_dir: Path) -> None:
    """Drop the pending marker (called at onboarding-complete, non-blocking path).

    Idempotent, best-effort: a write failure is swallowed (a missed marker costs
    at most one un-asked preference question, never a raised error in the turn).
    """
    try:
        d = Path(user_dir)
        d.mkdir(parents=True, exist_ok=True)
        flag = d / _PENDING_FLAG
        if not flag.exists():
            flag.write_text("1", encoding="utf-8")
    except OSError:
        return


def mark_onboarding_preference_asked(user_dir: Path) -> None:
    """Drop the asked marker (called when the preference block is injected).

    Idempotent ask-once dedup, best-effort (swallows write errors).
    """
    try:
        d = Path(user_dir)
        d.mkdir(parents=True, exist_ok=True)
        flag = d / _ASKED_FLAG
        if not flag.exists():
            flag.write_text("1", encoding="utf-8")
    except OSError:
        return
