"""Application-count milestone detection for Coach's positive-affirm behavior.

Artemis S-0601-03 (fork side). The gateway injects a milestone state-reminder
block into Coach's system prompt each turn so the affirm fires deterministically
rather than depending on Coach choosing to call get_strategy. Counts are derived
from strategy.json archive[] application_submitted events (typed by S-0601-02);
dedup is a persisted milestones_affirmed[] ledger marked optimistically at inject
time (over-affirm fails safe toward silence).

This module reads per-user artemis files directly (the gateway already reads
them this way) — it does not import the Artemis MCP server.

S-0622-04: the application count migrated from the strategy.json archive
``application_submitted`` event (which under-counted — cover letters riding the
``save_cover_letter`` side path never produced an archive event, SIM-JOURNEY
scene 3) to the ``applications.json`` ledger — the single source of application
state. Count records with ``status`` >= submitted. Dedup still lives on the
strategy.json ``milestones_affirmed[]`` ledger (unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Application-count milestone tiers, ascending. Lifted from the simulation
# (2 apps, 3 apps in the Maya scenes) plus a sparse continuation (5, 10).
# first_screen / first_contact tiers are deferred until phone_screen /
# contact_made signals exist (Phase 2).
_APP_TIERS = (("apps_2", 2), ("apps_3", 3), ("apps_5", 5), ("apps_10", 10))

# Application statuses that count as "an application went out" — submitted and
# everything downstream of it (interviewed). Active pre-submit states
# (identified / materials_ready) and terminal outcome (carried separately on the
# record) are excluded. Terminal state lives in `outcome`, not `status`, so a
# rejected-after-submit application still has status="submitted" and counts.
_SUBMITTED_STATUSES = frozenset({"submitted", "interviewed"})


def count_submitted_applications(user_dir: Path) -> int:
    """Count applications with status >= submitted in ``<user_dir>/applications.json``.

    The single source of the application milestone count (S-0622-04). Returns 0
    on missing / corrupt ledger (fail-safe: never raises — callers run inside the
    gateway turn / cron job).
    """
    path = Path(user_dir) / "applications.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if isinstance(raw, dict):
        records = raw.get("applications")
    else:
        records = raw
    if not isinstance(records, list):
        return 0
    return sum(
        1 for r in records
        if isinstance(r, dict) and r.get("status") in _SUBMITTED_STATUSES
    )


def detect_milestone(user_dir: Path) -> dict[str, Any] | None:
    """Return the highest un-affirmed application-count milestone, or None.

    Counts submitted applications from ``<user_dir>/applications.json`` (S-0622-04;
    the count source migrated off the archive event), reads the dedup ledger from
    ``strategy.json`` ``milestones_affirmed[]``, picks the highest tier whose
    threshold is met and not yet affirmed. Returns None on missing / corrupt
    strategy, or when nothing new is crossed (fail-safe: a read error never raises
    here).
    """
    strategy_path = Path(user_dir) / "strategy.json"
    try:
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(strategy, dict):
        return None

    app_count = count_submitted_applications(user_dir)
    affirmed = set(strategy.get("milestones_affirmed") or [])

    # An affirmed tier covers every LOWER tier — they were implicitly crossed at
    # the same time, so a lower tier must never re-fire once a higher one is
    # affirmed (the dev 2026-06-09 double-inject was apps_2 re-firing after only
    # apps_3 had been marked). Floor candidate thresholds at the highest affirmed.
    affirmed_thresholds = [t for tier, t in _APP_TIERS if tier in affirmed]
    floor = max(affirmed_thresholds) if affirmed_thresholds else 0

    chosen = None
    for tier, threshold in _APP_TIERS:
        if app_count >= threshold and tier not in affirmed and threshold > floor:
            chosen = tier
    if chosen is None:
        return None
    return {
        "tier": chosen,
        "kind": "application_count",
        "count": app_count,
        "label": f"{app_count} applications submitted",
    }


def render_milestone_block(milestone: dict[str, Any] | None) -> str:
    """Render the system-prompt injection block for a detected milestone.

    Returns "" when milestone is None. The block tells Coach to voice ONE
    grounded, user-crediting sentence naming the count — no emoji, no hype, no
    action push — additive to the turn's base shape, ahead of any bridge.
    """
    if not milestone:
        return ""
    count = milestone.get("count")
    return (
        "\n**Positive milestone reached this turn.** The user has crossed a real "
        f"milestone: {count} job applications submitted so far. Voice exactly ONE "
        "grounded sentence that credits the user for it — name the number, make it "
        "about what they did, keep it plain. No emoji, no exclamation-pile, no "
        "streak language, no action push riding on it. It is additive to your "
        "normal reply; if a forward-pivot or A/B is also due this turn, the "
        "milestone sentence comes first, then the pivot — never merged. If the "
        "user's turn carries hard affect, read the feeling first (the Emotional "
        "Posture rules) — the milestone sentence stays one grounded line either way."
    )


# Completion-report signal words. The affirm only fires on a turn where the user
# is reporting that a milestone just landed (they submitted / sent an application).
# On a generic turn ("what's next") we neither inject nor mark, so the tier waits
# for the turn where crediting it is natural — instead of being burned silently.
# Deterministic word-list, not an LLM: routing is a closed classification, kept
# off the prompt-compliance path (the affirm voicing is the only LLM-judged part).
_COMPLETION_SIGNALS = (
    "submitted", "submit", "sent it", "sent the", "sent off", "just sent",
    "applied", "application in", "fired off", "shipped it", "put it in",
    "got it in", "out the door", "hit submit",
)


def user_reported_completion(text: str | None) -> bool:
    """True when the user's turn reads as reporting a just-completed application.

    Deterministic substring match against ``_COMPLETION_SIGNALS`` (case-insensitive).
    Gates the milestone affirm injection: only a completion-report turn injects +
    marks, so a generic turn never burns an un-voiced tier. Conservative — an
    ambiguous turn returns False (the affirm waits for a clearer report), matching
    the "over-affirm fails safe toward silence" stance.
    """
    if not text or not isinstance(text, str):
        return False
    low = text.lower()
    return any(sig in low for sig in _COMPLETION_SIGNALS)


def mark_milestone_affirmed(user_dir: Path, tier: str) -> None:
    """Append ``tier`` to the persisted ``milestones_affirmed[]`` ledger.

    Dedup mark — written on a completion-report turn (the gateway gates the whole
    inject+mark on ``user_reported_completion``), so the tier is consumed only when
    the affirm is actually due, not on an unrelated generic turn. Idempotent.
    Best-effort: a write/parse failure is swallowed (a missed mark costs at most one
    re-affirm, far less bad than raising inside the gateway turn). Writes the full
    strategy back, so it never truncates the archive.
    """
    strategy_path = Path(user_dir) / "strategy.json"
    try:
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        if not isinstance(strategy, dict):
            return
        affirmed = list(strategy.get("milestones_affirmed") or [])
        if tier in affirmed:
            return
        affirmed.append(tier)
        strategy["milestones_affirmed"] = affirmed
        strategy_path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        return
