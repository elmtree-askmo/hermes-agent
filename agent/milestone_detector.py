"""Application-count milestone detection for Coach's positive-affirm behavior.

Artemis S-0601-03 (fork side). The gateway injects a milestone state-reminder
block into Coach's system prompt each turn so the affirm fires deterministically
rather than depending on Coach choosing to call get_strategy. Counts are derived
from strategy.json archive[] application_submitted events (typed by S-0601-02);
dedup is a persisted milestones_affirmed[] ledger marked optimistically at inject
time (over-affirm fails safe toward silence).

This module reads strategy.json directly (the gateway already reads per-user
artemis files this way) — it does not import the Artemis MCP server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Application-count milestone tiers, ascending. Lifted from the simulation
# (2 apps, 3 apps in the Maya scenes) plus a sparse continuation (5, 10).
# first_screen / first_contact tiers are deferred until phone_screen /
# contact_made event_types exist (S-0601-02 typed application_submitted only).
_APP_TIERS = (("apps_2", 2), ("apps_3", 3), ("apps_5", 5), ("apps_10", 10))


def detect_milestone(user_dir: Path) -> dict[str, Any] | None:
    """Return the highest un-affirmed application-count milestone, or None.

    Reads ``<user_dir>/strategy.json``. Counts ``application_submitted`` archive
    events, picks the highest tier whose threshold is met and not yet in
    ``milestones_affirmed[]``. Returns None on missing / corrupt strategy, or
    when nothing new is crossed (fail-safe: a read error never raises here).
    """
    strategy_path = Path(user_dir) / "strategy.json"
    try:
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(strategy, dict):
        return None

    archive = strategy.get("archive") or []
    app_count = sum(
        1 for a in archive
        if isinstance(a, dict) and a.get("event_type") == "application_submitted"
    )
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
