"""Post-submission bridge detection for Coach's next-role offer.

Artemis P-0609-01 (fork side). After the user reports submitting an application,
the gateway injects a bridge state-reminder block so Coach offers the next queued
role (one pivot sentence + A/B) rather than depending on Coach choosing to call
get_strategy — a past-tense submit report is classified as a casual turn by
Coach's message calibration, which short-circuits the get_strategy read. The
trigger is gated on the same user_reported_completion signal as the milestone
affirm (agent/milestone_detector); this module only supplies the detect + render.

Detection reads strategy.json action_queue[] directly (the gateway already reads
per-user artemis files this way) — it does not import the Artemis MCP server. The
next-role item is the Strategist-staged deliver-<company>-materials role-decision
entry (agent/strategist-hermes.md § Staging the next role behind a submit). We key
on the deterministic id shape + pending status, not the LLM-authored
trigger_condition text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# A staged next-role item carries this id prefix/suffix shape and stays pending
# until its submit lands. Coach's SOUL.md § Juncture A/B/Pause bridges to it.
_DELIVER_PREFIX = "deliver-"
_DELIVER_SUFFIX = "-materials"


def _company_label(item_id: str) -> str:
    """Derive a human-readable company label from a deliver-<company>-materials id.

    ``deliver-brightline-health-materials`` -> ``Brightline Health``. Best-effort
    titleization; the LLM refines phrasing when it voices the bridge.
    """
    core = item_id
    if core.startswith(_DELIVER_PREFIX):
        core = core[len(_DELIVER_PREFIX):]
    if core.endswith(_DELIVER_SUFFIX):
        core = core[: -len(_DELIVER_SUFFIX)]
    return " ".join(part for part in core.split("-") if part).title()


def detect_next_queued_role(user_dir: Path) -> dict[str, Any] | None:
    """Return the next pending staged role to bridge to, or None.

    Reads ``<user_dir>/strategy.json`` and scans ``action_queue[]`` for the first
    pending ``deliver-<company>-materials`` entry — the Strategist-staged next role
    behind a submit. Returns ``{"id", "company_label"}`` or None when none is
    queued. Fail-safe: a missing / corrupt strategy never raises (returns None),
    so it can't break the gateway turn.
    """
    strategy_path = Path(user_dir) / "strategy.json"
    try:
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(strategy, dict):
        return None

    for item in strategy.get("action_queue") or []:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or ""
        if (
            item_id.startswith(_DELIVER_PREFIX)
            and item_id.endswith(_DELIVER_SUFFIX)
            and item.get("status") == "pending"
        ):
            return {"id": item_id, "company_label": _company_label(item_id)}
    return None


def render_post_submission_bridge_block(role: dict[str, Any] | None) -> str:
    """Render the system-prompt injection block for a next-role bridge.

    Returns "" when role is None. The block tells Coach to bridge to the named
    role with ONE forward-pivot sentence + an A/B (look now / hold it) — Rule 1
    surface-and-invite, no milestone-celebration tone, no re-gating of the submit.
    Additive to the turn; if a milestone affirm is also due, the affirm sentence
    comes first (Coach orders them per SOUL.md § Juncture A/B/Pause).
    """
    if not role:
        return ""
    label = role.get("company_label") or "the next role"
    return (
        "\n**Post-submission bridge due this turn.** The user just reported "
        f"submitting an application, and the next role — {label} — is staged and "
        "ready in their queue. After a plain ack of the submit (do NOT re-gate or "
        "re-confirm what they already sent), bridge to it with ONE forward-pivot "
        "sentence and an A/B: look at it now, or hold it for later. Keep it to that "
        "one sentence plus the choice. This is a Rule 1 surface-and-invite — "
        "offering the next role commits nothing, so don't wait for a go-ahead. No "
        "celebration tone, no streak language, no emoji; a grounded milestone "
        "affirm (if separately due this turn) leads, then this bridge follows."
    )
