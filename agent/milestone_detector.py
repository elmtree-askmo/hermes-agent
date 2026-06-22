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
    "submitted", "submit", "sent it", "sent the", "sent off", "sent over",
    "sent in", "just sent", "applied", "application in", "fired off",
    "shipped it", "put it in", "got it in", "turned it in", "out the door",
    "hit submit", "emailed it", "emailed the", "emailed my",
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


# ===== S-0622-04 Phase 2: interview / outcome detection + advance =====
#
# Same pre-LLM deterministic gate as the milestone affirm: the gateway detects
# the user-report signal off the user's own message and writes the applications
# ledger directly, before Coach's agent loop runs (so this can't be an MCP
# tool_call — there is no loop yet). The fork reaches Artemis data by direct file
# I/O only (it never imports the Artemis MCP server), so the narrow company
# normalizer below MIRRORS Artemis tools/applications.py `_canonical_company`;
# `test_canonical_company_mirrors_artemis` locks the two in sync.

import re  # noqa: E402
import unicodedata  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

_INTERVIEW_SIGNALS = (
    "screen", "phone screen", "interview", "interviewed", "got out of",
    "first round", "second round", "spoke with", "talked to",
)
_OUTCOME_SIGNALS = {
    "rejected": (
        "said no", "passed on", "didn't get", "did not get", "not moving forward",
        "moving forward with other", "rejected", "turned me down", "no thanks",
        "went with someone else",
    ),
}
_ACTIVE_STATUSES = frozenset({"submitted", "interviewed"})


def _canonical_company(name) -> str:
    """The application record key: lowercase, ascii-folded, hyphenated.

    MIRROR of Artemis tools/applications.py `_canonical_company` — kept in sync by
    test_canonical_company_mirrors_artemis. The fork can't import the Artemis
    module, so the normalizer is duplicated here (narrow, 6 lines)."""
    if not name or not isinstance(name, str):
        return ""
    folded = unicodedata.normalize("NFKD", name)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = folded.replace("'", "").replace("’", "").replace('"', "")
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def _load_applications_raw(user_dir: Path) -> list[dict]:
    path = Path(user_dir) / "applications.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    records = raw.get("applications") if isinstance(raw, dict) else raw
    return records if isinstance(records, list) else []


def _active_applications(user_dir: Path, statuses=_ACTIVE_STATUSES) -> list[dict]:
    """Records still awaiting a result: status in `statuses` and no terminal
    outcome. These are the records a user-report can map to. `statuses` widens for
    the submit class (which maps against materials_ready records)."""
    return [
        r for r in _load_applications_raw(user_dir)
        if isinstance(r, dict)
        and r.get("status") in statuses
        and not r.get("outcome")
    ]


def _match_company(text: str, user_dir: Path, statuses=_ACTIVE_STATUSES) -> str | None:
    """Map the user's free text to a known application's canonical key.

    Canonical-folds the text and each candidate record's display_name/company, then
    looks for a hyphen-segment-bounded containment (so "target" won't match inside
    "my-target"). On no named match, falls back to the most-recent candidate record
    (spec § Known limitation — the one probabilistic edge). Returns None when there
    are no candidate applications at all. `statuses` selects the candidate set
    (submit maps against materials_ready too; interview/outcome do not)."""
    active = _active_applications(user_dir, statuses)
    if not active:
        return None
    folded_text = _canonical_company(text)
    text_segs = set(folded_text.split("-"))
    for rec in active:
        key = rec.get("company") or _canonical_company(rec.get("display_name") or "")
        for cand in (key, _canonical_company(rec.get("display_name") or "")):
            if not cand:
                continue
            cand_segs = cand.split("-")
            # all segments of the company name present as whole segments in the text
            if cand_segs and all(seg in text_segs for seg in cand_segs):
                return key
    # No named company — fall back to most-recent active (max submitted_at / updated_at).
    return max(
        active, key=lambda r: r.get("submitted_at") or r.get("updated_at") or ""
    ).get("company")


def detect_interview(text, user_dir: Path) -> dict | None:
    """Detect a user-reported interview and map it to a known application.

    Returns {"company": <canonical key>} or None. Conservative: requires both an
    interview signal word AND at least one active application to map to."""
    if not text or not isinstance(text, str):
        return None
    low = text.lower()
    if not any(sig in low for sig in _INTERVIEW_SIGNALS):
        return None
    company = _match_company(text, user_dir)
    return {"company": company} if company else None


def detect_outcome(text, user_dir: Path) -> dict | None:
    """Detect a user-reported terminal outcome and map it to a known application.

    Returns {"company": <key>, "result": <result>} or None. This round only
    `rejected` is produced (Maya scene-4's only outcome); the result value is open
    for future outcomes without a code change here."""
    if not text or not isinstance(text, str):
        return None
    low = text.lower()
    result = next(
        (res for res, sigs in _OUTCOME_SIGNALS.items() if any(s in low for s in sigs)),
        None,
    )
    if result is None:
        return None
    company = _match_company(text, user_dir)
    return {"company": company, "result": result} if company else None


def _write_applications_raw(user_dir: Path, records: list[dict]) -> None:
    payload = {
        "applications": records,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(user_dir) / "applications.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def advance_interview(user_dir: Path, company: str) -> bool:
    """Advance the matching application to `interviewed`. Returns True on a write,
    False on no-match / missing ledger. Best-effort: never raises in the gateway."""
    try:
        records = _load_applications_raw(user_dir)
        key = _canonical_company(company)
        for rec in records:
            if rec.get("company") == key:
                rec["status"] = "interviewed"
                rec["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_applications_raw(user_dir, records)
                return True
        return False
    except (OSError, ValueError):
        return False


def advance_outcome(user_dir: Path, company: str, result: str, note) -> bool:
    """Set the terminal outcome on the matching application (status unchanged —
    outcome != null IS the terminal signal). Returns True on a write."""
    try:
        records = _load_applications_raw(user_dir)
        key = _canonical_company(company)
        for rec in records:
            if rec.get("company") == key:
                now = datetime.now(timezone.utc).isoformat()
                rec["outcome"] = {"result": result, "at": now, "note": note}
                rec["updated_at"] = now
                _write_applications_raw(user_dir, records)
                return True
        return False
    except (OSError, ValueError):
        return False


def detect_submit(text, user_dir: Path) -> dict | None:
    """Detect a user-reported submit and map it to a known application.

    The submit SIGNAL reuses `user_reported_completion` (the existing submit
    word-list); the company maps against the known active records. Returns
    {"company": <key>} or None. This is spec line 134's user-report submit class:
    the materials_ready -> submitted advance is now driven by the user saying they
    sent it (not by Executor finishing the draft)."""
    if not user_reported_completion(text):
        return None
    # Submit maps against materials_ready (the drafted-but-not-sent record) as well
    # as already-active ones (a user re-reporting a submit).
    company = _match_company(
        text, user_dir, statuses=frozenset({"materials_ready"}) | _ACTIVE_STATUSES
    )
    return {"company": company} if company else None


def advance_submitted(user_dir: Path, company: str) -> bool:
    """Advance the matching application to `submitted`, stamping `submitted_at`
    (date) the first time. Idempotent: an already-submitted record keeps its
    original submitted_at. Returns True on a write, False on no-match."""
    try:
        records = _load_applications_raw(user_dir)
        key = _canonical_company(company)
        for rec in records:
            if rec.get("company") == key:
                now = datetime.now(timezone.utc)
                rec["status"] = "submitted"
                if not rec.get("submitted_at"):
                    rec["submitted_at"] = now.date().isoformat()
                rec["updated_at"] = now.isoformat()
                _write_applications_raw(user_dir, records)
                return True
        return False
    except (OSError, ValueError):
        return False
