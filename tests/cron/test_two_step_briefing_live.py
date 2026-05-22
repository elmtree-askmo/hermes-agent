"""Live integration tests for Phase 6 two-step briefing — require real OpenRouter key.

These tests call the real API with verbatim prod fixtures from B-0510-01:
- Amy 5/21 raw output (voice FAIL + structure FAIL under Phase 5)
- Garwin 5/22 raw output (structure FAIL under Phase 5)
- James 5/12 A-class total replacement

Run: OPENROUTER_API_KEY=<key> ./venv/bin/python -m pytest tests/cron/test_two_step_briefing_live.py -v -s

All tests are skipped when OPENROUTER_API_KEY is unset.
"""
import os
import re
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — live tests skipped"
)

from cron.scheduler import (
    _briefing_decide_call,
    _briefing_write_call,
    _run_two_step_briefing,
)

# ---------------------------------------------------------------------------
# Verbatim prod fixtures
# ---------------------------------------------------------------------------

AMY_20260521_RAW = """Nothing is going to come through the job search for a founder building their own agency — and no resume on file anyway. The strategy is clear: waiting on Amy's reply to the May 20 check-in, 48-hour window through May 22. It's day 1 of that window. Emotional context is founder delivery overload. No action needed today.

This is a quiet-day / low-action briefing — the only real content is the response window status.

Day 1 of your 48-hour wait on Amy's reply. Nothing to push today.

\U0001F4CC Follow-ups
───────────
⏰ 5/22    48-hour response window closes
           Amy's May 20 check-in — if she responds, activate intake flow
⏰ 5/27    Evaluation checkpoint
           No reply → pause active engagement, transition to monthly touchpoints
\U0001F504 ongoing 7 artifacts pre-built and ready when Amy has bandwidth

\U0001F4AC **Coach's Take:** Day 1 of the wait — silence from Amy reads as founder delivery overload, not disinterest. You've done the move (low-pressure signal on 5/20), now patience is the strategy. I'll keep monitoring; nothing for you to do until the window closes on 5/22."""

GARWIN_20260522_RAW = """Day 17 — pipeline monitoring continues, Day 21 decision lands May 26.

\U0001F4CC Follow-ups
───────────
⭐ May 26   Day 21 pipeline checkpoint
            Binary: any response from 17 firms, or silence (auto-pivot to direct CEO outreach)
\U0001F504 ongoing  Passive monitoring — 17 firms tracked · likely responders: Software Equity Group, Houlihan Lokey, Solomon Partners

\U0001F4AC **Coach's Take:** Four days until the Day 21 decision point on May 26 — the one binary is: any response from the 17 firms, or zero confirmed silence (auto-defaults to direct CEO outreach at New Oriental, TAL Education, NetDragon). The drafts are already sitting in your inbox waiting. Until then, tracking stays quiet on your end. I'll keep scanning in the background."""

JAMES_20260512_A_CLASS = """Here is the situation:

- User is 11+ days post-graduation (June 2026, now May 12 2026 — actually user hasn't graduated yet!)
- Status: no_resume — no job matches available.
- Let me check the do_not_do list: ...
- Wait — the strategy is stale ...
- Actually, let me reconsider. ...
- Let me write a quiet-day note."""

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

_REASONING_LEAK_PATTERNS = [
    r"(?i)\blet me\b",
    r"(?i)\bhere is the situation\b",
    r"(?i)\bthe strategy is\b",
    # Match reasoning self-correction "Wait —" / "Wait—" only at sentence start
    # (not "the wait—" which is valid prose)
    r"(?m)^\s*wait\s*[—–-]",
    r"(?i)\bactually,? let me\b",
    r"(?i)\bemotional context is\b",
    r"(?i)\bno action needed today\b",
]

_THIRD_PERSON_PATTERNS = [
    r"(?i)\bif (she|he|they) respond",
    r"(?i)\bwhen (she|he|they) ha",
    r"(?i)\b(amy|garwin|james|crystal|catherine|vishal)\b(?!'s)",
]


def _assert_clean_output(text: str, label: str):
    for pat in _REASONING_LEAK_PATTERNS:
        assert not re.search(pat, text), (
            f"[{label}] reasoning leak pattern {pat!r} found in:\n{text[:400]}"
        )
    for pat in _THIRD_PERSON_PATTERNS:
        assert not re.search(pat, text), (
            f"[{label}] third-person pattern {pat!r} found in:\n{text[:400]}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_live_decide_amy_extracts_follow_ups():
    """decide call on Amy 5/21 raw must extract >=1 follow_up and have a non-empty coaches_take.

    Note: coaches_take in the decide pkg is a distilled extraction of the raw input — the model
    may legitimately echo names from the source text here. The name-clean constraint applies only
    to the final rendered output (tested in test_live_full_two_step_amy).
    """
    pkg = _briefing_decide_call(AMY_20260521_RAW, "live-amy-decide")
    assert pkg is not None, "decide call returned None"
    assert pkg.get("briefing_type") in ("quiet_day", "content")
    assert len(pkg.get("follow_ups", [])) >= 1, f"expected >=1 follow_ups, got {pkg.get('follow_ups')}"
    coaches_take = pkg.get("coaches_take", "")
    assert coaches_take, "coaches_take must be non-empty"
    print(f"\n[live-amy-decide] pkg={pkg}")


def test_live_decide_garwin_extracts_pipeline_context():
    """decide call on Garwin 5/22 raw must extract >=1 follow_up."""
    pkg = _briefing_decide_call(GARWIN_20260522_RAW, "live-garwin-decide")
    assert pkg is not None, "decide call returned None"
    assert len(pkg.get("follow_ups", [])) >= 1, f"expected >=1 follow_ups, got {pkg.get('follow_ups')}"
    print(f"\n[live-garwin-decide] pkg={pkg}")


def test_live_decide_james_a_class_produces_quiet_day():
    """decide call on James A-class (no deliverable) must classify as quiet_day."""
    pkg = _briefing_decide_call(JAMES_20260512_A_CLASS, "live-james-decide")
    assert pkg is not None, "decide call returned None"
    assert pkg.get("briefing_type") == "quiet_day", (
        f"expected quiet_day for A-class input, got {pkg.get('briefing_type')}"
    )
    print(f"\n[live-james-decide] pkg={pkg}")


def test_live_full_two_step_amy():
    """Full two-step on Amy 5/21 raw: output must be clean (no reasoning leak, no third-person)."""
    result = _run_two_step_briefing(AMY_20260521_RAW, "live-amy-full")
    assert result is not None, "two-step returned None"
    _assert_clean_output(result, "live-amy-full")
    print(f"\n[live-amy-full] output:\n{result}")


def test_live_full_two_step_garwin():
    """Full two-step on Garwin 5/22 raw: output must be clean."""
    result = _run_two_step_briefing(GARWIN_20260522_RAW, "live-garwin-full")
    assert result is not None, "two-step returned None"
    _assert_clean_output(result, "live-garwin-full")
    print(f"\n[live-garwin-full] output:\n{result}")


def test_live_full_two_step_james_a_class():
    """Full two-step on James A-class: must be non-empty and clean."""
    result = _run_two_step_briefing(JAMES_20260512_A_CLASS, "live-james-full")
    assert result is not None, "two-step returned None"
    assert len(result.strip()) > 10
    _assert_clean_output(result, "live-james-full")
    print(f"\n[live-james-full] output:\n{result}")
