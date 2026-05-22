"""Tests for B-0510-01 Phase 6 — two-step briefing call.

_briefing_decide_call: receives raw Coach output (may contain reasoning),
returns structured JSON decision package.
_briefing_write_call: receives decision package, returns clean Slack text.
_run_two_step_briefing: orchestrates both calls; falls back to Phase 5 path
on any failure.
"""
import json
import pytest
from cron.scheduler import _briefing_decide_call


# ---------------------------------------------------------------------------
# Prod fixtures — verbatim raw Coach outputs from prod session jsonl.
# ---------------------------------------------------------------------------

AMY_20260521_RAW = """Nothing is going to come through the job search for a founder building their own agency — and no resume on file anyway. The strategy is clear: waiting on Amy's reply to the May 20 check-in, 48-hour window through May 22. It's day 1 of that window. Emotional context is founder delivery overload. No action needed today.

This is a quiet-day / low-action briefing — the only real content is the response window status.

Day 1 of your 48-hour wait on Amy's reply. Nothing to push today.

\U0001f4cc Follow-ups
───────────
⏰ 5/22    48-hour response window closes
           Amy's May 20 check-in — if she responds, activate intake flow
⏰ 5/27    Evaluation checkpoint
           No reply → pause active engagement, transition to monthly touchpoints
\U0001f504 ongoing 7 artifacts pre-built and ready when Amy has bandwidth

\U0001f4ac **Coach's Take:** Day 1 of the wait — silence from Amy reads as founder delivery overload, not disinterest. You've done the move (low-pressure signal on 5/20), now patience is the strategy. I'll keep monitoring; nothing for you to do until the window closes on 5/22."""

GARWIN_20260522_RAW = """Day 17 — pipeline monitoring continues, Day 21 decision lands May 26.

\U0001f4cc Follow-ups
───────────
⭐ May 26   Day 21 pipeline checkpoint
            Binary: any response from 17 firms, or silence (auto-pivot to direct CEO outreach)
\U0001f504 ongoing  Passive monitoring — 17 firms tracked · likely responders: Software Equity Group, Houlihan Lokey, Solomon Partners

\U0001f4ac **Coach's Take:** Four days until the Day 21 decision point on May 26 — the one binary is: any response from the 17 firms, or zero confirmed silence (auto-defaults to direct CEO outreach at New Oriental, TAL Education, NetDragon). The drafts are already sitting in your inbox waiting. Until then, tracking stays quiet on your end. I'll keep scanning in the background."""

JAMES_20260512_A_CLASS = """Here is the situation:

- User is 11+ days post-graduation (June 2026, now May 12 2026 — actually user hasn't graduated yet! Wait - the profile says "graduating June 2026" and today is May 12, 2026. So the user hasn't graduated yet, they have about a month to go. The emotional context says "11 days post-graduation" which seems wrong — that might have been written assuming a later run date).
- Status: no_resume — no job matches available.
- Emotional context: heavy stress, zero engagement, avoidance behavior.
- Today is May 12, 2026. The follow-ups are June 12, June 15, June 18 — all in the future.

Let me check the do_not_do list: ...

Wait — the strategy is stale (updated May 8, and the staleness threshold is 48 hours). ...

Actually, let me reconsider. The user is still a student graduating in June. ...

Let me write a quiet-day note."""


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False


def _fake_urlopen_decide(pkg):
    def fake(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(pkg)}}]})
    return fake


def _fake_urlopen_error(exc):
    def fake(req, timeout=None):
        raise exc
    return fake


# ---------------------------------------------------------------------------
# Tests for _briefing_decide_call
# ---------------------------------------------------------------------------

def test_decide_call_extracts_follow_ups_from_amy_raw(monkeypatch):
    """decide call must extract follow_ups from Amy's raw output."""
    pkg = {
        "briefing_type": "quiet_day",
        "follow_ups": ["5/22 48-hour response window closes", "5/27 evaluation checkpoint"],
        "coaches_take": "Silence reads as founder delivery overload, not disinterest. Patience is the strategy.",
        "tone_signal": "low_pressure",
    }
    import urllib.request
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_decide(pkg))
    result = _briefing_decide_call(AMY_20260521_RAW, "test-job-amy")
    assert result is not None
    assert result["briefing_type"] == "quiet_day"
    assert len(result["follow_ups"]) >= 1
    assert "coaches_take" in result
    assert result["coaches_take"]


def test_decide_call_extracts_follow_ups_from_garwin_raw(monkeypatch):
    """decide call must extract follow_ups from Garwin's raw output."""
    pkg = {
        "briefing_type": "quiet_day",
        "follow_ups": ["May 26 Day 21 pipeline checkpoint"],
        "coaches_take": "Four days until the Day 21 decision. Direct CEO outreach drafts are ready.",
        "tone_signal": "neutral",
    }
    import urllib.request
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_decide(pkg))
    result = _briefing_decide_call(GARWIN_20260522_RAW, "test-job-garwin")
    assert result is not None
    assert len(result["follow_ups"]) >= 1


def test_decide_call_returns_none_on_http_error(monkeypatch):
    """decide call must return None on network failure."""
    import urllib.request, urllib.error
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    monkeypatch.setattr(urllib.request, "urlopen",
                        _fake_urlopen_error(urllib.error.URLError("timeout")))
    result = _briefing_decide_call(AMY_20260521_RAW, "test-job-err")
    assert result is None


def test_decide_call_returns_none_on_non_json(monkeypatch):
    """decide call must return None when model output is not valid JSON."""
    import urllib.request
    def fake(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": "Sorry, I can't help."}}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = _briefing_decide_call(JAMES_20260512_A_CLASS, "test-job-nonjson")
    assert result is None
