"""S-0626-02 Plan B — step-0 emits structured JSON; the scheduler parses it
directly instead of running a decide LLM.

step-0 (the artemis-briefing agent session) now emits a single JSON object
{coaches_take, opener, response_window_checkin}. `_parse_step0_output` turns
that raw agent text into the package dict the render pipeline consumes, or
None on any parse failure (caller falls back to the Phase-5 voice-scan path).
"""

import pytest

from cron.scheduler import _parse_step0_output

pytestmark = pytest.mark.xdist_group("cron_scheduler")


def test_parse_step0_json_extracts_three_fields():
    raw = '{"coaches_take": "My take: X. Want A, or B?", "opener": "Morning.", "response_window_checkin": null}'
    pkg = _parse_step0_output(raw, job_id="t1")
    assert pkg["coaches_take"].startswith("My take")
    assert pkg["opener"] == "Morning."
    assert pkg["response_window_checkin"] is None


def test_parse_step0_strips_code_fence():
    raw = '```json\n{"coaches_take": "Y", "opener": null, "response_window_checkin": null}\n```'
    pkg = _parse_step0_output(raw, job_id="t2")
    assert pkg["coaches_take"] == "Y"
    assert pkg["opener"] is None


def test_parse_step0_defaults_missing_optional_keys():
    # opener / response_window_checkin absent → defaulted to None, not KeyError
    raw = '{"coaches_take": "Z"}'
    pkg = _parse_step0_output(raw, job_id="t3")
    assert pkg["coaches_take"] == "Z"
    assert pkg["opener"] is None
    assert pkg["response_window_checkin"] is None


def test_parse_step0_returns_none_on_non_json():
    assert _parse_step0_output("Morning. Your team ran 3 things.", job_id="t4") is None


def test_parse_step0_returns_none_when_missing_required_key():
    # coaches_take is the one required field
    assert _parse_step0_output('{"opener": "Hi"}', job_id="t5") is None


def test_parse_step0_returns_none_on_empty():
    assert _parse_step0_output("", job_id="t6") is None
    assert _parse_step0_output(None, job_id="t7") is None


# ---------------------------------------------------------------------------
# Reasoning-spillover salvage (Artemis P-0714-02)
#
# A degraded provider call (slow, reasoning-heavy) sometimes writes its
# analysis narration into the content channel before the final JSON — the
# whole-string json.loads fails and the day's briefing was silently replaced
# by the quiet-day fallback. Both fixtures below are VERBATIM live outputs
# (dev, 2026-07-08 / 2026-07-09); the narration may contain earlier draft
# objects, so salvage must pick the LAST candidate carrying coaches_take.
# ---------------------------------------------------------------------------

# dev 2026-07-08 01:55 UTC (trace bce16d8d3a97…): narration + one bare JSON.
SPILLOVER_SINGLE = '''The skill instructions specify no web search for roles -- only for events. Given this is a briefing at 1:55 AM local time (late night for the user in Boston), and the user has been active today with a Dealer Tire interview and Wayfair submission, and has the Brigham packet ready -- I should note that today is actually a content day for Maya. The response window for Wayfair has passed (day 9 of 7). The Brigham packet is ready but not submitted.

However, since it's a cron briefing at ~2 AM Boston time and the user just logged off ("feels pretty good and is logging off for the night" per Mem0), this briefing will be waiting for them in the morning. The key content:
1. Wayfair response window passed - offer Publicist follow-up draft
2. Brigham materials ready but not submitted - gentle nudge
3. Dealer Tire phone screen happened yesterday and Maya felt uncertain

I'll structure the coaches_take to address the Brigham submission nudge (today's one thing), while mentioning the response-window checkin for Wayfair.

{"coaches_take": "The Brigham materials are still standing by from Tuesday — fully remote clinical/operational modeling, top-ranked against your healthcare axis, and ready to click submit whenever you're ready this morning. That's the one thing worth moving today. Wayfair's response window closed without a reply — no signal yet either way.", "opener": "Keeping eyes on healthcare and mission-driven data science roles in Boston today.", "response_window_checkin": "No reply yet from Wayfair — the response window has passed. Would you like Publicist to draft a follow-up you can send?"}'''

# dev 2026-07-09 01:00 UTC (trace fb503811a460…), VERBATIM full output:
# narration (itself carrying the provider's char-drop corruption — "lands tw",
# ".esponse window checkin", "rech"), then a FENCED DRAFT object (corrupted
# "draftr"), then the final bare JSON (with its own "on yr end" drop). Naive
# first-object extraction returns the corrupted draft — last must win.
SPILLOVER_WITH_DRAFT = '''The skill instructions specify that I need to produce a JSON object with three fields: `coaches_take`, `opener`, and `response_window_checkin`. Let me analyze what I have:

1. **User profile**: Marketing analyst roles in Bangkok, recent grad from Chulalongkorn
2. **Strategy state**:
   - Today is July 9, 2026
   - Day 6 of 7 response window (outbound sent July 3)
   - Tomorrow July 10 is Day 7 — when follow-up draft offers should surface for Sarah Lin (Brighthouse) and Verdana Botanica
   - `do_not_do` explicitly says "Do NOT offer follow-up drafts to Sarah Lin or Verdana Botanica hiring team before Day 7 (July 10)"
   - So today (July 9) is NOT yet Day 7 — the follow-up offer is held
   - Emotional context is positive: user engaged July 8 asking about Sarah Lin's background
   - No new job matches (16 days of 0 Bangkok-relevant matches from resume-ranked pool)
   - Pending actions: CP Group/True/dtac career pages research, FMCG career pages research, BrightHouse passive monitor, Agoda/MONIX rech
3. **Mem0**: Mostly older memories (May-June) about decision fatigue and 10-day silence. The current `emotional_context.read` in strategy is fresher — user engaged July 8, stable capacity, controlled hold pattern.

4. **`todays_follow_ups`**: The July 10 follow-up is tomorrow, not today. Today is the controlled hold day.

5. **What to surface today**:
   - Today is Day 6 — one more day until the Day-7 follow-up draft offer milestone
   - Career-page pipeline is the only active sourcing channel (CP Group/True/dtac and FMCG multinationals pending)
   - No new outbound action warranted today per top_recommendation — "controlled one-day hold until tomorrow's Day-7 milestone"
   - This is a "holding pattern" day

6. **Opener**: Reference the goal naturally, varied between days. No job counts, no "resume" word. Keep it warm.

7. **Coach's Take**: Warm-then-action. Today is a hold day — no action, but name what's happening and what comes tomorrow. Declarative landing (no templated "want me to X?" closer).esponse window checkin**: None today — the Day-7 milestone is tomorrow (July 10), not today. Set to null.

Let me craft the JSON:

- **Opener**: Something like "Tracking the Bangkok marketing analyst pipeline on day six of the callback window." — references the goal and the timing naturally without templating.
- **Coach's Take**: Warm acknowledgment that it's a controlled hold day, name the one thing coming tomorrow (Day-7 follow-up draft offers), and land on a declarative "I'm holding the surface tight until then" or equivalent. No templated solicitation.
- **Response window checkin**: null (tomorrow triggers it, not today)

Let me refine the Coach's Take to be tight (3 sentences max), warm-then-action, and avoid:
- Repetition across beats (opener mentions the window, Take should advance to the holding-pattern + tomorrow)
- Reasoning trace ("I'm looking at...")
- Template solicitation closer
- Third-person naming of the reader

Draft:
> A controlled hold day on the board — the Day-7 milestone lands tw, at which point I'll surface the follow-up drafts for Sarah Lin and the Verdana Botanica hiring team if neither has replied. The career-page pipeline (CP Group, True, dtac, and the FMCG rotation) is still the only active sourcing channel, and I'm keeping it running quietly in the background. Nothing to do on your end today; the next move surfaces with tomorrow's briefing.

Wait — I should check for any approach deadlines. `approaching_deadlines` is not in the strategy response, so assume none. `todays_follow_ups` is not explicitly populated as a separate field here, but the July 10 item is tomorrow.

Let me tighten the opener and Coach's Take to avoid repetition. The opener names the day-six timing. The Take should advance to "controlled hold, tomorrow is Day 7 with follow-up draft offers, career-page pipeline running in background, declarative landing."

Final JSON:
```json
{
  "coaches_take": "A controlled hold day on the board — the Day-7 milestone lands tomorrow, at which point the follow-up draftr Sarah Lin and the Verdana Botanica hiring team surface if neither has replied. The career-page pipeline (CP Group, True, dtac, and the FMCG rotation) keeps ticking in the background as the only active sourcing channel. Nothing to do on your end today; the next move drops with tomorrow's briefing.",
  "opener": "Tracking the Bangkok marketing analyst pipeline on day six of the callback window.",
  "response_window_checkin": null
}
```

That's tight, warm-then-action, no reasoning trace, no template closer, no repetition across beats (opener names the day/window, Take advances to the hold-pattern + tomorrow's milestone + declarative landing).

{"coaches_take": "A controlled hold day on the board — the Day-7 milestone lands tomorrow, at which point the follow-up drafts for Sarah Lin and the Verdana Botanica hiring team surface if neither has replied. The career-page pipeline (CP Group, True, dtac, and the FMCG rotation) keeps ticking in the background as the only active sourcing channel. Nothing to do on yr end today; the next move drops with tomorrow's briefing.", "opener": "Tracking the Bangkok marketing analyst pipeline on day six of the callback window.", "response_window_checkin": null}'''


def test_salvage_narration_prefix_single_json():
    """The 07-08 live shape: narration prefix + one bare JSON at the end.
    Whole-string parse fails; salvage must recover the package, with the
    server-computed response_window_checkin preserved verbatim."""
    pkg = _parse_step0_output(SPILLOVER_SINGLE, job_id="s1")
    assert pkg is not None
    assert pkg["coaches_take"].startswith("The Brigham materials")
    assert pkg["response_window_checkin"] == (
        "No reply yet from Wayfair — the response window has passed. "
        "Would you like Publicist to draft a follow-up you can send?"
    )


def test_salvage_picks_last_candidate_not_draft():
    """The 07-09 live shape carries TWO parseable candidates: a fenced draft
    (with the provider's char-drop corruption "draftr") and the clean final
    bare JSON. Drafts precede the final answer — the LAST candidate wins."""
    pkg = _parse_step0_output(SPILLOVER_WITH_DRAFT, job_id="s2")
    assert pkg is not None
    assert "drafts for Sarah Lin" in pkg["coaches_take"]
    assert "draftr" not in pkg["coaches_take"]
    assert pkg["opener"].startswith("Tracking the Bangkok")
    assert pkg["response_window_checkin"] is None


def test_salvage_ignores_objects_without_required_key():
    raw = 'Considering {"opener": "Hi", "note": "no take here"} as an option.'
    assert _parse_step0_output(raw, job_id="s3") is None


def test_salvage_garbage_brace_still_falls_back():
    raw = "Some analysis { that never becomes an object, sadly."
    assert _parse_step0_output(raw, job_id="s4") is None


def test_salvage_silent_marker_unaffected():
    # [SILENT] is the one non-JSON contract case — must keep reaching the
    # Phase-5 path untouched, never be "salvaged".
    assert _parse_step0_output("[SILENT]", job_id="s5") is None
