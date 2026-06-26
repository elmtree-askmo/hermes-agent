"""S-0626-02 Plan B — briefing body assembly.

After step-0 JSON is parsed and the take/opener are name-strip-repaired by the
write LLM, the server assembles the delivered body: the name-stripped take,
then the response-window check-in appended VERBATIM as its own beat below it.
The check-in never passes through the LLM, so its server-computed wording
reaches the user byte-exact (fixes Finding 4-3).
"""

import pytest

from cron.scheduler import _assemble_briefing_body

pytestmark = pytest.mark.xdist_group("cron_scheduler")


def test_assemble_appends_checkin_verbatim_after_take():
    checkin = (
        "No reply yet from acme-corp — the response window has passed. "
        "Offer: ask whether they want Publicist to draft a follow-up they can send."
    )
    body = _assemble_briefing_body("Cleaned take here.", checkin)
    assert checkin in body                                       # verbatim, not paraphrased
    assert body.index("Cleaned take") < body.index(checkin)      # take first, checkin below
    assert "\n\n" in body                                        # checkin is its own beat


def test_assemble_take_only_when_no_checkin():
    assert _assemble_briefing_body("Just the take.", None) == "Just the take."
    assert _assemble_briefing_body("Just the take.", "") == "Just the take."


def test_assemble_strips_surrounding_whitespace():
    assert _assemble_briefing_body("  Take.  ", None) == "Take."


def test_assemble_empty_take_with_checkin_keeps_checkin():
    checkin = "No reply yet from acme-corp — the response window has passed."
    assert _assemble_briefing_body("", checkin) == checkin
