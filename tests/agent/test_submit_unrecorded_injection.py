"""Submit-unrecorded injection — Artemis B-0715-02 (fork side).

When a user reports a submission with an abbreviated / acronym employer name
("just submitted the BVARI application"), the strict segment-bounded
``_match_company`` in ``detect_submit`` returns None (the record's canonical key
``boston-va-research-institute-inc-bvari`` needs every distinctive segment, and
"BVARI" supplies only one), so the ledger is NOT advanced. But the post-submission
bridge does no text match — it fires on any ``materials_ready`` record — so Coach's
turn context carries a "next role" signal with no signal that the submit itself was
never recorded, and Coach confirms a submit the ledger never made.

c-arch fix: on the exact divergence condition — ``user_reported_completion`` AND
``detect_submit`` returns None AND the bridge fired (a ``materials_ready`` record
exists) — the gateway injects a deterministic ``submit_unrecorded`` block telling
Coach NOT to confirm the submit and to ask which role / name the employer so it can
be logged. Matcher stays strict; bridge keeps its tolerance; the only change is the
negative signal Coach was missing now reaches it.

The behavioral half (Coach's wording) is a SIM spot-check; these tests pin the
deterministic half — the detect + render pure functions.
"""

import json

from agent.milestone_detector import detect_submit, user_reported_completion
from agent.post_submission_bridge import (
    detect_submit_unrecorded,
    render_submit_unrecorded_block,
)

# canonical key = boston-va-research-institute-inc-bvari (five distinctive segments);
# display_name deliberately does NOT carry "(BVARI)" — the acronym lives only in the
# canonical key, so "BVARI" matches no whole segment of either candidate string. A
# second materials_ready record is present because the strict no-match only survives
# under MULTIPLE candidates: with a single candidate the submit-class single-candidate
# fallback would guess it (advancing the ledger, no divergence). The real 2026-07-15
# walk had BWH + BVARI both staged — this fixture reproduces that.
_BVARI_KEY = "boston-va-research-institute-inc-bvari"
_BVARI_DISPLAY = "Boston VA Research Institute"
_BWH_KEY = "brigham-and-womens-hospital"
_BWH_DISPLAY = "Brigham and Womens Hospital"


def _setup(tmp_path, user_id="U123", apps=None):
    user_dir = tmp_path / "artemis" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "applications.json").write_text(
        json.dumps({"applications": apps or []}), encoding="utf-8")
    return user_dir


def _apprec(company, display, status, submitted_at=None, outcome=None):
    return {"company": company, "display_name": display, "status": status,
            "artifacts": [{"kind": "resume"}], "outcome": outcome,
            "submitted_at": submitted_at}


def _two_materials_ready():
    # BVARI + a second staged role -> multi-candidate, so the abbreviation genuinely
    # fails to advance (single-candidate fallback does not rescue it).
    return [
        _apprec(_BVARI_KEY, _BVARI_DISPLAY, "materials_ready"),
        _apprec(_BWH_KEY, _BWH_DISPLAY, "materials_ready"),
    ]


class TestDivergencePremise:
    """Pin the premise the fix rests on: abbreviation misses, full name matches."""

    def test_abbrev_submit_does_not_match(self, tmp_path):
        ud = _setup(tmp_path, apps=_two_materials_ready())
        # "BVARI" is one distinctive segment of a five-segment key, and with >1
        # candidate the single-candidate fallback does not fire -> no advance.
        assert detect_submit("just submitted the BVARI application this morning", ud) is None

    def test_full_name_submit_matches(self, tmp_path):
        ud = _setup(tmp_path, apps=_two_materials_ready())
        got = detect_submit("just submitted the Boston VA Research Institute application", ud)
        assert got is not None
        assert got["company"] == _BVARI_KEY


class TestDetect:
    def test_abbrev_submit_with_materials_ready_triggers(self, tmp_path):
        ud = _setup(tmp_path, apps=_two_materials_ready())
        text = "just submitted the BVARI application this morning"
        # premise holds: report, no match, bridge would fire
        assert user_reported_completion(text)
        assert detect_submit(text, ud) is None
        r = detect_submit_unrecorded(text, ud)
        assert r is not None
        # bridge picks the first materials_ready record as the next role
        assert r["company_label"] == _BVARI_DISPLAY

    def test_matched_full_name_submit_does_not_trigger(self, tmp_path):
        # detect_submit matches -> ledger advances -> no divergence -> no injection.
        ud = _setup(tmp_path, apps=_two_materials_ready())
        text = "just submitted the Boston VA Research Institute application"
        assert detect_submit_unrecorded(text, ud) is None

    def test_no_materials_ready_does_not_trigger(self, tmp_path):
        # No bridge would fire (nothing materials_ready) -> Coach gets no next-role
        # signal to false-confirm against -> no injection needed.
        ud = _setup(tmp_path, apps=[_apprec(_BVARI_KEY, _BVARI_DISPLAY, "submitted",
                    submitted_at="2026-07-15")])
        text = "just submitted the BVARI application this morning"
        assert detect_submit_unrecorded(text, ud) is None

    def test_non_completion_report_does_not_trigger(self, tmp_path):
        # Not a submit report at all -> no gate.
        ud = _setup(tmp_path, apps=_two_materials_ready())
        assert detect_submit_unrecorded("what should I work on next?", ud) is None

    def test_missing_ledger_returns_none(self, tmp_path):
        user_dir = tmp_path / "artemis" / "U404"
        user_dir.mkdir(parents=True, exist_ok=True)
        assert detect_submit_unrecorded("just submitted the BVARI application", user_dir) is None


class TestRender:
    def test_none_renders_empty(self):
        assert render_submit_unrecorded_block(None) == ""

    def test_block_names_role_and_forbids_confirming(self, tmp_path):
        ud = _setup(tmp_path, apps=_two_materials_ready())
        r = detect_submit_unrecorded("just submitted the BVARI application", ud)
        block = render_submit_unrecorded_block(r)
        assert block
        assert _BVARI_DISPLAY in block
        low = block.lower()
        # must tell Coach NOT to confirm the submit, and to ask
        assert "not" in low and "confirm" in low
        assert "ask" in low
