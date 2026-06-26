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
