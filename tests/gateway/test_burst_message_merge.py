"""Regression (B-0623-03): rapid burst follow-ups must not be silently dropped.

When a turn is in flight, the base adapter queues follow-up messages in
``_pending_messages[session_key]``.  The pre-fix code stored a single event
per session (``_pending_messages[sk] = event``), so a second follow-up
*overwrote* the first — the middle message of a burst was lost with no error.

Observed 2026-06-23: a user firing three quick texts ("hi" / "what's up
today?" / "any good jobs available today?") got a reply to only one; the rest
vanished (one silently dropped at this overwrite, the first-response delivery
separately broken by a ``name 'event' is not defined`` NameError in
``run.py`` ``_run_agent``).  These tests assert burst text is MERGED into the
pending slot so no message is dropped.

Artemis fork bug — see docs/plans/investigations/hermes-fork-drops-burst-messages.md.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    """Concrete adapter with abstract methods stubbed out."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)

    async def _mock_handler(event):
        return f"handled:{event.text}"

    adapter._message_handler = _mock_handler
    return adapter


def _make_event(text, chat_id="12345"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _session_key(chat_id="12345"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    return build_session_key(source)


class TestBurstMessageMerge:
    @pytest.mark.asyncio
    async def test_two_text_followups_merge_not_overwrite(self):
        """Two follow-ups during an active turn must both survive in pending."""
        adapter = _make_adapter()
        sk = _session_key()
        # Simulate a turn already in flight for this session.
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("first follow-up"))
        await adapter.handle_message(_make_event("second follow-up"))

        pending = adapter._pending_messages.get(sk)
        assert pending is not None, "follow-up was not queued at all"
        # Pre-fix: pending.text == 'second follow-up' (first silently overwritten).
        assert "first follow-up" in pending.text, "first burst message was dropped"
        assert "second follow-up" in pending.text

    @pytest.mark.asyncio
    async def test_three_message_burst_preserves_middle(self):
        """The real-world repro: 3 quick texts; the middle one must not vanish."""
        adapter = _make_adapter()
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event("hi"))
        await adapter.handle_message(_make_event("whats up today"))
        await adapter.handle_message(_make_event("any good jobs available today"))

        pending = adapter._pending_messages.get(sk)
        assert pending is not None
        for fragment in ("hi", "whats up today", "any good jobs available today"):
            assert fragment in pending.text, f"burst fragment dropped: {fragment!r}"


def _make_event_id(text, message_id, chat_id="12345"):
    # Same source shape as _session_key() so handle_message's computed key matches.
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, message_id=message_id
    )


class TestQueueModeDebounce:
    """Tier B (B-0623-03 follow-on): BUSY_TEXT_MODE=queue debounces a burst into
    one ordered pending turn, without interrupting the running turn, and anchors
    the reply to the latest message (message_id bumped to the last fragment)."""

    @pytest.mark.asyncio
    async def test_queue_mode_coalesces_burst_in_order(self):
        adapter = _make_adapter()
        adapter._busy_text_mode = "queue"
        adapter._busy_text_debounce_seconds = 0.05
        adapter._busy_text_hard_cap_seconds = 0.3
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()  # a turn is in flight

        await adapter.handle_message(_make_event_id("msg one", "1.0"))
        await adapter.handle_message(_make_event_id("msg two", "2.0"))
        await adapter.handle_message(_make_event_id("msg three", "3.0"))

        # Buffered in the debounce window: not yet in pending, and the running
        # turn is NOT interrupted (the Tier A failure mode).
        assert sk not in adapter._pending_messages, "queue mode must not merge into pending immediately"
        assert not adapter._active_sessions[sk].is_set(), "queue mode must not interrupt the running turn"

        await asyncio.sleep(0.25)  # let the debounce window elapse + flush

        pending = adapter._pending_messages.get(sk)
        assert pending is not None, "burst never flushed to pending"
        assert pending.text == "msg one\nmsg two\nmsg three", f"order/merge wrong: {pending.text!r}"
        assert str(pending.message_id) == "3.0", "message_id not bumped to latest (reply would thread under wrong msg)"
        assert not adapter._active_sessions[sk].is_set(), "queue mode must not interrupt the running turn"

    @pytest.mark.asyncio
    async def test_queue_mode_reorders_scrambled_arrival_by_ts(self):
        """Burst A repro: the async pre-handle awaits in _handle_slack_message can
        deliver a later-sent message BEFORE an earlier-sent one. Flush must
        re-sort by Slack ts (message_id) so the merged text is in send order and
        the reply anchors to the truly last-SENT message, not the last-arrived."""
        adapter = _make_adapter()
        adapter._busy_text_mode = "queue"
        adapter._busy_text_debounce_seconds = 0.05
        adapter._busy_text_hard_cap_seconds = 0.3
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()  # a turn is in flight

        # SCRAMBLED arrival: msg3 (ts 3.0) reaches handle_message before msg2
        # (ts 2.0) because msg2's pre-handle network awaits were slower.
        await adapter.handle_message(_make_event_id("third sent", "3.0"))
        await adapter.handle_message(_make_event_id("second sent", "2.0"))

        await asyncio.sleep(0.25)  # let the window elapse + flush

        pending = adapter._pending_messages.get(sk)
        assert pending is not None, "burst never flushed to pending"
        assert pending.text == "second sent\nthird sent", (
            f"flush did not re-sort by ts: {pending.text!r}"
        )
        assert str(pending.message_id) == "3.0", (
            "anchor must be the latest-SENT message (max ts), not the last-arrived"
        )

    @pytest.mark.asyncio
    async def test_default_interrupt_mode_unchanged(self):
        """With the default mode, Tier A behavior is preserved (merge + interrupt)."""
        adapter = _make_adapter()  # _busy_text_mode defaults to "interrupt"
        sk = _session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_event_id("a", "1.0"))
        await adapter.handle_message(_make_event_id("b", "2.0"))

        pending = adapter._pending_messages.get(sk)
        assert pending is not None and "a" in pending.text and "b" in pending.text
        assert adapter._active_sessions[sk].is_set(), "interrupt mode must still interrupt"
