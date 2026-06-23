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
