"""Session-ContextVar teardown ordering (S-0629-01 turn-teardown gap).

The gateway used to clear the session ContextVars (incl. ``session_trace_id``)
in the agent handler's own ``finally`` — but the message task isn't over
there: ``_process_message_background`` still runs the post-processing tail
(``on_processing_complete`` → ack-emoji, auto-title). Everything in that tail
ran trace-less: post-turn log lines lost their ``[trace]`` prefix and the
aux-call generations couldn't join their turn's Langfuse trace (P-0706-01).

Contract locked here: the ContextVars survive until the true end of the
message task (visible during the post-processing hooks), and are cleared when
the task finishes — on the background path AND the inline command-bypass path.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from tools.session_context import clear_session, get_trace_id, get_user_id, set_session, set_trace_id


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


def _make_adapter(handler):
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter._message_handler = handler
    return adapter


def _make_event(text, chat_id="12345"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _seeding_handler(reply="ok"):
    """Mimic the gateway agent handler: seeds the session ContextVars."""

    async def _handler(event):
        # Mirror the gateway: mint the trace id, then _set_session_env
        # carries it into set_session (set_session overwrites ALL vars —
        # omitting trace_id here would wipe it).
        set_trace_id("ht-teardown-1")
        set_session(
            platform="telegram",
            chat_id="12345",
            user_id="U-TEARDOWN",
            trace_id="ht-teardown-1",
        )
        return reply

    return _handler


@pytest.fixture(autouse=True)
def _clean_ctx():
    clear_session()
    yield
    clear_session()


class TestBackgroundTaskTeardown:
    @pytest.mark.asyncio
    async def test_ctx_visible_during_processing_complete_hook(self):
        """The post-processing tail must still see the turn's identity —
        ack-emoji / auto-title fire here and need the trace id."""
        seen = {}
        adapter = _make_adapter(_seeding_handler())

        original_hook = adapter._run_processing_hook

        async def spy_hook(name, *args, **kwargs):
            if name == "on_processing_complete":
                seen["trace_id"] = get_trace_id()
                seen["user_id"] = get_user_id()
            return await original_hook(name, *args, **kwargs)

        adapter._run_processing_hook = spy_hook
        event = _make_event("hello")
        sk = build_session_key(event.source)

        await adapter._process_message_background(event, sk)

        assert seen["trace_id"] == "ht-teardown-1"
        assert seen["user_id"] == "U-TEARDOWN"

    @pytest.mark.asyncio
    async def test_ctx_cleared_at_task_end(self):
        """The isolation contract: nothing leaks past the message task."""
        adapter = _make_adapter(_seeding_handler())
        event = _make_event("hello")
        sk = build_session_key(event.source)

        await adapter._process_message_background(event, sk)

        assert get_trace_id() is None
        assert get_user_id() is None


class TestCommandBypassTeardown:
    @pytest.mark.asyncio
    async def test_inline_command_path_clears_ctx(self):
        """/status-style commands bypass _process_message_background and
        dispatch inline — that path needs its own clear."""
        adapter = _make_adapter(_seeding_handler(reply="status: fine"))
        event = _make_event("/status")
        sk = build_session_key(event.source)
        # Simulate an active session so the command takes the bypass branch.
        import asyncio as _asyncio

        adapter._active_sessions[sk] = _asyncio.Event()

        await adapter.handle_message(event)

        assert get_trace_id() is None
        assert get_user_id() is None
