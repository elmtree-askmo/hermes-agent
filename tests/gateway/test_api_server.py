"""
Tests for the OpenAI-compatible API server gateway adapter.

Tests cover:
- Chat Completions endpoint (request parsing, response format)
- Responses API endpoint (request parsing, response format)
- previous_response_id chaining (store/retrieve)
- Auth (valid key, invalid key, no key configured)
- /v1/models endpoint
- /health endpoint
- System prompt extraction
- Error handling (invalid JSON, missing fields)
"""

import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _CORS_HEADERS,
    check_api_server_requirements,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_api_server_requirements() is True

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3

    def test_access_refreshes_lru(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Access resp_1 to move it to end
        store.get("resp_1")
        # Now resp_2 is the oldest — adding a 4th should evict resp_2
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_2") is None
        assert store.get("resp_1") is not None

    def test_update_existing_key(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "v1"})
        store.put("resp_1", {"output": "v2"})
        assert store.get("resp_1") == {"output": "v2"}
        assert len(store) == 1

    def test_delete_existing(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None
        assert len(store) == 0

    def test_delete_missing(self):
        store = ResponseStore(max_size=10)
        assert store.delete("resp_missing") is False


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_default_config(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8642
        assert adapter._api_key == ""
        assert adapter.platform == Platform.API_SERVER

    def test_custom_config_from_extra(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "host": "0.0.0.0",
                "port": 9999,
                "key": "sk-test",
                "cors_origins": ["http://localhost:3000"],
            },
        )
        adapter = APIServerAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._api_key == "sk-test"
        assert adapter._cors_origins == ("http://localhost:3000",)

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_HOST", "10.0.0.1")
        monkeypatch.setenv("API_SERVER_PORT", "7777")
        monkeypatch.setenv("API_SERVER_KEY", "sk-env")
        monkeypatch.setenv("API_SERVER_CORS_ORIGINS", "http://localhost:3000, http://127.0.0.1:3000")
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "10.0.0.1"
        assert adapter._port == 7777
        assert adapter._api_key == "sk-env"
        assert adapter._cors_origins == (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        )


# ---------------------------------------------------------------------------
# Auth checking
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_configured_allows_all(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert adapter._check_auth(mock_request) is None

    def test_valid_key_passes(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer sk-test123"}
        assert adapter._check_auth(mock_request) is None

    def test_invalid_key_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401

    def test_missing_auth_header_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401

    def test_malformed_auth_header_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# Helpers for HTTP tests
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    app.router.add_get("/api/jobs", adapter._handle_list_jobs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_security_headers_present(self, adapter):
        """Responses should include basic security headers."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("Referrer-Policy") == "no-referrer"

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "hermes-agent"

    @pytest.mark.asyncio
    async def test_v1_health_alias_returns_ok(self, adapter):
        """GET /v1/health should return the same response as /health."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "hermes-agent"


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_hermes_agent(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "hermes-agent"
            assert data["data"][0]["owned_by"] == "hermes"

    @pytest.mark.asyncio
    async def test_models_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_models_with_valid_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/v1/models",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "messages" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test", "messages": []})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_stream_true_returns_sse(self, adapter):
        """stream=true returns SSE format with the full response."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # Simulate streaming: invoke stream_delta_callback with tokens
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)  # End signal
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                body = await resp.text()
                assert "data: " in body
                assert "[DONE]" in body
                assert "Hello!" in body

    @pytest.mark.asyncio
    async def test_stream_survives_tool_call_none_sentinel(self, adapter):
        """stream_delta_callback(None) mid-stream (tool calls) must NOT kill the SSE stream.

        The agent fires stream_delta_callback(None) to tell the CLI display to
        close its response box before executing tool calls.  The API server's
        _on_delta must filter this out so the SSE response stays open and the
        final answer (streamed after tool execution) reaches the client.
        """
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    # Simulate: agent streams partial text, then fires None
                    # (tool call box-close signal), then streams the final answer
                    cb("Thinking")
                    cb(None)          # mid-stream None from tool calls
                    await asyncio.sleep(0.05)  # simulate tool execution delay
                    cb(" about it...")
                    cb(None)          # another None (possible second tool round)
                    await asyncio.sleep(0.05)
                    cb(" The answer is 42.")
                return (
                    {"final_response": "Thinking about it... The answer is 42.", "messages": [], "api_calls": 3},
                    {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "What is the answer?"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # The final answer text must appear in the SSE stream
                assert "The answer is 42." in body
                # All partial text must be present too
                assert "Thinking" in body
                assert " about it..." in body

    @pytest.mark.asyncio
    async def test_stream_includes_tool_progress(self, adapter):
        """tool_progress_callback fires → progress appears in the SSE stream."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                tp_cb = kwargs.get("tool_progress_callback")
                # Simulate tool progress before streaming content
                if tp_cb:
                    tp_cb("tool.started", "terminal", "ls -la", {"command": "ls -la"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Here are the files.")
                return (
                    {"final_response": "Here are the files.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list files"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # Tool progress message must appear in the stream
                assert "ls -la" in body
                # Final content must also be present
                assert "Here are the files." in body

    @pytest.mark.asyncio
    async def test_stream_tool_progress_skips_internal_events(self, adapter):
        """Internal events (name starting with _) are not streamed."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                tp_cb = kwargs.get("tool_progress_callback")
                if tp_cb:
                    tp_cb("tool.started", "_thinking", "some internal state", {})
                    tp_cb("tool.started", "web_search", "Python docs", {"query": "Python docs"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Found it.")
                return (
                    {"final_response": "Found it.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "search"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                # Internal _thinking event should NOT appear
                assert "some internal state" not in body
                # Real tool progress should appear
                assert "Python docs" in body

    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "system", "content": "You are helpful."}],
                },
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_successful_completion(self, adapter):
        """Test a successful chat completion with mocked agent."""
        mock_result = {
            "final_response": "Hello! How can I help you today?",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "chat.completion"
            assert data["id"].startswith("chatcmpl-")
            assert data["model"] == "hermes-agent"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert "usage" in data

    @pytest.mark.asyncio
    async def test_system_prompt_extracted(self, adapter):
        """System messages from the client are passed as ephemeral_system_prompt."""
        mock_result = {
            "final_response": "I am a pirate! Arrr!",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are a pirate."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            # Check that _run_agent was called with the system prompt
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("ephemeral_system_prompt") == "You are a pirate."
            assert call_kwargs.kwargs.get("user_message") == "Hello"

    @pytest.mark.asyncio
    async def test_conversation_history_passed(self, adapter):
        """Previous user/assistant messages become conversation_history."""
        mock_result = {"final_response": "3", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "1+1=?"},
                            {"role": "assistant", "content": "2"},
                            {"role": "user", "content": "Now add 1 more"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["user_message"] == "Now add 1 more"
            assert len(call_kwargs["conversation_history"]) == 2
            assert call_kwargs["conversation_history"][0] == {"role": "user", "content": "1+1=?"}
            assert call_kwargs["conversation_history"][1] == {"role": "assistant", "content": "2"}

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        """Agent exception returns 500."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Provider failed")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 500
            data = await resp.json()
            assert "Provider failed" in data["error"]["message"]


# ---------------------------------------------------------------------------
# /v1/responses endpoint
# ---------------------------------------------------------------------------


class TestResponsesEndpoint:
    @pytest.mark.asyncio
    async def test_missing_input_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_successful_response_with_string_input(self, adapter):
        """String input is wrapped in a user message."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "What is the capital of France?",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["id"].startswith("resp_")
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_successful_response_with_array_input(self, adapter):
        """Array input with role/content objects."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {"role": "user", "content": "Hello"},
                            {"role": "user", "content": "What is 2+2?"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # Last message is user_message, rest are history
            assert call_kwargs["user_message"] == "What is 2+2?"
            assert len(call_kwargs["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_instructions_as_ephemeral_prompt(self, adapter):
        """The instructions field maps to ephemeral_system_prompt."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Talk like a pirate.",
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Talk like a pirate."

    @pytest.mark.asyncio
    async def test_previous_response_id_chaining(self, adapter):
        """Test that responses can be chained via previous_response_id."""
        mock_result_1 = {
            "final_response": "2",
            "messages": [{"role": "assistant", "content": "2"}],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_1, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 1+1?"},
                )

            assert resp1.status == 200
            data1 = await resp1.json()
            response_id = data1["id"]

            # Second request chaining from the first
            mock_result_2 = {
                "final_response": "3",
                "messages": [{"role": "assistant", "content": "3"}],
                "api_calls": 1,
            }

            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_2, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": response_id,
                    },
                )

            assert resp2.status == 200
            # The conversation_history should contain the full history from the first response
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["conversation_history"]) > 0
            assert call_kwargs["user_message"] == "Now add 1 more"

    @pytest.mark.asyncio
    async def test_invalid_previous_response_id_returns_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "follow up",
                    "previous_response_id": "resp_nonexistent",
                },
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_store_false_does_not_store(self, adapter):
        """When store=false, the response is NOT stored."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "store": False,
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            # The response has an ID but it shouldn't be retrievable
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_instructions_inherited_from_previous(self, adapter):
        """If no instructions provided, carry forward from previous response."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request with instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Be a pirate",
                    },
                )

            data1 = await resp1.json()
            resp_id = data1["id"]

            # Second request without instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Tell me more",
                        "previous_response_id": resp_id,
                    },
                )

            assert resp2.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Be a pirate"

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Boom")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 500

    @pytest.mark.asyncio
    async def test_invalid_input_type_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": 42},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Auth on endpoints
# ---------------------------------------------------------------------------


class TestEndpointAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_responses_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "test", "input": "hi"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_models_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_platform_enum_has_api_server(self):
        assert Platform.API_SERVER.value == "api_server"

    def test_env_override_enables_api_server(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert Platform.API_SERVER in config.platforms
        assert config.platforms[Platform.API_SERVER].enabled is True

    def test_env_override_with_key(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "sk-mykey")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert Platform.API_SERVER in config.platforms
        assert config.platforms[Platform.API_SERVER].extra.get("key") == "sk-mykey"

    def test_env_override_port_and_host(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv("API_SERVER_PORT", "9999")
        monkeypatch.setenv("API_SERVER_HOST", "0.0.0.0")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("port") == 9999
        assert config.platforms[Platform.API_SERVER].extra.get("host") == "0.0.0.0"

    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv(
            "API_SERVER_CORS_ORIGINS",
            "http://localhost:3000, http://127.0.0.1:3000",
        )
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("cors_origins") == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_api_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=True)
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER in connected

    def test_api_server_not_in_connected_when_disabled(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=False)
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER not in connected


# ---------------------------------------------------------------------------
# Multiple system messages
# ---------------------------------------------------------------------------


class TestMultipleSystemMessages:
    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self, adapter):
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            prompt = call_kwargs["ephemeral_system_prompt"]
            assert "You are helpful." in prompt
            assert "Be concise." in prompt


# ---------------------------------------------------------------------------
# send() method (not used but required by base)
# ---------------------------------------------------------------------------


class TestSendMethod:
    @pytest.mark.asyncio
    async def test_send_returns_not_supported(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        result = await adapter.send("chat1", "hello")
        assert result.success is False
        assert "HTTP request/response" in result.error


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_stored_response(self, adapter):
        """GET returns a previously stored response."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Create a response first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            response_id = data["id"]

            # Now GET it
            resp2 = await cli.get(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_not_found(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_any")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestDeleteResponse:
    @pytest.mark.asyncio
    async def test_delete_stored_response(self, adapter):
        """DELETE removes a stored response and returns confirmation."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            data = await resp.json()
            response_id = data["id"]

            # Delete it
            resp2 = await cli.delete(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["deleted"] is True

            # Verify it's gone
            resp3 = await cli.get(f"/v1/responses/{response_id}")
            assert resp3.status == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/v1/responses/resp_nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/v1/responses/resp_any")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Tool calls in output
# ---------------------------------------------------------------------------


class TestToolCallsInOutput:
    @pytest.mark.asyncio
    async def test_tool_calls_in_output(self, adapter):
        """When agent returns tool calls, they appear as function_call items."""
        mock_result = {
            "final_response": "The result is 42.",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "6*7"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "42",
                },
                {
                    "role": "assistant",
                    "content": "The result is 42.",
                },
            ],
            "api_calls": 2,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 6*7?"},
                )

            assert resp.status == 200
            data = await resp.json()
            output = data["output"]

            # Should have: function_call, function_call_output, message
            assert len(output) == 3
            assert output[0]["type"] == "function_call"
            assert output[0]["name"] == "calculator"
            assert output[0]["arguments"] == '{"expression": "6*7"}'
            assert output[0]["call_id"] == "call_abc123"
            assert output[1]["type"] == "function_call_output"
            assert output[1]["call_id"] == "call_abc123"
            assert output[1]["output"] == "42"
            assert output[2]["type"] == "message"
            assert output[2]["content"][0]["text"] == "The result is 42."

    @pytest.mark.asyncio
    async def test_no_tool_calls_still_works(self, adapter):
        """Without tool calls, output is just a message."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------


class TestUsageCounting:
    @pytest.mark.asyncio
    async def test_responses_usage(self, adapter):
        """Responses API returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["input_tokens"] == 100
            assert data["usage"]["output_tokens"] == 50
            assert data["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_chat_completions_usage(self, adapter):
        """Chat completions returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["prompt_tokens"] == 200
            assert data["usage"]["completion_tokens"] == 80
            assert data["usage"]["total_tokens"] == 280


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_truncation_auto_limits_history(self, adapter):
        """With truncation=auto, history over 100 messages is trimmed."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        # Pre-seed a stored response with a long history
        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(150)]
        adapter._response_store.put("resp_prev", {
            "response": {"id": "resp_prev", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_prev",
                        "truncation": "auto",
                    },
                )

        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        # History should be truncated to 100
        assert len(call_kwargs["conversation_history"]) <= 100

    @pytest.mark.asyncio
    async def test_no_truncation_keeps_full_history(self, adapter):
        """Without truncation=auto, long history is passed as-is."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(150)]
        adapter._response_store.put("resp_prev2", {
            "response": {"id": "resp_prev2", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_prev2",
                    },
                )

        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        assert len(call_kwargs["conversation_history"]) == 150


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_origin_allowed_for_non_browser_client(self, adapter):
        assert adapter._origin_allowed("") is True

    def test_origin_rejected_by_default(self, adapter):
        assert adapter._origin_allowed("http://evil.example") is False

    def test_origin_allowed_for_allowlist_match(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._origin_allowed("http://localhost:3000") is True

    def test_cors_headers_for_origin_disabled_by_default(self, adapter):
        assert adapter._cors_headers_for_origin("http://localhost:3000") is None

    def test_cors_headers_for_origin_matches_allowlist(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        headers = adapter._cors_headers_for_origin("http://localhost:3000")
        assert headers is not None
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "POST" in headers["Access-Control-Allow-Methods"]

    def test_cors_headers_for_origin_rejects_unknown_origin(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._cors_headers_for_origin("http://evil.example") is None

    @pytest.mark.asyncio
    async def test_cors_headers_not_present_by_default(self, adapter):
        """CORS is disabled unless explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_browser_origin_rejected_by_default(self, adapter):
        """Browser-originated requests are rejected unless explicitly allowed."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://evil.example"})
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_cors_options_preflight_rejected_by_default(self, adapter):
        """Browser preflight is rejected unless CORS is explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://evil.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_cors_headers_present_for_allowed_origin(self):
        """Allowed origins receive explicit CORS headers."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
            assert "DELETE" in resp.headers.get("Access-Control-Allow-Methods", "")

    @pytest.mark.asyncio
    async def test_cors_allows_idempotency_key_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Idempotency-Key",
                },
            )
            assert resp.status == 200
            assert "Idempotency-Key" in resp.headers.get("Access-Control-Allow-Headers", "")

    @pytest.mark.asyncio
    async def test_cors_sets_vary_origin_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Vary") == "Origin"

    @pytest.mark.asyncio
    async def test_cors_options_preflight_allowed_for_configured_origin(self):
        """Configured origins can complete browser preflight."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")


    @pytest.mark.asyncio
    async def test_cors_preflight_sets_max_age(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Max-Age") == "600"
# ---------------------------------------------------------------------------
# Conversation parameter
# ---------------------------------------------------------------------------


class TestConversationParameter:
    @pytest.mark.asyncio
    async def test_conversation_creates_new(self, adapter):
        """First request with a conversation name works (new conversation)."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post("/v1/responses", json={
                    "input": "hi",
                    "conversation": "my-chat",
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "completed"
                # Conversation mapping should be set
                assert adapter._response_store.get_conversation("my-chat") is not None

    @pytest.mark.asyncio
    async def test_conversation_chains_automatically(self, adapter):
        """Second request with same conversation name chains to first."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First response", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # First request
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "test-conv",
                })
                assert resp1.status == 200
                data1 = await resp1.json()
                resp1_id = data1["id"]

                # Second request — should chain
                mock_run.return_value = (
                    {"final_response": "Second response", "messages": [], "api_calls": 1},
                    {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                )
                resp2 = await cli.post("/v1/responses", json={
                    "input": "follow up",
                    "conversation": "test-conv",
                })
                assert resp2.status == 200

                # The second call should have received conversation history from the first
                assert mock_run.call_count == 2
                second_call_kwargs = mock_run.call_args_list[1]
                history = second_call_kwargs.kwargs.get("conversation_history",
                          second_call_kwargs[1].get("conversation_history", []) if len(second_call_kwargs) > 1 else [])
                # History should be non-empty (contains messages from first response)
                assert len(history) > 0

    @pytest.mark.asyncio
    async def test_conversation_and_previous_response_id_conflict(self, adapter):
        """Cannot use both conversation and previous_response_id."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={
                "input": "hi",
                "conversation": "my-chat",
                "previous_response_id": "resp_abc123",
            })
            assert resp.status == 400
            data = await resp.json()
            assert "Cannot use both" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_separate_conversations_are_isolated(self, adapter):
        """Different conversation names have independent histories."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Response A", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Conversation A
                await cli.post("/v1/responses", json={"input": "conv-a msg", "conversation": "conv-a"})
                # Conversation B
                mock_run.return_value = (
                    {"final_response": "Response B", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "conv-b msg", "conversation": "conv-b"})

                # They should have different response IDs in the mapping
                assert adapter._response_store.get_conversation("conv-a") != adapter._response_store.get_conversation("conv-b")

    @pytest.mark.asyncio
    async def test_conversation_store_false_no_mapping(self, adapter):
        """If store=false, conversation mapping is not updated."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Ephemeral", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post("/v1/responses", json={
                    "input": "hi",
                    "conversation": "ephemeral-chat",
                    "store": False,
                })
                assert resp.status == 200
                # Conversation mapping should NOT be set since store=false
                assert adapter._response_store.get_conversation("ephemeral-chat") is None


# ---------------------------------------------------------------------------
# X-Hermes-Session-Id header (session continuity)
# ---------------------------------------------------------------------------


class TestSessionIdHeader:
    @pytest.mark.asyncio
    async def test_new_session_response_includes_session_id_header(self, adapter):
        """Without X-Hermes-Session-Id, a new session is created and returned in the header."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "Hi"}]},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Id") is not None

    @pytest.mark.asyncio
    async def test_provided_session_id_is_used_and_echoed(self, adapter):
        """When X-Hermes-Session-Id is provided, it's passed to the agent and echoed in the response."""
        mock_result = {"final_response": "Continuing!", "messages": [], "api_calls": 1}
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "previous reply"},
        ]
        adapter._session_db = mock_db
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "my-session-123"},
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "Continue"}]},
                )

            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Id") == "my-session-123"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["session_id"] == "my-session-123"

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        adapter._session_db = mock_db
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"

    @pytest.mark.asyncio
    async def test_db_failure_falls_back_to_empty_history(self, adapter):
        """If SessionDB raises, history falls back to empty and request still succeeds."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        # Simulate DB failure: _session_db is None and SessionDB() constructor raises
        adapter._session_db = None
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
                 patch("hermes_state.SessionDB", side_effect=Exception("DB unavailable")):
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "some-session"},
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "Hi"}]},
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["conversation_history"] == []
            assert call_kwargs["session_id"] == "some-session"


# ---------------------------------------------------------------------------
# Multi-user mode (S-0724-01) — CT Chat API per-user identity
# ---------------------------------------------------------------------------


_MU_KEY = "sk-ct-secret"
_MU_AUTH = {"Authorization": f"Bearer {_MU_KEY}"}


class _FakeSessionEntry:
    def __init__(self, session_key, session_id):
        self.session_key = session_key
        self.session_id = session_id


class _FakeSessionStore:
    """Minimal SessionStore stand-in: maps the derived session_key to a real
    (distinct) session_id, mirroring the registry's key≠id split. session_id
    starts as '<key>::sid0' so tests can tell key and id apart."""

    def __init__(self):
        self._entries = {}
        self.saved = 0

    def get_or_create_session(self, source):
        from gateway.session import build_session_key
        key = build_session_key(source)
        if key not in self._entries:
            self._entries[key] = _FakeSessionEntry(key, f"{key}::sid0")
        return self._entries[key]

    def _save(self):
        self.saved += 1


def _make_multi_user_adapter(api_key: str = _MU_KEY) -> APIServerAdapter:
    """Create an adapter with multi-user mode enabled + a fake session store.

    Multi-user mode requires an API key (the identity header must sit behind an
    authenticated caller) AND a session store (the gateway wires one at
    startup; D2 routes through it). End-to-end requests must include
    ``_MU_AUTH`` in their headers.
    """
    extra = {"multi_user": True, "key": api_key}
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    adapter.set_session_store(_FakeSessionStore())
    return adapter


class TestMultiUserConfig:
    def test_multi_user_off_by_default(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        assert adapter._multi_user is False

    def test_multi_user_from_extra(self):
        # Multi-user mode requires an API key (see TestMultiUserAuthAndNativeIsolation).
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"multi_user": True, "key": "sk-x"}))
        assert adapter._multi_user is True

    def test_multi_user_from_env(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_MULTI_USER", "true")
        monkeypatch.setenv("API_SERVER_KEY", "sk-env")
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        assert adapter._multi_user is True

    def test_multi_user_string_false_stays_off(self):
        """A quoted-YAML / string 'false' must NOT enable multi-user
        (plain bool('false') is True — the coercion guards against that)."""
        for falsey in ("false", "0", "no", "off", ""):
            adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"multi_user": falsey}))
            assert adapter._multi_user is False, f"{falsey!r} should not enable multi-user"

    def test_multi_user_string_true_enables(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"multi_user": "true", "key": "sk-x"}))
        assert adapter._multi_user is True


class TestResolveCtUserId:
    """D1 — identity header validation."""

    def test_missing_header_returns_401(self):
        adapter = _make_multi_user_adapter()
        req = MagicMock()
        req.headers = {}
        user_id, err = adapter._resolve_ct_user_id(req)
        assert user_id is None
        assert err is not None and err.status == 401

    def test_valid_header_returns_user_id(self):
        adapter = _make_multi_user_adapter()
        req = MagicMock()
        req.headers = {"X-Hermes-User-Id": "ct-abc123"}
        user_id, err = adapter._resolve_ct_user_id(req)
        assert err is None
        assert user_id == "ct-abc123"

    def test_malformed_header_returns_401(self):
        adapter = _make_multi_user_adapter()
        for bad in ["abc123", "ct-", "ct-../other", "ct-with space", "U0123"]:
            req = MagicMock()
            req.headers = {"X-Hermes-User-Id": bad}
            user_id, err = adapter._resolve_ct_user_id(req)
            assert user_id is None, f"expected reject for {bad!r}"
            assert err is not None and err.status == 401

    def test_single_owner_mode_ignores_header(self):
        """Multi-user off: no header required, resolver returns (None, None)."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        req = MagicMock()
        req.headers = {}
        user_id, err = adapter._resolve_ct_user_id(req)
        assert user_id is None
        assert err is None


class TestMultiUserSessionRouting:
    """D2 — session-registry routing."""

    def test_session_key_matches_registry_scheme(self):
        from gateway.config import Platform as _P
        from gateway.session import SessionSource, build_session_key

        source = SessionSource(platform=_P.API_SERVER, chat_id="ct-abc123", user_id="ct-abc123")
        assert build_session_key(source) == "agent:main:api_server:dm:ct-abc123"

    @pytest.mark.asyncio
    async def test_repeated_calls_same_user_share_session(self):
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        seen_sessions = []

        seen_release_keys = []

        async def _mock_run_agent(**kwargs):
            seen_sessions.append(kwargs.get("session_id"))
            seen_release_keys.append(kwargs.get("release_key"))
            # Mirror the real _run_agent's in-worker guard release so the second
            # sequential call is not falsely rejected as busy.
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                for _ in range(2):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"},
                        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert resp.status == 200
        _key = "agent:main:api_server:dm:ct-abc123"
        # The agent receives the registry's REAL session_id (distinct from the
        # stable key), and both calls resolve the same conversation.
        assert seen_sessions[0] == seen_sessions[1] == f"{_key}::sid0"
        # The busy guard is keyed by the stable session_key, not the id.
        assert seen_release_keys[0] == seen_release_keys[1] == _key

    @pytest.mark.asyncio
    async def test_compression_continuation_followed_across_turns(self):
        """P2 (round-12): if a run compresses context and the agent switches to
        a new continuation session_id, the registry entry must be updated so the
        NEXT turn resolves the continuation — not the stale pre-compression id.
        Simulated by having the mocked run return a changed session_id."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        _key = "agent:main:api_server:dm:ct-compress"
        seen_ids = []
        turn = {"n": 0}

        async def _mock_run_agent(**kwargs):
            seen_ids.append(kwargs.get("session_id"))
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            turn["n"] += 1
            # First turn compresses → agent moves to a NEW continuation id.
            new_sid = f"{_key}::sid_compressed" if turn["n"] == 1 else kwargs.get("session_id")
            # Mirror the real _run_agent's IN-THREAD writeback.
            we = kwargs.get("writeback_entry")
            if we is not None:
                adapter._writeback_session_id(we, {"session_id": new_sid})
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1, "session_id": new_sid},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                for _ in range(2):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-compress"},
                        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert resp.status == 200
        # Turn 1 ran against the original id; turn 2 must resolve the compressed
        # continuation the writeback persisted, NOT the stale original.
        assert seen_ids[0] == f"{_key}::sid0"
        assert seen_ids[1] == f"{_key}::sid_compressed", "continuation not followed — R12 regression"

    @pytest.mark.asyncio
    async def test_idempotency_cache_hit_does_not_rewind_session(self):
        """P2 (round-13): an idempotency cache HIT must NOT write the cached
        (possibly stale) session_id back to the registry — that would rewind a
        session that a later compression already advanced."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        _key = "agent:main:api_server:dm:ct-idem"
        # Prime the store entry and advance its id (as a later compression would).
        from gateway.session import SessionSource
        se = adapter._session_store.get_or_create_session(
            SessionSource(platform=Platform.API_SERVER, chat_id="ct-idem", user_id="ct-idem"))

        async def _mock_run_agent(**kwargs):
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            # Returns a result carrying an OLD session id (simulating a cached
            # response from before a later compression).
            return (
                {"final_response": "cached", "messages": [], "api_calls": 1,
                 "session_id": f"{_key}::sid0"},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        hdr = {**_MU_AUTH, "X-Hermes-User-Id": "ct-idem", "Idempotency-Key": "kkk"}
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                # First call computes + writes back sid0.
                await cli.post("/v1/chat/completions", headers=hdr, json=body)
                # Simulate a later compression advancing the registry entry.
                se.session_id = f"{_key}::sid_compressed"
                # Second call (same key + body) is a CACHE HIT → must NOT rewind.
                await cli.post("/v1/chat/completions", headers=hdr, json=body)
        assert se.session_id == f"{_key}::sid_compressed", "cache hit rewound the session id"

    @pytest.mark.asyncio
    async def test_stream_disconnect_still_writes_back_session_id(self):
        """P2 (round-13): on the stream path the session_id writeback happens
        inside the executor thread, so it survives a client disconnect (which
        cancels the asyncio wrapper but not the thread)."""
        adapter = _make_multi_user_adapter()
        _key = "agent:main:api_server:dm:ct-streamd"
        from gateway.session import SessionSource
        se = adapter._session_store.get_or_create_session(
            SessionSource(platform=Platform.API_SERVER, chat_id="ct-streamd", user_id="ct-streamd"))

        # Drive _run_agent directly (as the stream path does) with a fake agent
        # whose run_conversation "compresses" — changes session_id.
        fake = MagicMock()
        def _run_conv(**kw):
            fake.session_id = f"{_key}::sid_compressed"
            return {"final_response": "ok", "messages": [], "api_calls": 1}
        fake.run_conversation.side_effect = _run_conv
        fake.session_id = f"{_key}::sid0"
        fake.session_prompt_tokens = fake.session_completion_tokens = fake.session_total_tokens = 0

        with patch.object(adapter, "_create_agent", return_value=fake):
            await adapter._run_agent(
                user_message="hi", conversation_history=[],
                session_id=f"{_key}::sid0", user_id="ct-streamd",
                release_key=_key, writeback_entry=se,
            )
        # In-thread writeback ran regardless of any wrapper cancellation.
        assert se.session_id == f"{_key}::sid_compressed"

    @pytest.mark.asyncio
    async def test_non_stream_passes_writeback_entry(self):
        """P2 (round-14): the NON-stream path must also thread writeback_entry
        into _run_agent, so a compression-driven id change survives a mid-run
        cancellation (post-await writeback would be skipped)."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        captured = {}

        async def _spy(**kwargs):
            captured["writeback_entry"] = kwargs.get("writeback_entry")
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_spy):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-nswb"},
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}],
                          "stream": False},
                )
                assert resp.status == 200
        # A registry entry was threaded so the in-thread writeback can persist it.
        assert captured["writeback_entry"] is not None
        assert captured["writeback_entry"].session_key == "agent:main:api_server:dm:ct-nswb"

    def test_writeback_uses_store_lock_when_present(self):
        """P2 (round-14): the writeback mutates the entry + _save() UNDER the
        store lock (normal store paths hold it around _entries + _save)."""
        import threading as _t

        adapter = _make_multi_user_adapter()
        _key = "agent:main:api_server:dm:ct-lock"
        from gateway.session import SessionSource
        se = adapter._session_store.get_or_create_session(
            SessionSource(platform=Platform.API_SERVER, chat_id="ct-lock", user_id="ct-lock"))

        # Give the fake store a real lock + a save that asserts the lock is held.
        lock = _t.Lock()
        held_during_save = {"ok": False}
        adapter._session_store._lock = lock
        orig_save = adapter._session_store._save
        def _save_checked():
            held_during_save["ok"] = not lock.acquire(blocking=False)  # already held → can't re-acquire
            if not held_during_save["ok"]:
                lock.release()
            orig_save()
        adapter._session_store._save = _save_checked

        adapter._writeback_session_id(se, {"session_id": f"{_key}::sid_new"})
        assert se.session_id == f"{_key}::sid_new"
        assert held_during_save["ok"], "writeback did not hold the store lock during _save()"

    @pytest.mark.asyncio
    async def test_missing_header_rejected_end_to_end(self):
        """Authenticated caller, but no identity header → 401 on identity (not auth)."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=dict(_MU_AUTH),
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401
            data = await resp.json()
            assert data["error"]["code"] == "invalid_user_id"


class TestMultiUserIdentityThreading:
    """D3 — per-call identity threading into the executor thread."""

    @pytest.mark.asyncio
    async def test_user_id_bound_inside_agent_thread(self):
        """The real _run_agent must set session_context user_id INSIDE the
        executor thread, so agent.run_conversation sees the calling user."""
        from tools import session_context

        adapter = _make_multi_user_adapter()
        captured = {}

        fake_agent = MagicMock()

        def _run_conversation(**kwargs):
            captured["user_id"] = session_context.get_user_id()
            return {"final_response": "ok", "messages": [], "api_calls": 1}

        fake_agent.run_conversation.side_effect = _run_conversation
        fake_agent.session_prompt_tokens = 1
        fake_agent.session_completion_tokens = 1
        fake_agent.session_total_tokens = 2

        with patch.object(adapter, "_create_agent", return_value=fake_agent):
            await adapter._run_agent(
                user_message="hi",
                conversation_history=[],
                session_id="agent:main:api_server:dm:ct-abc123",
                user_id="ct-abc123",
            )
        assert captured["user_id"] == "ct-abc123"
        # And cleared after the run so a reused thread carries no residue.
        assert session_context.get_user_id() is None

    @pytest.mark.asyncio
    async def test_thread_reuse_does_not_leak_identity(self):
        """Two runs on the same adapter (pooled threads reused) must not leak
        one user's identity into the next run."""
        from tools import session_context

        adapter = _make_multi_user_adapter()
        seen = []

        def _make_fake_agent():
            fake = MagicMock()
            fake.run_conversation.side_effect = lambda **kw: (
                seen.append(session_context.get_user_id())
                or {"final_response": "ok", "messages": [], "api_calls": 1}
            )
            fake.session_prompt_tokens = 0
            fake.session_completion_tokens = 0
            fake.session_total_tokens = 0
            return fake

        with patch.object(adapter, "_create_agent", side_effect=lambda **kw: _make_fake_agent()):
            await adapter._run_agent(
                user_message="hi", conversation_history=[],
                session_id="agent:main:api_server:dm:ct-A", user_id="ct-A",
            )
            await adapter._run_agent(
                user_message="hi", conversation_history=[],
                session_id="agent:main:api_server:dm:ct-B", user_id="ct-B",
            )
        assert seen == ["ct-A", "ct-B"]


class TestMultiUserBusyGuard:
    """D4 — 409 busy guard on concurrent same-session requests."""

    @pytest.mark.asyncio
    async def test_concurrent_same_session_returns_409(self):
        import asyncio

        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        release = asyncio.Event()

        async def _slow_run_agent(**kwargs):
            await release.wait()
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_slow_run_agent):
                hdr = {**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"}
                body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
                first = asyncio.ensure_future(cli.post("/v1/chat/completions", headers=hdr, json=body))
                await asyncio.sleep(0.1)  # let first acquire the session
                second = await cli.post("/v1/chat/completions", headers=hdr, json=body)
                assert second.status == 409
                release.set()
                first_resp = await first
                assert first_resp.status == 200

    @pytest.mark.asyncio
    async def test_busy_guard_acquired_before_history_load(self):
        """A 409-rejected request must NOT read session history first — the
        guard is acquired before the DB load, so a busy session never loads
        stale history in the pre-guard window."""
        import asyncio

        adapter = _make_multi_user_adapter()
        key = "agent:main:api_server:dm:ct-abc123"
        # Simulate a prior run already holding the session.
        adapter._active_chat_sessions.add(key)
        app = _create_app(adapter)

        db = MagicMock()
        db.get_messages_as_conversation.return_value = []
        with patch.object(adapter, "_ensure_session_db", return_value=db):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"},
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status == 409
        # History was never loaded — the guard short-circuited before the DB read.
        db.get_messages_as_conversation.assert_not_called()


class TestMultiUserIsolationHardening:
    """Codex-review findings (S-0724-01): idempotency namespacing + endpoint gating."""

    @pytest.mark.asyncio
    async def test_idempotency_key_namespaced_by_user(self):
        """Two CT users, same Idempotency-Key + body, must NOT share a cached
        result (P1: global idempotency cache leaked across users)."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        calls = []

        async def _mock_run_agent(**kwargs):
            calls.append(kwargs.get("user_id"))
            return (
                {"final_response": f"hi {kwargs.get('user_id')}", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
                r1 = await cli.post("/v1/chat/completions",
                                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-A", "Idempotency-Key": "k1"}, json=body)
                r2 = await cli.post("/v1/chat/completions",
                                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-B", "Idempotency-Key": "k1"}, json=body)
                assert r1.status == 200 and r2.status == 200
                d1, d2 = await r1.json(), await r2.json()
        # Both users ran the agent (no cross-user cache hit) and got their own reply.
        assert calls == ["ct-A", "ct-B"]
        assert d1["choices"][0]["message"]["content"] == "hi ct-A"
        assert d2["choices"][0]["message"]["content"] == "hi ct-B"

    @pytest.mark.asyncio
    async def test_responses_endpoint_rejected_in_multi_user(self):
        """P2: /v1/responses must not run identity-less in multi-user mode."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"},
                json={"model": "test", "input": "hi"},
            )
            assert resp.status == 501

    @pytest.mark.asyncio
    async def test_responses_endpoint_works_in_single_owner(self):
        """Single-owner mode: /v1/responses still works (no regression)."""
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post("/v1/responses", json={"model": "test", "input": "hi"})
            assert resp.status == 200

    def test_jobs_api_rejected_in_multi_user(self):
        """P1 (round-7): /api/jobs operates on GLOBAL cron state with no per-user
        scoping — the shared jobs gate must reject in multi-user mode. All 8 job
        handlers route through _check_jobs_available, so gating it covers them."""
        adapter = _make_multi_user_adapter()
        resp = adapter._check_jobs_available()
        assert resp is not None and resp.status == 501

    def test_jobs_api_available_in_single_owner(self):
        """Single-owner mode: jobs gate does not reject on the multi-user axis."""
        adapter = _make_adapter()
        # _CRON_AVAILABLE may be True/False in the test env; assert only that the
        # gate does NOT reject for the multi-user reason (it would if it did).
        resp = adapter._check_jobs_available()
        if resp is not None:
            # Only acceptable rejection here is "cron module not available".
            import json as _json
            body = _json.loads(resp.body.decode())
            assert body["error"] == "Cron module not available"


class TestGuardSentinelIsNotNone:
    """S-0724-01 (codex round 8/9): guard helpers return aiohttp responses used
    as error sentinels. These MUST be checked with ``is not None``, never plain
    truthiness — aiohttp <3.13 (this project supports aiohttp>=3.9.0) has NO
    ``StreamResponse.__bool__`` and inherits MutableMapping's ``__len__``, so a
    freshly-built response with empty _state is FALSY and ``if err:`` silently
    bypasses the guard. This class pins the invariant across the supported range
    so a regression back to truthiness is caught on any aiohttp version."""

    def test_no_bare_truthiness_guard_remains_in_source(self):
        """Every response-sentinel check in the adapter uses `is not None`."""
        import re
        import gateway.platforms.api_server as mod
        src = open(mod.__file__).read()
        bad = re.findall(r"^\s+if (?:auth_err|id_err|mu_err|cron_err):\s*$", src, flags=re.M)
        assert bad == [], f"found {len(bad)} bare-truthiness guard(s); use `is not None`"

    def test_guards_fire_even_when_response_is_falsy(self):
        """Simulate the aiohttp 3.9.0 case: a guard response that is falsy under
        bool() must still be treated as an error by the handler's `is not None`
        check. We assert the guards return a NON-None response on the reject
        path (which `is not None` catches regardless of its truthiness)."""
        adapter = _make_multi_user_adapter()
        req = MagicMock()
        req.headers = {}  # no identity header → reject
        _uid, id_err = adapter._resolve_ct_user_id(req)
        assert id_err is not None and id_err.status == 401
        mu_err = adapter._reject_if_multi_user()
        assert mu_err is not None and mu_err.status == 501

    @pytest.mark.asyncio
    async def test_reject_still_enforced_when_falsy_response_patched(self):
        """End-to-end: patch the guard to return a genuinely-falsy response and
        confirm the endpoint still rejects — proving the handler does not rely
        on the response being truthy (the aiohttp <3.13 hazard)."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)

        class _FalsyResponse(web.Response):
            def __bool__(self):  # emulate aiohttp <3.13 empty-state falsiness
                return False

        falsy = _FalsyResponse(status=501)
        assert bool(falsy) is False  # sanity: it really is falsy

        with patch.object(adapter, "_reject_if_multi_user", return_value=falsy):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"},
                    json={"model": "test", "input": "hi"},
                )
                # is-not-None check must return the falsy 501, not fall through.
                assert resp.status == 501


class TestMultiUserEndpointSurfaceSweep:
    """Class-of-issue guard (S-0724-01): EVERY non-chat endpoint that reaches an
    agent run or global/unscoped state must reject in multi-user mode. This is
    the full-set assertion — a future endpoint added without the guard fails
    here, so the class stays closed without waiting for a reviewer to find it.

    Only /v1/chat/completions is the intended multi-user surface. /health and
    /v1/models are stateless and intentionally open.
    """

    # (method, path) for endpoints that MUST reject in multi-user mode.
    REJECTED = [
        ("POST", "/v1/responses"),
        ("GET", "/v1/responses/resp_abc"),
        ("DELETE", "/v1/responses/resp_abc"),
        ("GET", "/api/jobs"),
        ("GET", "/v1/runs/run_abc/events"),
        # /v1/runs POST + the /api/jobs/{id} mutation routes reject via the same
        # _reject_if_multi_user / _check_jobs_available gates covered elsewhere;
        # this list is the HTTP-observable subset registered in the test app.
    ]

    @pytest.mark.asyncio
    async def test_all_non_chat_endpoints_reject_in_multi_user(self):
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for method, path in self.REJECTED:
                resp = await cli.request(
                    method, path,
                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"},
                    json={"model": "test", "input": "hi"} if method == "POST" else None,
                )
                assert resp.status == 501, f"{method} {path} should reject in multi-user mode, got {resp.status}"

    @pytest.mark.asyncio
    async def test_chat_completions_is_the_only_open_agent_surface(self):
        """Sanity: the intended surface is NOT rejected (guards didn't over-reach)."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)

        async def _mock_run_agent(**kwargs):
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-abc123"},
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status == 200


class TestMultiUserAuthAndNativeIsolation:
    """Codex round-2 findings (S-0724-01): auth requirement + AIAgent identity + CORS."""

    def test_multi_user_without_api_key_raises(self):
        """P1: the identity header must sit behind auth — refuse to start
        multi-user mode with no API key (else header is spoofable)."""
        with pytest.raises(ValueError, match="requires an API key"):
            APIServerAdapter(PlatformConfig(enabled=True, extra={"multi_user": True}))

    def test_multi_user_with_api_key_ok(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"multi_user": True, "key": "sk-x"}))
        assert adapter._multi_user is True

    @pytest.mark.asyncio
    async def test_user_id_reaches_aiagent_constructor(self):
        """P1: _run_agent must pass user_id to _create_agent so AIAgent's own
        memory provider + SessionDB ownership are user-scoped (not just MCP)."""
        adapter = _make_multi_user_adapter()
        seen = {}

        real_create = adapter._create_agent

        def _spy_create(**kwargs):
            seen["user_id"] = kwargs.get("user_id")
            fake = MagicMock()
            fake.run_conversation.return_value = {"final_response": "ok", "messages": [], "api_calls": 1}
            fake.session_prompt_tokens = fake.session_completion_tokens = fake.session_total_tokens = 0
            return fake

        with patch.object(adapter, "_create_agent", side_effect=_spy_create):
            await adapter._run_agent(
                user_message="hi", conversation_history=[],
                session_id="agent:main:api_server:dm:ct-xyz", user_id="ct-xyz",
            )
        assert seen["user_id"] == "ct-xyz"

    def test_cors_allows_identity_header(self):
        assert "X-Hermes-User-Id" in _CORS_HEADERS["Access-Control-Allow-Headers"]

    def _capture_create_agent_toolsets(self, adapter, user_id):
        """Run _create_agent with the AIAgent + config chain stubbed, capturing
        the enabled_toolsets that reach the AIAgent constructor."""
        captured = {}

        def _fake_aiagent(**kwargs):
            captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")
            captured["user_id"] = kwargs.get("user_id")
            return MagicMock()

        import run_agent
        import gateway.run as gwrun
        from hermes_cli import tools_config
        # Stub the platform toolset to the full dangerous bundle so we can assert
        # the allowlist keeps only the safe groups.
        full_bundle = {
            "web", "memory", "todo", "skills",          # safe (allowlisted)
            "delegation", "session_search", "cronjob",  # unsafe (must drop)
            "file", "code_execution", "homeassistant",  # unsafe (must drop)
        }
        with patch.object(run_agent, "AIAgent", _fake_aiagent), \
             patch.object(gwrun, "_resolve_runtime_agent_kwargs", return_value={}), \
             patch.object(gwrun, "_resolve_gateway_model", return_value="m"), \
             patch.object(gwrun, "_load_gateway_config", return_value={}), \
             patch.object(gwrun.GatewayRunner, "_load_fallback_model", return_value=None), \
             patch.object(tools_config, "_get_platform_tools", return_value=full_bundle):
            adapter._create_agent(user_id=user_id)
        return captured

    def test_multi_user_toolset_is_safe_allowlist(self):
        """P1 (round-6/10/11): the full hermes-api-server bundle exposes
        host-level + cross-user tools. Multi-user mode keeps ONLY the safe
        allowlist. `memory` and `skills` are EXCLUDED (round-11): built-in
        memory writes shared files; `skills` bundles skill_manage (global
        write)."""
        adapter = _make_multi_user_adapter()
        captured = self._capture_create_agent_toolsets(adapter, user_id="ct-abc")
        kept = set(captured["enabled_toolsets"])
        assert kept == {"web", "todo"}, f"unexpected kept set: {kept}"
        # Explicitly assert every unsafe group is gone — incl. memory + skills.
        for danger in ("delegation", "session_search", "cronjob", "file",
                       "code_execution", "homeassistant", "memory", "skills"):
            assert danger not in kept, f"{danger} must be stripped in multi-user mode"
        assert captured["user_id"] == "ct-abc"

    def test_single_owner_toolset_untouched(self):
        """Single-owner mode keeps the full bundle (no allowlist filtering)."""
        adapter = _make_adapter()
        captured = self._capture_create_agent_toolsets(adapter, user_id=None)
        kept = set(captured["enabled_toolsets"])
        # Dangerous tools remain available for trusted single-owner / R&D use.
        assert "delegation" in kept and "session_search" in kept and "file" in kept
        assert "memory" in kept and "skills" in kept

    def _capture_created_agent(self, adapter, user_id):
        """Return the (mock) agent object produced by _create_agent, so callers
        can inspect attributes set on it post-construction."""
        class _Agent:
            _memory_enabled = True
            _user_profile_enabled = True

        made = _Agent()
        import run_agent
        import gateway.run as gwrun
        from hermes_cli import tools_config
        with patch.object(run_agent, "AIAgent", lambda **kw: made), \
             patch.object(gwrun, "_resolve_runtime_agent_kwargs", return_value={}), \
             patch.object(gwrun, "_resolve_gateway_model", return_value="m"), \
             patch.object(gwrun, "_load_gateway_config", return_value={}), \
             patch.object(gwrun.GatewayRunner, "_load_fallback_model", return_value=None), \
             patch.object(tools_config, "_get_platform_tools", return_value={"web", "todo"}):
            return adapter._create_agent(user_id=user_id)

    def test_builtin_memory_disabled_in_multi_user(self):
        """P1 (round-11): shared ~/.hermes/memories/MEMORY.md + USER.md must not
        be injected into a CT user's prompt. Built-in memory flags are forced
        off in multi-user mode (external Mem0, user-scoped, is untouched)."""
        adapter = _make_multi_user_adapter()
        agent = self._capture_created_agent(adapter, user_id="ct-abc")
        assert agent._memory_enabled is False
        assert agent._user_profile_enabled is False

    def test_builtin_memory_kept_in_single_owner(self):
        """Single-owner mode leaves built-in memory flags as constructed."""
        adapter = _make_adapter()
        agent = self._capture_created_agent(adapter, user_id=None)
        assert agent._memory_enabled is True
        assert agent._user_profile_enabled is True


class TestMultiUserGuardReleaseTiming:
    """Codex round-3 finding (S-0724-01): busy guard must be held until the
    executor thread truly exits, not when the asyncio wrapper is cancelled."""

    @pytest.mark.asyncio
    async def test_release_key_discarded_inside_worker_thread(self):
        """_run_agent releases release_key from inside _run (thread), so the
        guard is held for the worker's full lifetime."""
        adapter = _make_multi_user_adapter()
        key = "agent:main:api_server:dm:ct-race"
        adapter._active_chat_sessions.add(key)

        observed = {}

        def _run_conversation(**kwargs):
            # While the worker runs, the guard must still be held.
            observed["held_during_run"] = key in adapter._active_chat_sessions
            return {"final_response": "ok", "messages": [], "api_calls": 1}

        fake = MagicMock()
        fake.run_conversation.side_effect = _run_conversation
        fake.session_prompt_tokens = fake.session_completion_tokens = fake.session_total_tokens = 0

        with patch.object(adapter, "_create_agent", return_value=fake):
            await adapter._run_agent(
                user_message="hi", conversation_history=[],
                session_id=key, user_id="ct-race", release_key=key,
            )
        assert observed["held_during_run"] is True
        # Released only after the worker returned.
        assert key not in adapter._active_chat_sessions

    @pytest.mark.asyncio
    async def test_cancelling_wrapper_does_not_release_until_thread_done(self):
        """Cancelling the awaiting coroutine must NOT free the guard while the
        executor thread is still running (the disconnect race)."""
        import asyncio

        adapter = _make_multi_user_adapter()
        key = "agent:main:api_server:dm:ct-cancel"
        adapter._active_chat_sessions.add(key)

        import threading

        started = threading.Event()   # set from the worker thread
        proceed = threading.Event()   # blocks the worker until the test allows it

        def _run_conversation(**kwargs):
            started.set()
            proceed.wait(timeout=5)
            return {"final_response": "ok", "messages": [], "api_calls": 1}

        fake = MagicMock()
        fake.run_conversation.side_effect = _run_conversation
        fake.session_prompt_tokens = fake.session_completion_tokens = fake.session_total_tokens = 0

        with patch.object(adapter, "_create_agent", return_value=fake):
            task = asyncio.ensure_future(adapter._run_agent(
                user_message="hi", conversation_history=[],
                session_id=key, user_id="ct-cancel", release_key=key,
            ))
            # Wait for the worker thread to actually start.
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Wrapper cancelled, but worker thread still blocked → guard held.
            assert key in adapter._active_chat_sessions

            # Let the worker finish; the in-thread finally then releases it.
            proceed.set()
            for _ in range(200):
                if key not in adapter._active_chat_sessions:
                    break
                await asyncio.sleep(0.01)
            assert key not in adapter._active_chat_sessions

    @pytest.mark.asyncio
    async def test_non_stream_passes_release_key_to_worker(self):
        """Codex round-5: the non-stream compute path must release the guard
        in-worker (release_key), not only via the handler-coroutine finally —
        else a cancellation frees the guard while the worker still runs."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        captured = {}

        async def _spy_run_agent(**kwargs):
            captured["release_key"] = kwargs.get("release_key")
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_spy_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={**_MU_AUTH, "X-Hermes-User-Id": "ct-nstest"},
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                )
                assert resp.status == 200
        # The compute path threaded the session key as release_key (in-worker release).
        assert captured["release_key"] == "agent:main:api_server:dm:ct-nstest"

    @pytest.mark.asyncio
    async def test_idempotency_cache_hit_releases_guard(self):
        """Codex round-5: on a cache HIT the worker never runs, so the guard
        must be released by the coroutine finally (else it leaks forever)."""
        adapter = _make_multi_user_adapter()
        app = _create_app(adapter)
        run_count = {"n": 0}

        async def _mock_run_agent(**kwargs):
            run_count["n"] += 1
            rk = kwargs.get("release_key")
            if rk is not None:
                adapter._active_chat_sessions.discard(rk)
            return (
                {"final_response": "cached", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        hdr = {**_MU_AUTH, "X-Hermes-User-Id": "ct-hit", "Idempotency-Key": "same-key"}
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                r1 = await cli.post("/v1/chat/completions", headers=hdr, json=body)
                r2 = await cli.post("/v1/chat/completions", headers=hdr, json=body)
                assert r1.status == 200 and r2.status == 200
        # Second call was a cache hit (worker ran once), and the guard is free
        # afterward — the coroutine finally released it on the hit path.
        assert run_count["n"] == 1
        assert "agent:main:api_server:dm:ct-hit" not in adapter._active_chat_sessions
