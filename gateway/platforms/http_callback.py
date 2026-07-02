"""Generic HTTP callback platform adapter.

Hosts small deployment-supplied HTTP handlers inside the gateway process —
e.g. an OAuth redirect callback — so they need no separate service, port
supervision, or systemd unit: the listener starts and stops with the gateway.

Ingress-only: this platform never dispatches messages into the agent and has
no chats; the HTTP request/response cycle is the whole interaction.

Configuration (env-driven, like api_server):
  HERMES_HTTP_CALLBACK_HANDLER_DIR    directory put on sys.path (required)
  HERMES_HTTP_CALLBACK_HANDLER_MODULE importable module in that dir (required)
  HERMES_HTTP_CALLBACK_HOST           default 127.0.0.1
  HERMES_HTTP_CALLBACK_PORT           default 8765

Handler contract — the module exposes:

    HTTP_ROUTES: list[tuple[str, str, Callable[[str], tuple]]]

where each entry is ``(method, path, handler)`` and ``handler(query_string)``
returns either ``(status, html)`` or ``(status, html, redirect_location)``
(a non-empty ``redirect_location`` produces a 302 with that Location).
Handlers run in a thread (they may block on I/O); exceptions map to a 500.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def check_http_callback_requirements() -> bool:
    """Return whether required dependencies are available."""
    return AIOHTTP_AVAILABLE


class HTTPCallbackAdapter(BasePlatformAdapter):
    """Serve deployment-supplied HTTP callback routes from the gateway process."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.HTTP_CALLBACK)
        extra = config.extra or {}
        self._host: str = str(extra.get("host", DEFAULT_HOST))
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        self._handler_dir: str = str(extra.get("handler_dir", "")).strip()
        self._handler_module: str = str(extra.get("handler_module", "")).strip()
        self._routes: list = []
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None

    # -- handler loading -----------------------------------------------------

    def _load_routes(self) -> list:
        """Import the handler module and return its HTTP_ROUTES (validated)."""
        if not self._handler_module:
            raise ValueError("http_callback: handler_module is not configured")
        if self._handler_dir and self._handler_dir not in sys.path:
            sys.path.insert(0, self._handler_dir)
        module = importlib.import_module(self._handler_module)
        routes = getattr(module, "HTTP_ROUTES", None)
        if not isinstance(routes, (list, tuple)) or not routes:
            raise ValueError(
                f"http_callback: {self._handler_module} does not expose a "
                "non-empty HTTP_ROUTES list"
            )
        validated = []
        for entry in routes:
            method, path, handler = entry  # malformed entries raise here
            method = str(method).upper()
            if method not in ("GET", "POST"):
                raise ValueError(f"http_callback: unsupported method {method!r}")
            if not str(path).startswith("/"):
                raise ValueError(f"http_callback: path must start with '/': {path!r}")
            if not callable(handler):
                raise ValueError(f"http_callback: handler for {path!r} is not callable")
            validated.append((method, str(path), handler))
        return validated

    def _make_aiohttp_handler(self, handler):
        async def _handle(request: "web.Request") -> "web.StreamResponse":
            try:
                result = await asyncio.to_thread(handler, request.query_string)
            except Exception:  # noqa: BLE001 — a broken handler must not kill the server
                logger.exception("[%s] handler error for %s", self.name, request.path)
                return web.Response(status=500, text="internal error")
            location = None
            if len(result) == 3:
                status, html, location = result
            else:
                status, html = result
            if location:
                return web.Response(status=302, headers={"Location": location})
            return web.Response(
                status=int(status), text=html or "", content_type="text/html"
            )

        return _handle

    # -- platform lifecycle ---------------------------------------------------

    async def connect(self) -> bool:
        """Import the handler module and start the aiohttp listener."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False
        try:
            self._routes = self._load_routes()
        except Exception as exc:  # noqa: BLE001 — surface config errors, don't crash gateway
            logger.error("[%s] handler load failed: %s", self.name, exc)
            return False
        try:
            self._app = web.Application()
            for method, path, handler in self._routes:
                self._app.router.add_route(method, path, self._make_aiohttp_handler(handler))
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] listener start failed on %s:%s: %s",
                         self.name, self._host, self._port, exc)
            return False
        logger.info(
            "[%s] serving %d route(s) from %s on %s:%s",
            self.name, len(self._routes), self._handler_module, self._host, self._port,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        self._app = None
        self._runner = None
        self._site = None

    # -- messaging stubs (ingress-only platform: no chats, nothing to send) --

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return SendResult(success=False, error="http_callback platform cannot send messages")

    async def send_typing(self, chat_id: str) -> None:
        return None

    async def send_image(
        self, chat_id: str, image_url: str, caption: str = ""
    ) -> SendResult:
        return SendResult(success=False, error="http_callback platform cannot send images")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "http_callback", "type": "service", "chat_id": chat_id}
