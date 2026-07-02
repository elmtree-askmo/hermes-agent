"""Tests for the generic HTTP callback platform adapter.

Covers: handler-module loading + validation, request dispatch (html / 302 /
handler exception / unknown path), lifecycle start-stop, and env-driven
platform enablement.
"""

import textwrap

import pytest
import pytest_asyncio
from aiohttp import ClientSession

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.http_callback import (
    HTTPCallbackAdapter,
    check_http_callback_requirements,
)


def _write_handler_module(tmp_path, body: str, name: str = "cb_handlers"):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    return str(tmp_path), name


def _adapter(handler_dir: str, handler_module: str, port: int = 0, **extra):
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "handler_dir": handler_dir,
        "handler_module": handler_module,
        "host": "127.0.0.1",
        "port": port,
        **extra,
    }
    return HTTPCallbackAdapter(config)


def test_check_requirements():
    assert check_http_callback_requirements() is True


# ---------------------------------------------------------------------------
# handler loading / validation
# ---------------------------------------------------------------------------


def test_load_routes_missing_module(tmp_path):
    adapter = _adapter(str(tmp_path), "does_not_exist_xyz")
    with pytest.raises(ModuleNotFoundError):
        adapter._load_routes()


def test_load_routes_missing_http_routes(tmp_path):
    d, mod = _write_handler_module(tmp_path, "X = 1\n", name="cb_no_routes")
    adapter = _adapter(d, mod)
    with pytest.raises(ValueError, match="HTTP_ROUTES"):
        adapter._load_routes()


def test_load_routes_rejects_bad_entries(tmp_path):
    d, mod = _write_handler_module(
        tmp_path,
        """
        def h(q):
            return 200, "ok"
        HTTP_ROUTES = [("DELETE", "/x", h)]
        """,
        name="cb_bad_method",
    )
    adapter = _adapter(d, mod)
    with pytest.raises(ValueError, match="unsupported method"):
        adapter._load_routes()


@pytest.mark.asyncio
async def test_connect_returns_false_on_broken_handler(tmp_path):
    adapter = _adapter(str(tmp_path), "also_missing_xyz")
    assert await adapter.connect() is False


# ---------------------------------------------------------------------------
# request dispatch (live listener on an ephemeral port)
# ---------------------------------------------------------------------------


HANDLERS_SRC = """
def ok(query_string):
    return 200, "<html>hello " + query_string + "</html>"

def redirect(query_string):
    if "t=good" in query_string:
        return 302, "", "https://example.com/authorize?state=abc"
    return 400, "<html>bad token</html>", None

def boom(query_string):
    raise RuntimeError("handler exploded")

HTTP_ROUTES = [
    ("GET", "/cb/ok", ok),
    ("GET", "/cb/start", redirect),
    ("GET", "/cb/boom", boom),
]
"""


@pytest_asyncio.fixture
async def live_adapter(tmp_path):
    d, mod = _write_handler_module(tmp_path, HANDLERS_SRC, name="cb_live")
    adapter = _adapter(d, mod)
    assert await adapter.connect() is True
    # ephemeral port: read back what the OS assigned
    port = adapter._site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    yield adapter, f"http://127.0.0.1:{port}"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_dispatch_html(live_adapter):
    _adapter_, base = live_adapter
    async with ClientSession() as session:
        async with session.get(f"{base}/cb/ok?x=1") as resp:
            assert resp.status == 200
            assert "hello x=1" in await resp.text()


@pytest.mark.asyncio
async def test_dispatch_302_redirect(live_adapter):
    _adapter_, base = live_adapter
    async with ClientSession() as session:
        async with session.get(f"{base}/cb/start?t=good", allow_redirects=False) as resp:
            assert resp.status == 302
            assert resp.headers["Location"] == "https://example.com/authorize?state=abc"


@pytest.mark.asyncio
async def test_dispatch_3tuple_without_location_is_plain_response(live_adapter):
    _adapter_, base = live_adapter
    async with ClientSession() as session:
        async with session.get(f"{base}/cb/start?t=bad", allow_redirects=False) as resp:
            assert resp.status == 400
            assert "bad token" in await resp.text()


@pytest.mark.asyncio
async def test_handler_exception_maps_to_500(live_adapter):
    _adapter_, base = live_adapter
    async with ClientSession() as session:
        async with session.get(f"{base}/cb/boom") as resp:
            assert resp.status == 500


@pytest.mark.asyncio
async def test_unknown_path_404(live_adapter):
    _adapter_, base = live_adapter
    async with ClientSession() as session:
        async with session.get(f"{base}/nope") as resp:
            assert resp.status == 404


# ---------------------------------------------------------------------------
# env-driven enablement
# ---------------------------------------------------------------------------


def test_env_enables_platform(monkeypatch, tmp_path):
    from gateway import config as config_mod

    monkeypatch.setenv("HERMES_HTTP_CALLBACK_HANDLER_MODULE", "cb_handlers")
    monkeypatch.setenv("HERMES_HTTP_CALLBACK_HANDLER_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_HTTP_CALLBACK_PORT", "8123")
    cfg = config_mod.load_gateway_config()
    platform_cfg = cfg.platforms.get(Platform.HTTP_CALLBACK)
    assert platform_cfg is not None and platform_cfg.enabled
    assert platform_cfg.extra["handler_module"] == "cb_handlers"
    assert platform_cfg.extra["handler_dir"] == str(tmp_path)
    assert platform_cfg.extra["port"] == 8123
    assert Platform.HTTP_CALLBACK in cfg.get_connected_platforms()


def test_no_env_no_platform(monkeypatch):
    from gateway import config as config_mod

    monkeypatch.delenv("HERMES_HTTP_CALLBACK_HANDLER_MODULE", raising=False)
    cfg = config_mod.load_gateway_config()
    assert Platform.HTTP_CALLBACK not in cfg.get_connected_platforms()
