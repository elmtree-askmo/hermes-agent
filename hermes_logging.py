"""Centralized logging setup for Hermes Agent.

Provides a single ``setup_logging()`` entry point that both the CLI and
gateway call early in their startup path.  All log files live under
``~/.hermes/logs/`` (profile-aware via ``get_hermes_home()``).

Log files produced:
    agent.log   — INFO+, all agent/tool/session activity (the main log)
    errors.log  — WARNING+, errors and warnings only (quick triage)

Both files use ``RotatingFileHandler`` with ``RedactingFormatter`` so
secrets are never written to disk.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

# Sentinel to track whether setup_logging() has already run.  The function
# is idempotent — calling it twice is safe but the second call is a no-op
# unless ``force=True``.
_logging_initialized = False

# Default log format — timestamp, level, optional run-scoped trace tag, logger
# name, message. ``trace_tag`` is ``" [<trace_id>]"`` during a run (the bracket
# form mirrors upstream's ``[session_id]`` convention) and ``""`` outside one,
# so background/startup lines stay clean instead of carrying a ``trace=-``.
_LOG_FORMAT = "%(asctime)s %(levelname)s%(trace_tag)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = "%(asctime)s - %(name)s - %(levelname)s%(trace_tag)s - %(message)s"

# Guard so the LogRecord factory is wrapped at most once (setup_logging is
# idempotent but may be re-invoked with force=True).
_trace_factory_installed = False


def _install_trace_record_factory() -> None:
    """Wrap the LogRecord factory so every record carries ``trace_tag`` from the
    session ContextVar (``""`` when no run is active).

    Done via the record factory rather than a per-handler ``Filter`` so the
    attribute is present on *every* record — including third-party and
    propagated ones — and ``%(trace_tag)s`` in the format can never KeyError.
    Idempotent.
    """
    global _trace_factory_installed
    if _trace_factory_installed:
        return

    old_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        try:
            from tools.session_context import get_trace_id
            tid = get_trace_id()
        except Exception:
            tid = None
        # Subprocess fallback: spawned Strategist/Executor have no inbound
        # session so the ContextVar is unset, but the spawn path injects
        # HERMES_TRACE_ID into their env — read it so their logs join the run.
        if not tid:
            tid = os.environ.get("HERMES_TRACE_ID") or None
        # Bracketed tag (leading space), omitted entirely when there's no run —
        # so only run-scoped lines carry ``[<trace_id>]``; background lines stay
        # clean. Mirrors upstream hermes_logging's ``session_tag`` convention.
        record.trace_tag = f" [{tid}]" if tid else ""
        return record

    logging.setLogRecordFactory(_factory)
    _trace_factory_installed = True

# Third-party loggers that are noisy at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
    "markdown_it",
)


def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Configure the Hermes logging subsystem.

    Safe to call multiple times — the second call is a no-op unless
    *force* is ``True``.

    Parameters
    ----------
    hermes_home
        Override for the Hermes home directory.  Falls back to
        ``get_hermes_home()`` (profile-aware).
    log_level
        Minimum level for the ``agent.log`` file handler.  Accepts any
        standard Python level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        Defaults to ``"INFO"`` or the value from config.yaml ``logging.level``.
    max_size_mb
        Maximum size of each log file in megabytes before rotation.
        Defaults to 5 or the value from config.yaml ``logging.max_size_mb``.
    backup_count
        Number of rotated backup files to keep.
        Defaults to 3 or the value from config.yaml ``logging.backup_count``.
    mode
        Hint for the caller context: ``"cli"``, ``"gateway"``, ``"cron"``.
        Currently used only for log format tuning (gateway includes PID).
    force
        Re-run setup even if it has already been called.

    Returns
    -------
    Path
        The ``logs/`` directory where files are written.
    """
    global _logging_initialized
    if _logging_initialized and not force:
        home = hermes_home or get_hermes_home()
        return home / "logs"

    home = hermes_home or get_hermes_home()
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Read config defaults (best-effort — config may not be loaded yet).
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # Lazy import to avoid circular dependency at module load time.
    from agent.redact import RedactingFormatter

    # Stamp every record with trace_tag before any handler uses the format that
    # references it (otherwise %(trace_tag)s would KeyError).
    _install_trace_record_factory()

    root = logging.getLogger()

    # --- agent.log (INFO+) — the main activity log -------------------------
    _add_rotating_handler(
        root,
        log_dir / "agent.log",
        level=level,
        max_bytes=max_bytes,
        backup_count=backups,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- errors.log (WARNING+) — quick triage log --------------------------
    _add_rotating_handler(
        root,
        log_dir / "errors.log",
        level=logging.WARNING,
        max_bytes=2 * 1024 * 1024,
        backup_count=2,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # Ensure root logger level is low enough for the handlers to fire.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # Suppress noisy third-party loggers.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_initialized = True
    return log_dir


def setup_verbose_logging() -> None:
    """Enable DEBUG-level console logging for ``--verbose`` / ``-v`` mode.

    Called by ``AIAgent.__init__()`` when ``verbose_logging=True``.
    """
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # Avoid adding duplicate stream handlers.
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            if getattr(h, "_hermes_verbose", False):
                return

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(RedactingFormatter(_LOG_FORMAT_VERBOSE, datefmt="%H:%M:%S"))
    handler._hermes_verbose = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Lower root logger level so DEBUG records reach all handlers.
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    # Keep third-party libraries at WARNING to reduce noise.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # rex-deploy at INFO for sandbox status.
    logging.getLogger("rex-deploy").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_rotating_handler(
    logger: logging.Logger,
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
) -> None:
    """Add a ``RotatingFileHandler`` to *logger*, skipping if one already
    exists for the same resolved file path (idempotent).
    """
    resolved = path.resolve()
    for existing in logger.handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return  # already attached

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _read_logging_config():
    """Best-effort read of ``logging.*`` from config.yaml.

    Returns ``(level, max_size_mb, backup_count)`` — any may be ``None``.
    """
    try:
        import yaml
        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            log_cfg = cfg.get("logging", {})
            if isinstance(log_cfg, dict):
                return (
                    log_cfg.get("level"),
                    log_cfg.get("max_size_mb"),
                    log_cfg.get("backup_count"),
                )
    except Exception:
        pass
    return (None, None, None)
