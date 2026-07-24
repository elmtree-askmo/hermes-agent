"""External-memory sync for short-circuited turns (Artemis B-0724-01).

The turn-intent short-circuit paths (multi lead-in, surface_existing) skip
``run_agent`` entirely, and ``run_agent`` holds the codebase's only
memory-sync call site — so a short-circuited turn's user message never
reached the external memory provider. High-signal turns (a user reporting an
interview while asking for team work) are precisely the ones that
multi-dispatch, so the drop was systematic and invisible: a missing memory is
indistinguishable from "the user never said it" at every consumer.

Per-turn side-effect inventory for the short-circuit path — who owns what
when Coach inference is skipped (keep this current; each line below was a
separately-discovered bug of the same "skip agent → lose a side effect"
class):

  - session transcript write   → gateway/run.py fallback branch writes the
    user message + a synthetic assistant entry (P-0612-03).
  - trace root + trace_index   → minted inside the short-circuit block
    (P-0721-01).
  - external memory sync       → THIS module, called from the short-circuit
    block (B-0724-01).
  - next-turn memory prefetch  → deliberately NOT replicated: gateway agents
    are rebuilt per turn, a queued prefetch dies with the instance; the mem0
    provider falls back to synchronous search (B-0722-01).
  - background memory/skill review → deliberately NOT replicated: periodic
    maintenance; the next normal turn covers it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def sync_short_circuit_turn(
    user_content: str,
    assistant_content: str,
    *,
    session_id: str,
    user_id: str,
    platform: str,
) -> bool:
    """Sync a short-circuited turn to the configured external memory provider.

    Mirrors run_agent's normal-turn sync: same provider resolution, same
    identity kwargs (so the write scope — ``user_id`` + the provider-config
    ``agent_id`` — is identical on both paths), same non-blocking provider
    semantics. ``assistant_content`` is the synthetic transcript text the
    short-circuit pushed to Slack — exactly what the session records.

    Returns True iff a sync was handed to a provider. Never raises.
    """
    if not user_content or not assistant_content:
        return False
    try:
        from hermes_cli.config import load_config

        provider_name = (load_config().get("memory") or {}).get("provider") or ""
        if not provider_name:
            return False

        import plugins.memory as _pm

        provider = _pm.load_memory_provider(provider_name)
        if not provider or not provider.is_available():
            return False

        from agent.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.add_provider(provider)
        if not manager.providers:
            return False

        init_kwargs = {
            "platform": platform or "cli",
            "agent_context": "primary",
        }
        if user_id:
            init_kwargs["user_id"] = user_id
        try:
            from hermes_cli.profiles import get_active_profile_name

            init_kwargs["agent_identity"] = get_active_profile_name()
            init_kwargs["agent_workspace"] = "hermes"
        except Exception:
            pass
        manager.initialize_all(session_id=session_id, **init_kwargs)
        manager.sync_all(user_content, assistant_content, session_id=session_id)
        return True
    except Exception as e:
        logger.warning("short-circuit memory sync failed: %s", e)
        return False
