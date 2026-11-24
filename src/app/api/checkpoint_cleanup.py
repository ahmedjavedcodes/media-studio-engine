"""Checkpoint cleanup. A run that ended (done, failed, expired) has no use for its saved graph
state, and an abandoned question would keep it forever. Two parts:

- `forget_thread` is called by the run service when a run ends.
- `sweep_old_threads` runs at startup and deletes threads whose newest checkpoint is older than
  24 hours (a run that was waiting when the server stopped, for example).
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

log = structlog.get_logger("checkpoints")

MAX_AGE = timedelta(hours=24)


async def forget_thread(checkpointer: Any, thread_id: str) -> None:
    """Delete one run's checkpoints. A failure here is logged, never raised."""
    delete = getattr(checkpointer, "adelete_thread", None)
    if delete is None:
        return
    try:
        await delete(thread_id)
    except Exception as exc:  # noqa: BLE001 - cleanup must never break a run
        log.warning("checkpoint_delete_failed", thread=thread_id, error=type(exc).__name__)


async def sweep_old_threads(
    checkpointer: Any, *, max_age: timedelta = MAX_AGE, now: datetime | None = None
) -> int:
    """Delete threads whose newest checkpoint is older than `max_age`. Returns how many."""
    now = now or datetime.now(UTC)
    newest: dict[str, datetime] = {}
    async for tup in checkpointer.alist(None):
        thread = (tup.config.get("configurable") or {}).get("thread_id")
        ts = tup.checkpoint.get("ts")
        if not thread or not ts:
            continue
        stamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if thread not in newest or stamp > newest[thread]:
            newest[thread] = stamp
    old = [t for t, stamp in newest.items() if now - stamp > max_age]
    for thread in old:
        await forget_thread(checkpointer, thread)
    if old:
        log.info("checkpoints_swept", threads=len(old))
    return len(old)
