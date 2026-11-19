"""EventSink port. Events are fire-and-forget: a failed event write never breaks generation."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from media_engine.core.logging import get_logger
from media_engine.harness.run_state import PublicLabel

Phase = Literal["started", "progress", "finished"]

log = get_logger("events")


class Event(BaseModel):
    """The only event shape. Carries the public label, never an internal agent name."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    step: int
    public_label: PublicLabel
    phase: Phase
    ts: str


def make_event(run_id: str, step: int, public_label: PublicLabel, phase: Phase) -> dict[str, Any]:
    return Event(
        run_id=run_id,
        step=step,
        public_label=public_label,
        phase=phase,
        ts=datetime.now(UTC).isoformat(),
    ).model_dump()


@runtime_checkable
class EventSink(Protocol):
    def emit(self, run_id: str, event: dict[str, Any]) -> None: ...  # sync, fire-and-forget


def safe_emit(sink: EventSink, run_id: str, event: dict[str, Any]) -> None:
    """Call the sink and swallow anything it raises. Harness code emits only through this."""
    try:
        sink.emit(run_id, event)
    except Exception:  # noqa: BLE001 - by contract an event failure must never surface
        with contextlib.suppress(Exception):
            log.warning("events.emit_failed", run_id=run_id)


class ListEventSink:
    """Local implementation: keeps events in memory.

    TODO(backend): SSE/queue transport lives in the backend repo behind this Protocol."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        self.events.append((run_id, dict(event)))
