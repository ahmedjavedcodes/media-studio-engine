"""EventSink port. Events are fire-and-forget: a failed event write never breaks generation."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from media_engine.core.logging import get_logger
from media_engine.harness.run_state import PublicLabel

# `image_ready`: one image passed the Critic's safety check and can be shown NOW, before the run
# ends (IMAGE-CREATE-OPTIMIZE §5.3: stream each image as it finishes). Never sent for an image
# that is unchecked or blocked.
Phase = Literal["started", "progress", "finished", "image_ready"]

log = get_logger("events")


class Event(BaseModel):
    """The only event shape. Carries the public label, never an internal agent name."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    step: int
    public_label: PublicLabel
    phase: Phase
    ts: str
    image_id: str | None = None  # only on `image_ready`: an id, never bytes or an internal name


def make_event(
    run_id: str, step: int, public_label: PublicLabel, phase: Phase, image_id: str | None = None
) -> Event:
    if (phase == "image_ready") != (image_id is not None):
        raise ValueError("image_id is set exactly when phase is image_ready")
    return Event(
        run_id=run_id,
        step=step,
        public_label=public_label,
        phase=phase,
        ts=datetime.now(UTC).isoformat(),
        image_id=image_id,
    )


@runtime_checkable
class EventSink(Protocol):
    def emit(self, run_id: str, event: dict[str, Any]) -> None: ...  # sync, fire-and-forget


def safe_emit(sink: EventSink, event: Event) -> None:
    """The only way harness code emits. Takes a typed `Event` (so the public-label rule holds
    by construction), serialises it here, and swallows anything the sink raises.

    Passing something that is not an `Event` is a programming error, not a sink failure, so it
    raises (before the swallow) instead of silently dropping the event."""
    if not isinstance(event, Event):
        raise TypeError(f"safe_emit takes an Event, got {type(event).__name__}; use make_event()")
    try:
        # exclude_none: an ordinary event keeps its exact public shape; only `image_ready`
        # carries `image_id`.
        sink.emit(event.run_id, event.model_dump(exclude_none=True))
    except Exception:  # noqa: BLE001 - by contract an event failure must never surface
        with contextlib.suppress(Exception):
            log.warning("events.emit_failed", run_id=event.run_id)


class ListEventSink:
    """Local implementation: keeps events in memory.

    TODO(backend): SSE/queue transport lives in the backend repo behind this Protocol."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        self.events.append((run_id, dict(event)))
