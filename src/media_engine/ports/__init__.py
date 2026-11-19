"""Backend seams. Agents depend on these Protocols only; the backend repo supplies real
implementations. The local ones here make the harness runnable from a plain script."""

from media_engine.ports.artifact_store import ArtifactNotFound, ArtifactStore, LocalArtifactStore
from media_engine.ports.event_sink import Event, EventSink, ListEventSink, make_event, safe_emit
from media_engine.ports.run_store import InMemoryRunStore, RunStore

__all__ = [
    "ArtifactNotFound",
    "ArtifactStore",
    "Event",
    "EventSink",
    "InMemoryRunStore",
    "ListEventSink",
    "LocalArtifactStore",
    "RunStore",
    "make_event",
    "safe_emit",
]
