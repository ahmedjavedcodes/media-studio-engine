"""Send an event to the SSE stream from inside a graph node."""

from langgraph.config import get_stream_writer

from app.contracts.events import RunEvent


def emit(event: RunEvent) -> None:
    """A no-op when nobody is streaming (a plain `ainvoke`)."""
    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer(event.model_dump(mode="json", exclude_none=True))
