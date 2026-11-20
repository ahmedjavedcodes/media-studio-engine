"""Designer & Layout Agent: the Image Grand Agent's second child. A STUB for now (built later:
text and layout layers on a document; image models never render text)."""

from __future__ import annotations

from media_engine.agents.stub import StubAgent


def designer_layout_agent() -> StubAgent:
    return StubAgent("design", "Design and layout is not available yet.")


__all__ = ["designer_layout_agent"]
