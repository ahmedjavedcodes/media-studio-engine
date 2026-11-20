"""A placeholder agent: no model call, no image. It ends the request with ESCALATE and its own
reason, so the caller can tell which part is not built yet. Used for edit, merge, restyle and
the Designer & Layout Agent until they exist."""

from __future__ import annotations

from typing import Any

from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Spend,
    Status,
    StoppingCondition,
)


class StubAgent:
    def __init__(self, name: str, summary: str) -> None:
        self.name = name
        self.reason = f"{name} not implemented"
        self.summary = summary

    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        return Envelope(
            output={"run_id": inputs.get("run_id"), "image_ids": [], "billable": False},
            summary=self.summary,
            state=EnvelopeState(status=Status.ESCALATE, spend=Spend(), reason=self.reason),
        )


__all__ = ["StubAgent"]
