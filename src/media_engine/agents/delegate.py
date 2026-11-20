"""The delegation contract every agent on the image side honours.

Every delegation passes Goal + StoppingCondition + AuthContext + inputs and gets one Envelope
back. LLM agents implement it through `agents.base.Agent`; the coordinators that make no model
call (the Image Grand Agent, the Image Generation Agent, the stubs) implement it directly.
"""

from __future__ import annotations

from typing import Any, Protocol

from media_engine.contracts import AuthContext, Envelope, Goal, StoppingCondition


class Delegate(Protocol):
    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope: ...


__all__ = ["Delegate"]
