"""Grand Agent stub (spec §8.5). The multi-step planner is a non-goal of this build: a request
the router calls `complex` is handed back as ESCALATE with one fixed reason, and nothing else.
No LLM call, no planning, no images.

TODO(backend): the real Grand Agent (planning, replanning, a team of step agents) is built
later; this class is only the seam the router's `complex` route lands on.
"""

from __future__ import annotations

from typing import Any

from media_engine.agents.base import Agent
from media_engine.contracts import AuthContext, Envelope, Goal, Spend, Status, StoppingCondition

ESCALATE_REASON = "multi-step planning not implemented"
SUMMARY = "This request needs more than one step, which is not available yet."


class GrandAgent(Agent):
    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        return self.envelope(Status.ESCALATE, {}, SUMMARY, Spend(), reason=ESCALATE_REASON)
