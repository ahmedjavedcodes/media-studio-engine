"""Complexity router: one small, fast structured call that decides how a request is handled.

inputs:
    prompt: str   the user's raw request (required)

output (envelope): {"route": "simple" | "complex"}. `simple` goes straight to the Image Grand
Agent; `complex` goes to the Grand Agent stub (spec §7.2).

The router never fails a run. If its call fails (or the reply is unusable even after the one
repair) the request is treated as `simple`: the supervisor can always make one image, and a
classifier outage must not block image creation. The envelope is DONE, with `defaulted` set and
the real spend, so the fallback is visible to the caller and in the logs.
"""

from __future__ import annotations

from typing import Any

from media_engine.agents.base import Agent
from media_engine.contracts import (
    AuthContext,
    Envelope,
    Goal,
    Spend,
    Status,
    StoppingCondition,
)
from media_engine.core.logging import get_logger
from media_engine.core.prompts import PROMPTS_DIR, load_prompt, render, tag_stripper
from media_engine.schemas import RouteDecision

INPUT_TEMPLATE = PROMPTS_DIR / "image_create" / "router.input.md"
SUMMARY = "Worked out how to handle the request."
_TAG = tag_stripper("user_request")

log = get_logger("agents.router")


class Router(Agent):
    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        raw = inputs.get("prompt")
        prompt = _TAG.sub("", raw).strip() if isinstance(raw, str) else ""
        if not prompt:
            return self.failed("no prompt given", Spend(), "invalid_input")

        content = render(load_prompt(INPUT_TEMPLATE), prompt=prompt)
        result = await self.call_structured(auth, stop, content, RouteDecision)
        if result.output is None:
            log.warning("router.defaulted_to_simple", failure=result.failure, error=result.error)
            return self.envelope(
                Status.DONE, {"route": "simple", "defaulted": True}, SUMMARY, result.spend
            )
        return self.envelope(Status.DONE, {"route": result.output.route}, SUMMARY, result.spend)
