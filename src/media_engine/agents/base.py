"""Agent base class. One class, built from config.

Every LLM output is parsed into the agent's `output_model`. On a validation failure the agent
makes ONE repair attempt (the validation error is handed back to the model), then gives up with
`Status.FAILED`. The repair draws from the step's existing token budget; it never extends it.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from media_engine.agents.config import AgentPolicy
from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Spend,
    Status,
    StoppingCondition,
)
from media_engine.core.logging import get_logger
from media_engine.core.prompts import PROMPTS_DIR, load_prompt, render
from media_engine.gateway.errors import (
    AllProvidersFailed,
    BadRequestError,
    ContentRefused,
    DeadlineExceeded,
    GatewayError,
)
from media_engine.gateway.llm_gateway import LLMGateway, LLMResult
from media_engine.ports.event_sink import EventSink
from media_engine.schemas import FailureKind

REPAIR_PROMPT = PROMPTS_DIR / "shared" / "repair.md"
FAILED_SUMMARY = "This step could not be completed."
# Heuristic (~4 chars/token for English + JSON) used only when a provider omits usage counts.
CHARS_PER_TOKEN = 4


log = get_logger("agents")

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    prompt_file: Traversable | None  # a pathlib.Path also qualifies
    model_policy: AgentPolicy
    tools: tuple[str, ...] = ()
    output_model: type[BaseModel] | None = None  # None only for an agent with no LLM

    @classmethod
    def from_policy(
        cls,
        name: str,
        policy: AgentPolicy,
        output_model: type[BaseModel] | None,
        tools: tuple[str, ...] = (),
    ) -> AgentConfig:
        return cls(
            name=name,
            prompt_file=policy.prompt_path() if policy.prompt else None,
            model_policy=policy,
            tools=tools,
            output_model=output_model,
        )


@dataclass
class Structured[T: BaseModel]:
    """Outcome of `Agent.call_structured`: a validated output, or the reason there is none."""

    output: T | None
    spend: Spend
    error: str | None = None
    failure: FailureKind | None = None  # set whenever output is None
    calls: list[LLMResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.output is not None


def estimate_tokens(content: object) -> int:
    """Rough token count for budgeting when the provider did not report usage."""
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return -(-len(text) // CHARS_PER_TOKEN)


def extract_json(text: str) -> str | None:
    """Pull the first decodable JSON object out of a model reply. Tolerates code fences and
    surrounding prose, including stray braces in the prose ("Here is {the answer}: {...}")."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            start = stripped.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return stripped[start:end]
        start = stripped.find("{", end)
    return None


def gateway_failure(err: GatewayError) -> FailureKind:
    if isinstance(err, ContentRefused):
        return "content_refused"
    if isinstance(err, AllProvidersFailed):
        return "no_provider"
    if isinstance(err, DeadlineExceeded):
        return "deadline"
    if isinstance(err, BadRequestError):
        return "bad_request"
    return "gateway_error"


def describe_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors(include_url=False, include_input=False):
        loc = ".".join(str(p) for p in item["loc"]) or "(root)"
        lines.append(f"- {loc}: {item['msg']}")
    return "\n".join(lines)


def parse_output[T: BaseModel](text: str, model: type[T]) -> tuple[T | None, str | None]:
    candidate = extract_json(text)
    if candidate is None:
        return None, "- (root): no JSON object found in the reply"
    try:
        return model.model_validate_json(candidate), None
    except ValidationError as err:
        return None, describe_validation_error(err)


class Agent(ABC):
    name: str

    def __init__(self, config: AgentConfig, gateway: LLMGateway | None, events: EventSink) -> None:
        self.config = config
        self.name = config.name
        self._gateway = gateway
        self._events = events

    @abstractmethod
    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope: ...

    # -- prompts ----------------------------------------------------------------------------

    def schema_json(self, model: type[BaseModel] | None = None) -> str:
        target = model or self.config.output_model
        if target is None:
            raise ValueError(f"Agent {self.name!r} has no output_model")
        return json.dumps(target.model_json_schema(), separators=(",", ":"))

    def system_prompt(self) -> str:
        if self.config.prompt_file is None:
            raise ValueError(f"Agent {self.name!r} has no prompt file")
        return render(load_prompt(self.config.prompt_file), schema=self.schema_json())

    # -- structured LLM call with one repair ------------------------------------------------

    async def call_structured[T: BaseModel](
        self,
        auth: AuthContext,
        stop: StoppingCondition,
        user_content: str | list[dict[str, Any]],
        output_model: type[T],
    ) -> Structured[T]:
        if self._gateway is None:
            raise RuntimeError(f"Agent {self.name!r} has no LLM gateway")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_content},
        ]
        result: Structured[T] = Structured(output=None, spend=Spend())
        started = time.perf_counter()
        # `deadline_ms` is a deadline for the whole step: a slow first call leaves less time
        # for the repair, by design. asyncio.CancelledError is a BaseException and still
        # propagates, so run cancellation is never swallowed here.
        try:
            async with asyncio.timeout(stop.deadline_ms / 1000):
                await self._attempt(auth, stop, messages, output_model, result)
        except TimeoutError:
            # The in-flight call's tokens are unknown (the provider may still bill them);
            # llm_calls already counts it.
            result.error = f"deadline of {stop.deadline_ms} ms exceeded (in-flight spend unknown)"
            result.failure = "deadline"
        except GatewayError as err:
            result.error = f"model call failed: {type(err).__name__}"
            result.failure = gateway_failure(err)
            if isinstance(err, ContentRefused):
                # A refusal that came back as HTTP 200 was billed: charge it to the step.
                result.spend = result.spend.plus(
                    Spend(neurons=err.neurons or 0.0, cost_usd=err.cost_usd or 0.0)
                )
            log.warning("agent.gateway_error", agent=self.name, error=str(err)[:300])
        except Exception as err:
            result.error = f"internal error: {type(err).__name__}"
            result.failure = "internal"
            log.exception("agent.internal_error", agent=self.name)
        if result.output is None and result.failure is None:
            result.failure = "invalid_output"
        result.spend.duration_ms = int((time.perf_counter() - started) * 1000)
        if not result.ok:
            log.warning(
                "agent.structured_failed",
                agent=self.name,
                calls=len(result.calls),
                tokens=result.spend.tokens,
                error=result.error,
            )
        return result

    async def _attempt[T: BaseModel](
        self,
        auth: AuthContext,
        stop: StoppingCondition,
        messages: list[dict[str, Any]],
        output_model: type[T],
        result: Structured[T],
    ) -> None:
        # Two quantities, kept apart: stop.max_tokens is the step's TOTAL budget (prompt +
        # completion, all calls); stop.max_completion_tokens is the provider cap on one reply.
        first_cap = min(stop.max_completion_tokens, stop.max_tokens)
        first = await self._complete(auth, messages, first_cap, result)
        parsed, error = parse_output(first.text, output_model)
        if parsed is not None:
            result.output = parsed
            return

        repair = render(
            load_prompt(REPAIR_PROMPT), error=error or "", schema=self.schema_json(output_model)
        )
        repair_messages = [
            *messages,
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": repair},
        ]
        # The repair call re-sends everything, so its prompt is the first prompt + the first
        # reply + the repair instruction. Budget for that before deciding to repair.
        est_prompt = (
            (first.prompt_tokens or estimate_tokens(messages))
            + (first.completion_tokens or estimate_tokens(first.text))
            + estimate_tokens(repair)
        )
        remaining = stop.max_tokens - result.spend.tokens  # total vs total
        repair_cap = min(stop.max_completion_tokens, remaining - est_prompt)
        # Repair only if it can afford a reply at least as long as the first one (a truncated
        # first reply is a common cause of invalid JSON, so a shorter cap would just repeat it).
        needed = min(stop.max_completion_tokens, max(first.completion_tokens, 1))
        if repair_cap < needed:
            result.error = (
                f"invalid output and no token budget left to repair it "
                f"(remaining {remaining}, repair prompt ~{est_prompt}):\n{error}"
            )
            return

        log.info(
            "agent.repair",
            agent=self.name,
            remaining_tokens=remaining,
            est_prompt_tokens=est_prompt,
            completion_cap=repair_cap,
        )
        second = await self._complete(auth, repair_messages, repair_cap, result)
        if result.spend.tokens > stop.max_tokens:
            log.warning(
                "agent.token_budget_overrun",
                agent=self.name,
                spent=result.spend.tokens,
                budget=stop.max_tokens,
            )
        parsed, error = parse_output(second.text, output_model)
        if parsed is not None:
            result.output = parsed
        else:
            result.error = f"invalid output after one repair attempt:\n{error}"

    async def _complete[T: BaseModel](
        self,
        auth: AuthContext,
        messages: list[dict[str, Any]],
        max_tokens: int,
        result: Structured[T],
    ) -> LLMResult:
        if self._gateway is None:  # pragma: no cover - checked in call_structured
            raise RuntimeError(f"Agent {self.name!r} has no LLM gateway")
        # Count the call before awaiting it, so a call cut off by the deadline is still counted.
        result.spend = result.spend.plus(Spend(llm_calls=1))
        reply = await self._gateway.complete(
            self.name,
            auth,
            messages,
            max_tokens=max_tokens,
            json_mode=True,
            reasoning_effort=self.config.model_policy.reasoning_effort,
        )
        result.calls.append(reply)
        result.spend = result.spend.plus(
            Spend(tokens=reply.tokens, cost_usd=reply.cost_usd, neurons=reply.neurons)
        )
        return reply

    # -- envelopes --------------------------------------------------------------------------

    @staticmethod
    def envelope(
        status: Status,
        output: dict[str, Any],
        summary: str,
        spend: Spend,
        reason: str | None = None,
    ) -> Envelope:
        return Envelope(
            output=output,
            summary=summary,
            state=EnvelopeState(status=status, spend=spend, reason=reason),
        )

    def failed(self, reason: str, spend: Spend, failure: FailureKind | None = None) -> Envelope:
        """FAILED envelope. `failure` goes in `output["failure"]` for the supervisor."""
        output: dict[str, Any] = {"failure": failure} if failure else {}
        return self.envelope(Status.FAILED, output, FAILED_SUMMARY, spend, reason)
