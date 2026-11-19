"""Text/vision model calls for agents. Picks the gateway chain by agent and plan tier (from
agents.yaml via a resolver), then delegates to the fallback-chain Gateway. No LLM of its own.

Stateless and shared across runs: per-run accounting (tokens, neurons, call counts) travels in
the returned `LLMResult` -> the agent's `Spend` -> the run's plan, never on this object."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel

from media_engine.contracts import AuthContext, Plan
from media_engine.gateway.types import CallResult, LLMRequest, LLMResponse, ReasoningEffort

ChainResolver = Callable[[str, Plan], str]


class ChatBackend(Protocol):
    """What LLMGateway needs from the provider gateway (`media_engine.gateway.router.Gateway`)."""

    async def chat(self, task: str, request: LLMRequest) -> CallResult: ...


class LLMResult(BaseModel):
    text: str
    agent: str
    chain: str
    provider: str
    model: str
    tokens: int  # prompt + completion
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    neurons: float
    duration_ms: int
    is_fallback: bool


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def token_counts(usage: Mapping[str, Any]) -> tuple[int, int, int]:
    """(prompt, completion, total). Total falls back to prompt + completion."""
    prompt, completion = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens")) or prompt + completion
    return prompt, completion, total


class LLMGateway:
    def __init__(self, backend: ChatBackend, resolve_chain: ChainResolver) -> None:
        self._backend = backend
        self._resolve_chain = resolve_chain

    async def complete(
        self,
        agent: str,
        auth: AuthContext,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        json_mode: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> LLMResult:
        # TODO: let auth.model_demand ("cheap"/"best") shift the chain once tiers are priced.
        chain = self._resolve_chain(agent, auth.plan)
        request = LLMRequest(
            messages=messages,
            max_tokens=max_tokens,
            json_mode=json_mode,
            reasoning_effort=reasoning_effort,
        )
        started = time.perf_counter()
        result = await self._backend.chat(chain, request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        output = result.output
        if not isinstance(output, LLMResponse):  # pragma: no cover - chat always returns text
            raise TypeError(f"chat returned {type(output).__name__}, expected LLMResponse")
        prompt_tokens, completion_tokens, tokens = token_counts(output.usage)
        neurons = result.neurons if result.neurons is not None else (result.neurons_est or 0.0)
        return LLMResult(
            text=output.text,
            agent=agent,
            chain=chain,
            provider=result.provider,
            model=result.model,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=result.cost_usd or 0.0,
            neurons=neurons,
            duration_ms=duration_ms,
            is_fallback=result.is_fallback,
        )


def chain_resolver(policies: Mapping[str, Any]) -> ChainResolver:
    """Build a resolver from `{agent_name: AgentPolicy}` (anything with `chain_for(plan)`)."""

    def resolve(agent: str, plan: Plan) -> str:
        try:
            policy = policies[agent]
        except KeyError:
            raise KeyError(f"No model policy for agent {agent!r}") from None
        chain: str = policy.chain_for(plan)
        return chain

    return resolve
