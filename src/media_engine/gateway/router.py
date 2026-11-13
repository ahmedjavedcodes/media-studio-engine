"""The gateway: the only door for model calls. Walks a fallback chain and logs every step."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import yaml
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

from media_engine.core.config import Settings, get_settings
from media_engine.core.logging import get_logger
from media_engine.core.telemetry import traced
from media_engine.gateway.budget import DailyCostMeter, DailyNeuronMeter, estimate_image_neurons
from media_engine.gateway.errors import AllProvidersFailed, BadRequestError, ProviderError
from media_engine.gateway.fallback import (
    MAX_ATTEMPTS,
    Action,
    AvailabilityState,
    decide,
)
from media_engine.gateway.providers import Provider, build_providers
from media_engine.gateway.registry import Kind, card_or_default
from media_engine.gateway.types import (
    Attempt,
    CallResult,
    ImageRequest,
    ImageResponse,
    LLMRequest,
    LLMResponse,
    ModelRef,
)

CHAINS_PATH = Path(__file__).with_name("chains.yaml")

log = get_logger("gateway")


def parse_ref(entry: str) -> ModelRef:
    provider, _, model = entry.partition("/")
    if not provider or not model:
        raise ValueError(f"Bad chain entry {entry!r}; expected '<provider>/<model>'")
    return ModelRef(provider, model)


def load_chains(path: Path = CHAINS_PATH) -> dict[str, list[ModelRef]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {task: [parse_ref(e) for e in entries] for task, entries in raw.items()}


def task_kind(task: str) -> Kind:
    return "image" if task.startswith("image_") else "llm"


def _safe_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"task": inputs.get("task")}  # never send prompts, images or keys to traces


def _safe_outputs(result: CallResult) -> dict[str, Any]:
    return result.summary()


class Gateway:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chains: dict[str, list[ModelRef]] | None = None,
        providers: dict[str, Provider] | None = None,
        state: AvailabilityState | None = None,
        meter: DailyNeuronMeter | None = None,
        cost_meter: DailyCostMeter | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self.providers = providers or build_providers(self.settings, self._client)
        self.chains = chains if chains is not None else load_chains()
        self.state = state or AvailabilityState()
        self.meter = meter or DailyNeuronMeter()
        self.cost_meter = cost_meter or DailyCostMeter()
        self._sleep = sleep

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- public API -------------------------------------------------------------------------

    @traced("gateway.chat", process_inputs=_safe_inputs, process_outputs=_safe_outputs)
    async def chat(self, task: str, request: LLMRequest) -> CallResult:
        return await self._run(task, "llm", lambda p, m: p.chat(m, request))

    @traced("gateway.image", process_inputs=_safe_inputs, process_outputs=_safe_outputs)
    async def image(self, task: str, request: ImageRequest) -> CallResult:
        return await self._run(
            task,
            "image",
            lambda p, m: p.generate_image(m, request),
            estimate=lambda ref: estimate_image_neurons(
                card_or_default(ref.provider, ref.model, "image"), request
            ),
        )

    # -- internals --------------------------------------------------------------------------

    async def _run(
        self,
        task: str,
        kind: Kind,
        invoke: Callable[[Provider, str], Awaitable[LLMResponse | ImageResponse]],
        estimate: Callable[[ModelRef], float | None] | None = None,
    ) -> CallResult:
        if task not in self.chains:
            raise ValueError(f"Unknown task {task!r}; known: {sorted(self.chains)}")
        if task_kind(task) != kind:
            raise ValueError(f"Task {task!r} is a {task_kind(task)} task, not {kind}")

        chain = self.chains[task]
        attempts: list[Attempt] = []

        for index, ref in enumerate(chain):
            skip = self._skip_reason(ref)
            if skip is not None:
                attempts.append(Attempt(ref.provider, ref.model, "skipped", reason=skip))
                log.info(
                    "gateway.skip", task=task, provider=ref.provider, model=ref.model, reason=skip
                )
                continue

            provider = self.providers[ref.provider]
            started = time.perf_counter()
            calls = 0

            async def call(provider: Provider = provider, ref: ModelRef = ref) -> Any:
                nonlocal calls
                calls += 1
                return await invoke(provider, ref.model)

            try:
                output = await self._with_retry(call, ref)
            except ProviderError as err:
                decision = decide(err, calls, self.state.now())
                self.state.apply(ref.provider, ref.model, decision)
                latency_ms = (time.perf_counter() - started) * 1000
                attempts.append(
                    Attempt(
                        ref.provider,
                        ref.model,
                        "failed",
                        reason=decision.reason,
                        status=err.status,
                        code=err.code,
                        calls=calls,
                    )
                )
                fields: dict[str, Any] = {
                    "task": task,
                    "provider": ref.provider,
                    "model": ref.model,
                    "reason": decision.reason,
                    "status": err.status,
                    "code": err.code,
                    "calls": calls,
                    "latency_ms": round(latency_ms, 1),
                    "message": err.message[:200],
                }
                if decision.action is Action.RAISE:
                    log.error("gateway.bad_request", **fields)
                    raise BadRequestError(err) from err
                getattr(log, decision.severity)(
                    "gateway.fallback",
                    next_action="next_provider",
                    exhausted_until=decision.exhausted_until.isoformat()
                    if decision.exhausted_until
                    else None,
                    model_disabled=decision.disable_model,
                    **fields,
                )
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            attempts.append(Attempt(ref.provider, ref.model, "success", calls=calls))
            neurons: float | None = None
            neurons_est: float | None = None
            daily_total: float | None = None
            if ref.provider == "cloudflare":
                if isinstance(output, LLMResponse):
                    raw = output.usage.get("neurons")
                    neurons = float(raw) if isinstance(raw, int | float) else None
                if neurons is None and estimate is not None:
                    neurons_est = estimate(ref)
                daily_total = self.meter.add(actual=neurons, estimate=neurons_est)
            cost_usd = output.cost_usd
            daily_cost = self.cost_meter.add(cost_usd) if cost_usd is not None else None
            result = CallResult(
                output=output,
                task=task,
                provider=ref.provider,
                model=ref.model,
                latency_ms=latency_ms,
                is_fallback=index > 0,
                neurons=neurons,
                neurons_est=neurons_est,
                daily_neurons_total=daily_total,
                cost_usd=cost_usd,
                daily_cost_usd_total=daily_cost,
                attempts=attempts,
            )
            usage = output.usage if isinstance(output, LLMResponse) else {}
            log.info(
                "gateway.call",
                task=task,
                provider=ref.provider,
                model=ref.model,
                latency_ms=round(latency_ms, 1),
                calls=calls,
                is_fallback=result.is_fallback,
                neurons=neurons,
                neurons_est=neurons_est,
                daily_neurons_total=daily_total,
                cost_usd=cost_usd,
                daily_cost_usd_total=daily_cost,
                usage=usage,
            )
            return result

        log.error("gateway.all_failed", task=task, attempts=[vars(a) for a in attempts])
        raise AllProvidersFailed(task, attempts)

    def _skip_reason(self, ref: ModelRef) -> str | None:
        if ref.is_placeholder:
            return "placeholder"
        provider = self.providers.get(ref.provider)
        if provider is None:
            return "provider_not_implemented"
        if not provider.is_configured():
            return "provider_not_configured"
        return self.state.unavailable_reason(ref.provider, ref.model)

    async def _with_retry(
        self, call: Callable[[], Awaitable[Any]], ref: ModelRef
    ) -> LLMResponse | ImageResponse:
        """Call once, retrying once when `fallback.decide` says so (tenacity)."""

        def should_retry(exc: BaseException) -> bool:
            if not isinstance(exc, ProviderError):
                return False
            # attempt number is unknown here; the first failure is always attempt 1 for the
            # retry predicate, and stop_after_attempt caps the total at MAX_ATTEMPTS.
            return decide(exc, 1, self.state.now()).action is Action.RETRY_SAME

        def wait(state: RetryCallState) -> float:
            exc = state.outcome.exception() if state.outcome else None
            if isinstance(exc, ProviderError):
                return decide(exc, state.attempt_number, self.state.now()).delay
            return 0.0

        retrying = AsyncRetrying(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            retry=retry_if_exception(should_retry),
            wait=wait,
            sleep=self._sleep,
            reraise=True,
        )
        return await retrying(call)


__all__ = ["Gateway", "load_chains", "parse_ref", "task_kind"]
