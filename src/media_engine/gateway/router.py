"""The gateway: the only door for model calls. Walks a fallback chain and logs every step.

Per chain entry, three independent checks decide whether to call it: `fallback.py` rules
(quota exhausted, model disabled, cooling down), the per-provider circuit breaker
(`breaker.py`), and the caller's deadline. A content refusal or a 400 stops the chain."""

from __future__ import annotations

import asyncio
import time
from asyncio import FIRST_COMPLETED
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

from media_engine.contracts import Plan
from media_engine.core.config import Settings, get_settings
from media_engine.core.logging import get_logger
from media_engine.core.telemetry import traced
from media_engine.gateway.breaker import BreakerBoard, CircuitBreaker, Permit, Verdict, verdict_for
from media_engine.gateway.budget import (
    DailyCostMeter,
    DailyNeuronMeter,
    estimate_image_neurons,
    estimate_image_usd,
)
from media_engine.gateway.compiler import (
    DEFAULT_LEAD,
    DEFAULT_MAX_WORDS,
    DEFAULT_STYLE,
    Lead,
    compile_prompt,
    style_for_cloudflare,
)
from media_engine.gateway.errors import (
    AllProvidersFailed,
    BadRequestError,
    ContentRefused,
    DeadlineExceeded,
    ProviderError,
    RequestNotAllowed,
)
from media_engine.gateway.fallback import (
    CF_UNCLASSIFIED_400,
    MAX_ATTEMPTS,
    Action,
    AvailabilityState,
    decide,
)
from media_engine.gateway.image_registry import image_registry
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
    Route,
)
from media_engine.guardrails.code_checks import check_prompt

CHAINS_PATH = Path(__file__).with_name("chains.yaml")

log = get_logger("gateway")

Invoke = Callable[[Provider, str], Awaitable[LLMResponse | ImageResponse]]
RouteCheck = Callable[[Provider, ModelRef], Route]

# Plan assumed when an image caller does not say: the most restrictive one (fail closed). A bug
# that drops the plan can then only make a request cheaper, never dearer.
DEFAULT_PLAN: Plan = "free"
# Largest side sent to a Cloudflare image model whose card names no max_side. The FLUX.2 Klein
# limits are not verified, so higher resolutions go only to models known to support them.
CLOUDFLARE_DEFAULT_MAX_SIDE = 1024


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


class _DeadlineHit(Exception):
    """Internal: the caller's deadline ran out inside one chain entry. Not a ProviderError, so
    the retry predicate never retries it; `_run` turns it into `DeadlineExceeded`."""


@dataclass(frozen=True)
class _Deadline:
    """The caller's budget for one chain walk. Uses time.monotonic, the clock asyncio.timeout
    effectively counts in, never the injectable breaker clock."""

    budget_s: float
    at: float

    @classmethod
    def start(cls, budget_s: float | None) -> _Deadline | None:
        return None if budget_s is None else cls(budget_s, time.monotonic() + budget_s)

    def remaining(self) -> float:
        return self.at - time.monotonic()


class Gateway:
    """The fallback-chain gateway. Create ONE instance per process and share it: what it
    learns about providers (AvailabilityState, circuit breakers, daily meters) lives on the
    instance, so a second instance would start blind and re-pay every timeout."""

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
        breakers: BreakerBoard | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self.providers = providers or build_providers(self.settings, self._client)
        self.chains = chains if chains is not None else load_chains()
        self.state = state or AvailabilityState()
        self.meter = meter or DailyNeuronMeter()
        self.cost_meter = cost_meter or DailyCostMeter()
        self.breakers = breakers or BreakerBoard()
        self._sleep = sleep

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- public API -------------------------------------------------------------------------

    @traced("gateway.chat", process_inputs=_safe_inputs, process_outputs=_safe_outputs)
    async def chat(
        self, task: str, request: LLMRequest, *, deadline_s: float | None = None
    ) -> CallResult:
        return await self._run(task, "llm", lambda p, m: p.chat(m, request), deadline_s=deadline_s)

    @traced("gateway.image", process_inputs=_safe_inputs, process_outputs=_safe_outputs)
    async def image(
        self,
        task: str,
        request: ImageRequest,
        *,
        deadline_s: float | None = None,
        start_at: int = 0,
        plan: Plan | None = None,
        max_cost_usd: float | None = None,
        first: ModelRef | None = None,
    ) -> CallResult:
        """`start_at` skips that many leading chain entries: the `switch_model` retry lever
        starts the walk at the next model. 0 is the normal walk.

        `first`: the model the USER picked. It is tried before the chain (and counts as the
        primary entry, not a fallback); if it is refused or fails, the walk continues down the
        normal chain. It goes through exactly the same plan, price and settings checks as a
        chain entry, so a pick can never reach a model the plan may not use.

        Each entry gets the request SIZED for it (`_sized_for`): a model with a known max side
        is never sent more, and the cost/neuron estimate uses the size actually sent.

        `plan` is enforced HERE, not by the agents: before any entry is called, its provider's
        `route_image` checks the model's tier, every price-multiplying setting that would be
        sent (e.g. gpt-image-2 quality high is ultra only), and `n`. A refused entry is
        skipped and never called. No plan given = "free" (fail closed).

        `max_cost_usd`: what this ONE call may cost. A paid entry whose estimate (budget.
        estimate_image_usd) is above it, or has no price, is skipped before the call: an unknown
        price is "cannot afford". None = no cap given (the plan guard still applies)."""
        effective_plan: Plan = plan or DEFAULT_PLAN
        return await self._run(
            task,
            "image",
            lambda p, m: p.generate_image(
                m, self._compiled_for(self._sized_for(request, p.name, m), p.name, m)
            ),
            estimate=lambda ref: estimate_image_neurons(
                card_or_default(ref.provider, ref.model, "image"),
                self._sized_for(request, ref.provider, ref.model),
            ),
            deadline_s=deadline_s,
            start_at=start_at,
            route=lambda p, ref: self._affordable(
                p.route_image(
                    ref.model, self._sized_for(request, ref.provider, ref.model), effective_plan
                ),
                ref,
                self._sized_for(request, ref.provider, ref.model),
                max_cost_usd,
            ),
            first=first,
        )

    @staticmethod
    def _sized_for(request: ImageRequest, provider: str, model: str) -> ImageRequest:
        """The request scaled down, ratio kept, so its long side fits what THIS model takes
        (its card's max_side; a Cloudflare model with none is held to 1024). Sides are kept
        multiples of 16. Unchanged when it already fits."""
        card = card_or_default(provider, model, "image")
        limit = card.max_side or (CLOUDFLARE_DEFAULT_MAX_SIDE if provider == "cloudflare" else None)
        side = max(request.width, request.height)
        if limit is None or side <= limit:
            return request
        scale = limit / side

        def fit(value: int) -> int:
            return max(16, int(value * scale) // 16 * 16)

        return request.model_copy(
            update={"width": fit(request.width), "height": fit(request.height)}
        )

    @staticmethod
    def _affordable(
        decided: Route, ref: ModelRef, request: ImageRequest, max_cost_usd: float | None
    ) -> Route:
        """The USD half of the pre-call guard (the plan half is `route_image`). Only OpenRouter
        bills dollars. The estimate is checked against the cost cap BEFORE the call."""
        if decided.model is None or max_cost_usd is None or ref.provider != "openrouter":
            return decided
        spec = image_registry().get(decided.model)
        estimate = estimate_image_usd(spec, request) if spec is not None else None
        if estimate is None:
            return Route(None, f"{decided.model}: price unknown, cannot afford")
        if estimate > max_cost_usd:
            return Route(
                None,
                f"{decided.model}: estimated ${estimate:.4f} is over the ${max_cost_usd:.4f} left",
            )
        return decided

    @staticmethod
    def _compiled_for(request: ImageRequest, provider: str, model: str) -> ImageRequest:
        """The request as THIS chain entry should get it: with a Brief, its prompt compiled for
        the model's style and word budget, then re-checked by the step 0 code checks. A compiled
        prompt that fails them is a bad request (never falls back): the Brief is what is wrong."""
        if request.brief is None:
            return request
        lead: Lead = DEFAULT_LEAD
        if provider == "cloudflare":
            style, words = style_for_cloudflare(model)
        else:
            spec = image_registry().get(model)
            style, words = (
                (spec.prompt_style, spec.prompt_max_words)
                if spec is not None
                else (DEFAULT_STYLE, DEFAULT_MAX_WORDS)
            )
            if spec is not None:
                lead = spec.lead_order
        compiled = compile_prompt(request.brief, style, words, fallback=request.prompt, lead=lead)
        verdict = check_prompt(compiled.text)
        if not verdict.ok:
            raise ProviderError(
                provider, model, f"compiled prompt refused ({verdict.rule})", status=400
            )
        log.info(
            "gateway.compiled",
            provider=provider,
            model=model,
            style=style,
            words=compiled.words,
            max_words=words,
            issues=compiled.issues,
        )
        return request.model_copy(update={"prompt": compiled.text})

    # -- internals --------------------------------------------------------------------------

    async def _run(
        self,
        task: str,
        kind: Kind,
        invoke: Invoke,
        estimate: Callable[[ModelRef], float | None] | None = None,
        deadline_s: float | None = None,
        start_at: int = 0,
        route: RouteCheck | None = None,
        first: ModelRef | None = None,
    ) -> CallResult:
        """Walk the chain. `deadline_s` bounds the whole walk, retries, fallbacks and probe
        waits included: each call gets only the time that is left, and is cancelled (httpx
        aborts the request) when it runs out.

        Probe window: if an entry is skipped only because another request is probing its
        provider (half-open), and nothing else in the chain works, wait for that probe's
        verdict and then retry just the entries that were skipped for it. On the free path
        Cloudflare is the only image provider, so without this every concurrent request (M8
        branches) would fail for the length of a probe."""
        if task not in self.chains:
            raise ValueError(f"Unknown task {task!r}; known: {sorted(self.chains)}")
        if task_kind(task) != kind:
            raise ValueError(f"Task {task!r} is a {task_kind(task)} task, not {kind}")

        attempts: list[Attempt] = []
        deadline = _Deadline.start(deadline_s)
        if start_at < 0:
            raise ValueError(f"start_at must be >= 0, got {start_at}")
        entries = list(enumerate(self.chains[task]))[start_at:]
        if first is not None:
            if first.provider not in self.providers:
                raise ValueError(f"Unknown provider for the picked model: {first.provider!r}")
            # The pick is entry 0 (the primary); the chain follows as fallbacks, without it.
            entries = [(0, first)] + [(i + 1, ref) for i, ref in entries if ref != first]
        limits: list[str] = []  # pre-call guard refusals (plan, paid setting, n)

        for walk in (1, 2):
            waiting: list[tuple[int, ModelRef]] = []
            probes: dict[str, CircuitBreaker] = {}
            for index, ref in entries:
                skip = self._skip_reason(ref)
                if skip is None and route is not None:
                    decided = route(self.providers[ref.provider], ref)
                    if decided.model is None:
                        skip = f"not_allowed: {decided.reason}"
                        if decided.reason and decided.reason not in limits:
                            limits.append(decided.reason)
                    elif decided.model != ref.model:
                        attempts.append(
                            Attempt(ref.provider, ref.model, "skipped", reason=decided.reason)
                        )
                        log.info("gateway.routed", task=task, reason=decided.reason)
                        ref = ModelRef(ref.provider, decided.model)
                        skip = self.state.unavailable_reason(ref.provider, ref.model)
                permit: Permit | None = None
                if skip is None:
                    # Breaker last: a PROBE permit claims the half-open slot, and every path in
                    # _attempt hands it back.
                    breaker = self.breakers.get(ref.provider)
                    permit = breaker.allow()
                    if not permit.allowed:
                        skip = breaker.skip_reason
                        if breaker.probe_in_flight:
                            waiting.append((index, ref))
                            probes[ref.provider] = breaker
                if skip is not None or permit is None:
                    attempts.append(Attempt(ref.provider, ref.model, "skipped", reason=skip))
                    log.info(
                        "gateway.skip",
                        task=task,
                        provider=ref.provider,
                        model=ref.model,
                        reason=skip,
                        walk=walk,
                    )
                    continue
                result = await self._attempt(
                    task, index, ref, invoke, estimate, permit, attempts, deadline
                )
                if result is not None:
                    return result
            if walk == 2 or not waiting:
                break
            await self._await_probes(task, list(probes.values()), deadline, attempts)
            entries = waiting

        log.error("gateway.all_failed", task=task, attempts=[vars(a) for a in attempts])
        if limits and all(a.outcome == "skipped" for a in attempts):
            raise RequestNotAllowed(task, attempts, limits)
        raise AllProvidersFailed(task, attempts)

    async def _await_probes(
        self,
        task: str,
        probes: list[CircuitBreaker],
        deadline: _Deadline | None,
        attempts: list[Attempt],
    ) -> None:
        """Wait until the first in-flight probe reports back, within the deadline."""
        timeout = None if deadline is None else deadline.remaining()
        if timeout is not None and timeout <= 0:
            raise DeadlineExceeded(task, deadline.budget_s if deadline else 0.0, attempts)
        log.info("gateway.probe_wait", task=task, providers=[b.provider for b in probes])
        waits = [asyncio.ensure_future(b.wait_for_probe()) for b in probes]
        try:
            done, _ = await asyncio.wait(waits, timeout=timeout, return_when=FIRST_COMPLETED)
        finally:
            for w in waits:
                w.cancel()
        if not done:
            if deadline is None:  # pragma: no cover - asyncio.wait without timeout always waits
                raise RuntimeError("probe wait ended without a result or a deadline")
            log.warning("gateway.deadline_exceeded", task=task, during="probe_wait")
            raise DeadlineExceeded(task, deadline.budget_s, attempts)

    async def _attempt(
        self,
        task: str,
        index: int,
        ref: ModelRef,
        invoke: Invoke,
        estimate: Callable[[ModelRef], float | None] | None,
        permit: Permit,
        attempts: list[Attempt],
        deadline: _Deadline | None,
    ) -> CallResult | None:
        """Try one chain entry (with its retry). Returns the result, or None to move on.
        Raises ContentRefused / BadRequestError / DeadlineExceeded to stop the chain. Every
        path reports the breaker permit back exactly once."""
        provider = self.providers[ref.provider]
        breaker = self.breakers.get(ref.provider)
        started = time.perf_counter()
        calls = 0
        cut_off = False  # a call was cancelled mid-flight by the deadline
        last_error: ProviderError | None = None

        async def call() -> LLMResponse | ImageResponse:
            nonlocal calls, cut_off, last_error
            if deadline is not None and deadline.remaining() <= 0:
                raise _DeadlineHit
            calls += 1
            try:
                if deadline is None:
                    return await invoke(provider, ref.model)
                scope = asyncio.timeout(deadline.remaining())
                try:
                    async with scope:
                        return await invoke(provider, ref.model)
                except TimeoutError:
                    if not scope.expired():
                        raise
                    cut_off = True
                    raise _DeadlineHit from None
            except ProviderError as err:
                last_error = err
                raise

        try:
            output = await self._with_retry(call, ref)
        except _DeadlineHit:
            if deadline is None:  # _DeadlineHit is only raised when a deadline is set
                raise RuntimeError("deadline hit without a deadline") from None
            provider_down = cut_off or (
                last_error is not None and verdict_for(last_error) is Verdict.FAILURE
            )
            if provider_down:
                breaker.report(Verdict.FAILURE, permit)
            else:
                breaker.release(permit)  # our deadline, not the provider's fault
            attempts.append(
                Attempt(ref.provider, ref.model, "failed", reason="deadline", calls=calls)
            )
            log.warning(
                "gateway.deadline_exceeded",
                task=task,
                provider=ref.provider,
                model=ref.model,
                deadline_s=deadline.budget_s,
                calls=calls,
                cut_off=cut_off,
            )
            raise DeadlineExceeded(task, deadline.budget_s, attempts) from None
        except ProviderError as err:
            decision = decide(err, calls, self.state.now())
            self.state.apply(ref.provider, ref.model, decision)
            breaker.report(verdict_for(err), permit)
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
            if decision.action is Action.REFUSE:
                # A refusal that arrived as HTTP 200 was billed: meter it like a success.
                refused = ContentRefused(err, task, neurons=self._meter_refusal(ref, err))
                log.warning("gateway.content_refused", neurons=refused.neurons, **fields)
                raise refused from err
            if decision.action is Action.RAISE:
                event = (
                    "gateway.cloudflare_bad_request_unclassified"
                    if decision.reason == CF_UNCLASSIFIED_400
                    else "gateway.bad_request"
                )
                log.error(event, **fields)
                raise BadRequestError(err) from err
            getattr(log, decision.severity)(
                "gateway.fallback",
                next_action="next_provider",
                exhausted_until=decision.exhausted_until.isoformat()
                if decision.exhausted_until
                else None,
                model_disabled=decision.disable_model,
                breaker=breaker.state.value,
                **fields,
            )
            return None
        except BaseException:
            # Cancelled from outside (run cancel, step deadline) or an unexpected bug: we
            # learned nothing about the provider. Free our probe slot if we held it, re-raise.
            breaker.release(permit)
            raise

        breaker.report(Verdict.SUCCESS, permit)
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

    def _meter_refusal(self, ref: ModelRef, err: ProviderError) -> float | None:
        """Add a billed refusal's neurons / cost to the daily meters; return the neurons."""
        neurons: float | None = None
        if ref.provider == "cloudflare" and err.usage:
            raw = err.usage.get("neurons")
            if isinstance(raw, int | float):
                neurons = float(raw)
                self.meter.add(actual=neurons, estimate=None)
        if err.cost_usd is not None:
            self.cost_meter.add(err.cost_usd)
        return neurons

    def _skip_reason(self, ref: ModelRef) -> str | None:
        """Every reason to skip an entry except the breaker, which _run checks last."""
        if ref.is_placeholder:
            return "placeholder"
        provider = self.providers.get(ref.provider)
        if provider is None:
            return "provider_not_implemented"
        if not provider.is_configured():
            return "provider_not_configured"
        if ref.provider in self.settings.paid_providers and not self.settings.allow_paid:
            return "paid_disabled"
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
