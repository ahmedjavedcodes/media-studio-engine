"""`create_images()`: the one public entry point of the image-creation harness (spec §3).

    route (1 LLM call) -> simple: Image Grand Agent  |  complex: Grand Agent stub (ESCALATE)

It is a library call: no server, no database, no queue. It runs to completion and returns one
Envelope. A first-pass success makes exactly three LLM calls in total (router, enhancer, critic):
the supervisor adds none.

The provider Gateway is built ONCE per process (`get_gateway()`): the circuit breakers, the
availability state and the daily meters live on it, so a second instance would start blind and
re-pay every timeout. Callers (the backend, tests) may pass their own. Everything else is cheap
and built per call.

Ports (all optional, local implementations by default): `run_store` (default: in-memory),
`artifacts` (default: `./.artifacts/`), `events`. TODO(backend): the backend passes its own.

WARNING: without a `run_store`, every call creates a FRESH `InMemoryRunStore`, and without
`events` a fresh `ListEventSink`, that nothing else can reach. A caller that wants to cancel a
run (the backend sets `cancel_requested` in the run store) or to read its events MUST pass its
own `run_store` and `events`, and a `run_id` to address the run.

The router's spend is added to the returned envelope's spend. It is not charged to the run's
budget (the run does not exist yet when the router runs; its estimate is a few neurons).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from media_engine.agents.base import AgentConfig
from media_engine.agents.grand.grand_agent import GrandAgent
from media_engine.agents.image_grand.supervisor import create_image_grand_agent
from media_engine.agents.router.router import Router
from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Goal,
    Spend,
    Status,
)
from media_engine.core.logging import get_logger
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.router import Gateway
from media_engine.ports import (
    ArtifactStore,
    EventSink,
    InMemoryRunStore,
    ListEventSink,
    LocalArtifactStore,
    RunStore,
)
from media_engine.schemas import RouteDecision
from media_engine.services.create.loader import CreateConfig, get_create_config

log = get_logger("create")

DEFAULT_AUTH = AuthContext(plan="free", execution_mode="auto", model_demand="balanced")


@lru_cache
def get_gateway() -> Gateway:
    """The process-wide provider gateway. One instance: see the module docstring."""
    return Gateway()


async def aclose_gateway() -> None:
    """Close the shared gateway's HTTP client (scripts call this on exit)."""
    if get_gateway.cache_info().currsize:
        await get_gateway().aclose()
        get_gateway.cache_clear()


def _invalid(reason: str) -> Envelope:
    return Envelope(
        output={"failure": "invalid_input"},
        summary="The request could not be started.",
        state=EnvelopeState(status=Status.FAILED, spend=Spend(), reason=reason),
    )


async def create_images(
    prompt: str,
    *,
    auth: AuthContext = DEFAULT_AUTH,
    n: int = 1,
    settings: dict[str, str] | None = None,
    user_id: str = "anonymous",
    run_id: str | None = None,
    gateway: Gateway | None = None,
    run_store: RunStore | None = None,
    artifacts: ArtifactStore | None = None,
    events: EventSink | None = None,
    cfg: CreateConfig | None = None,
    seed_source: Callable[[], int] | None = None,
) -> Envelope:
    """Turn `prompt` into `n` quality-checked images. See the supervisor's docstring for the
    envelope's output. Never talks to the user: NEEDS_INPUT and ESCALATE are returned upward."""
    cfg = cfg or get_create_config()
    if not prompt.strip():
        return _invalid("no prompt given")
    if not 1 <= n <= cfg.policy.max_images_per_request:
        return _invalid(f"n must be between 1 and {cfg.policy.max_images_per_request}")

    backend = gateway or get_gateway()
    sink = events if events is not None else ListEventSink()
    run_id = run_id or f"run_{uuid.uuid4().hex}"
    llm = LLMGateway(backend, chain_resolver(cfg.agents))

    # 1. route: one small call. A failure defaults to "simple" (see the router).
    router = Router(
        AgentConfig.from_policy("router", cfg.agent("router"), RouteDecision), llm, sink
    )
    routed = await router.run(
        Goal(task="route", instructions="Decide how much work the request needs."),
        cfg.policy.stopping_condition("route", auth.plan, "a route: simple or complex"),
        auth,
        {"prompt": prompt},
    )
    route = routed.output.get("route", "simple")
    log.info("create.routed", run_id=run_id, route=route)

    # 2. hand over
    if route == "complex":
        grand = GrandAgent(
            AgentConfig.from_policy("grand_agent", cfg.agent("grand_agent"), None), None, sink
        )
        env = await grand.run(
            Goal(task="plan", instructions="Plan the request."),
            cfg.policy.stopping_condition("supervise", auth.plan, "a plan"),
            auth,
            {"prompt": prompt},
        )
        env.output = {"run_id": run_id, **env.output}
    else:
        supervisor = create_image_grand_agent(
            cfg,
            llm=llm,
            images=backend,
            artifacts=artifacts or LocalArtifactStore(),
            events=sink,
            run_store=run_store or InMemoryRunStore(),
            seed_source=seed_source,
        )
        inputs: dict[str, Any] = {
            "prompt": prompt,
            "settings": settings or {},
            "n": n,
            "run_id": run_id,
            "user_id": user_id,
        }
        env = await supervisor.run(
            Goal(task="create", instructions="Create the image the user asked for."),
            cfg.policy.stopping_condition("supervise", auth.plan, "finished, checked images"),
            auth,
            inputs,
        )
    env.state.spend = env.state.spend.plus(routed.state.spend)
    return env


__all__ = ["DEFAULT_AUTH", "aclose_gateway", "create_images", "get_gateway"]
