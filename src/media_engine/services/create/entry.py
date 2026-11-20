"""`create_images()`: the one public entry point of the image side (spec §3). A thin wrapper:
it validates the arguments, builds the ports and the child agents, and hands the request to
the Image Grand Agent (agents/image_grand/grand.py), which runs the Safety Gate first on every
path and then delegates.

It is a library call: no server, no database, no queue. It runs to completion and returns one
Envelope. A first-pass success makes exactly three LLM calls in total (gate, brief, critic):
the coordinators add none.

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

Billing and failure signals (for the backend): see the Image Grand Agent's docstring. Every
envelope carries `output["billable"]`; a run that ends before an image step also carries
`output["failure"]`, one of EARLY_FAILURES (RETRYABLE_FAILURES are outages to retry).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import lru_cache

from media_engine.agents.decider import Decider
from media_engine.agents.delegate import Delegate
from media_engine.agents.image_generation.agent import OPERATIONS, ImageGenerationAgent
from media_engine.agents.image_generation.create.agent import create_image_create_agent
from media_engine.agents.image_grand.grand import (
    EARLY_FAILURES,
    RETRYABLE_FAILURES,
    EarlyFailure,
    ImageGrandAgent,
    invalid,
)
from media_engine.contracts import AuthContext, Envelope, Goal
from media_engine.gateway.budget import dearest_image_usd
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.router import Gateway
from media_engine.harness.trace import Trace
from media_engine.ports import (
    ArtifactStore,
    EventSink,
    InMemoryRunStore,
    ListEventSink,
    LocalArtifactStore,
    RunStore,
)
from media_engine.services.create.loader import CreateConfig, get_create_config

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
    trace: Trace | None = None,
    operation: str | None = None,
    decider: Decider | None = None,
) -> Envelope:
    """Turn `prompt` into `n` quality-checked images. See the create agent's docstring for the
    envelope's output. Never talks to the user: NEEDS_INPUT and ESCALATE are returned upward.

    `trace` (optional): a developer `Trace` the caller keeps, to read what the run did while it is
    still going (every model call, every decision). It is a separate channel: nothing in it
    reaches the Envelope or an event.

    `operation` (optional): an explicit operation (create|edit|merge|restyle) from a button or a
    slash command, routed by plain code. None: inferred (the Decider, else create). `decider`
    (optional): routes ambiguous text; without one, every ambiguous request is a create."""
    cfg = cfg or get_create_config()
    if not prompt.strip():
        return invalid("no prompt given")
    if not 1 <= n <= cfg.policy.max_images_per_request:
        return invalid(f"n must be between 1 and {cfg.policy.max_images_per_request}")
    if operation is not None and operation not in OPERATIONS:
        return invalid(f"operation must be one of {', '.join(OPERATIONS)}")

    backend = gateway or get_gateway()
    sink = events if events is not None else ListEventSink()
    run_id = run_id or f"run_{uuid.uuid4().hex}"
    llm = LLMGateway(backend, chain_resolver(cfg.agents))
    run_trace = trace or Trace()

    def create() -> Delegate:  # built only when a request is routed to create
        return create_image_create_agent(
            cfg,
            llm=llm,
            images=backend,
            artifacts=artifacts or LocalArtifactStore(),
            events=sink,
            run_store=run_store or InMemoryRunStore(),
            price_lookup=dearest_image_usd,
            seed_source=seed_source,
            trace=run_trace,
        )

    def generation() -> Delegate:
        return ImageGenerationAgent(
            create=create,
            trace=run_trace,
            decider=decider,
            ambiguous_to_stubs=cfg.policy.routing.ambiguous_to_stubs,
        )

    grand = ImageGrandAgent(
        cfg, llm=llm, events=sink, trace=run_trace, generation=generation, decider=decider
    )
    return await grand.run(
        Goal(task="create", instructions="Create the image the user asked for."),
        cfg.policy.stopping_condition("supervise", auth.plan, "finished, checked images"),
        auth,
        {
            "prompt": prompt,
            "settings": settings or {},
            "n": n,
            "run_id": run_id,
            "user_id": user_id,
            "operation": operation,
        },
    )


__all__ = [
    "DEFAULT_AUTH",
    "EARLY_FAILURES",
    "RETRYABLE_FAILURES",
    "EarlyFailure",
    "aclose_gateway",
    "create_images",
    "get_gateway",
]
