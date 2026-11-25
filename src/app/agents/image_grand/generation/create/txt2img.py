"""txt2img sub-agent. Spec TXT2IMG-AGENT-SPEC.md section 4.

code_checks -> router -> enhancer (-> interrupt -> enhancer) -> code_guard -> generate xN
-> critic -> settle. Linear: it runs once, with no retry loop. The user picks from the images.
"""

from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from app.agents.image_grand.generation.create.checks import run_code_checks
from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.generation.create.emit import emit
from app.agents.image_grand.generation.create.settle import settle_run
from app.agents.image_grand.generation.create.state import CreateState
from app.agents.image_grand.workers import critic as critic_worker
from app.agents.image_grand.workers import prompt_enhancer as enhancer_worker
from app.agents.image_grand.workers.brief import Brief
from app.agents.image_grand.workers.image_gen import generate_one
from app.agents.image_grand.workers.safety import Safety, check_image_detail
from app.contracts.events import RunEvent, status_event
from app.core.errors import InvalidModelOutput, OutputBlocked, ProviderCallFailed, ProviderDown
from app.core.plans import Plan
from app.core.user_errors import to_user_error
from app.gateway.compiler import CompiledPrompt, build_prompt
from app.gateway.options import RequestExtras, options_for
from app.gateway.pricing import credits_for_run, credits_per_image
from app.gateway.registry import ModelCard
from app.gateway.router import RouteRequest, route_auto

log = structlog.get_logger("txt2img")

GENERATE_RESOLUTION = "1K"  # every image is 1K for now. 2K comes from the upscale tool later

Node = Callable[[CreateState], Awaitable[dict[str, Any]]]


def safe(node: Node) -> Any:
    """A failing step ends the run with a friendly error instead of crashing the stream."""

    async def wrapper(state: CreateState) -> dict[str, Any]:
        try:
            return await node(state)
        except Exception as exc:  # noqa: BLE001 - GraphInterrupt is not wrapped (see `ask`)
            return {"error": to_user_error(exc)}  # the one door to the client

    wrapper.__name__ = node.__name__
    return wrapper


def _card(deps: CreateDeps, state: CreateState) -> ModelCard:
    card = deps.registry.get(state["model_id"])
    if card is None:
        raise ProviderDown("The selected model is not available.")
    return card


def build_txt2img_graph(deps: CreateDeps) -> StateGraph[CreateState]:
    s = deps.settings

    # ---- [1] code checks ---------------------------------------------------------------------
    async def code_checks(state: CreateState) -> dict[str, Any]:
        emit(status_event(state["run_id"], "understanding"))
        trace = deps.tracer.get(state["run_id"])
        with trace.span("code_checks", inputs={"settings": state["settings"]}):
            return await run_code_checks(deps, state)

    # ---- [2] model router (code, before the enhancer) -----------------------------------------
    async def router(state: CreateState) -> dict[str, Any]:
        trace = deps.tracer.get(state["run_id"])
        picked = state["settings"]["model"]
        with trace.span("router", inputs={"model_setting": picked}) as span:
            if picked != "auto":  # the user's own pick is final
                card = deps.registry.get(picked)
                category = "manual"
            else:
                routed = route_auto(
                    deps.registry,
                    RouteRequest(
                        prompt=state["prompt"],
                        plan=Plan(state["plan"]),
                        aspect_ratio=state["settings"]["aspect_ratio"],
                        enabled_providers=deps.enabled_providers,
                        n_refs=state.get("n_refs", 0),
                        brand_colors=tuple(state["settings"].get("brand_colors") or ()),
                        draft=bool(state["settings"].get("draft")),
                        transparent=bool(state["settings"].get("transparent_bg")),
                    ),
                )
                card, category = routed.card, routed.rule  # the rule that chose, for the trace
            if card is None:
                raise ProviderDown("The selected model is not available.")
            provider = card.providers_in(deps.enabled_providers)[0].name
            trace.set_metadata(model=card.id, provider=provider)
            # Hold exactly what this model costs. The tier is known now, so no worst-case guess.
            amount = credits_for_run(
                card,
                state["settings"]["n"],
                resolution=GENERATE_RESOLUTION,
                n_refs=state.get("n_refs", 0),
            )
            hold_id = await deps.credits.hold(state["user_id"], state["run_id"], amount)
            try:  # `safe()` drops this node's return value on an error, so `fail` would never
                # learn the hold id. Anything that can raise after the hold must release it.
                span.log(
                    category=category,  # "manual", or the rule that chose (vector, brand ...)
                    model=card.id,
                    model_name=card.name,
                    tier=card.tier.value,
                    provider=provider,
                    credits_per_image=credits_per_image(card, n_refs=state.get("n_refs", 0)),
                    held=amount,
                    n_refs=state.get("n_refs", 0),
                )
                return {"category": category, "model_id": card.id, "hold_id": hold_id}
            except BaseException:
                await deps.credits.release(hold_id)
                raise

    # ---- [3] enhancer: translate, missing check, brief, final prompt --------------------------
    async def enhancer(state: CreateState) -> dict[str, Any]:
        emit(status_event(state["run_id"], "thinking"))
        trace = deps.tracer.get(state["run_id"])
        card = _card(deps, state)
        answers = state.get("answers") or {}
        asked = bool(answers)
        prompt = state["prompt"]
        base = deps.prompts.get("enhancer")
        trace.set_metadata(enhancer_prompt_version=base.version)
        with trace.span(
            "enhancer",
            inputs={"prompt": prompt, "answers": answers, "asked_before": asked},
            run_type="llm",
        ) as span:
            span.meta(prompt_version=base.version, model=deps.enhancer.model, provider="cloudflare")
            usage: dict[str, Any] = {}
            refs = state.get("n_refs", 0) > 0  # an edit needs the enhancer to write the delta
            if not asked and not refs and enhancer_worker.should_skip(prompt):
                out = enhancer_worker.skipped_output(prompt)
                span.log(skipped=True)
            else:
                out = await enhancer_worker.enhance(
                    deps.enhancer,
                    card,
                    base_prompt=base.text,
                    prompt=prompt,
                    aspect_ratio=state["settings"]["aspect_ratio"],
                    resolution=GENERATE_RESOLUTION,
                    answers=answers,
                    allow_questions=not asked,  # at most 1 round
                    has_references=refs,
                    max_tokens=s.enhancer_max_tokens,
                    timeout=s.llm_timeout_s,
                    usage=usage,
                )
            span.log(status=out.status, brief=out.brief.model_dump(), final_prompt=out.final_prompt)
            span.log(**usage)  # prompt_tokens, completion_tokens, neurons
        if out.status == "missing":
            emit(status_event(state["run_id"], "need_detail"))
        return {
            "brief": out.brief.model_dump(),
            "final_prompt": out.final_prompt,
            "questions": [q.model_dump() for q in out.questions],
        }

    async def ask(state: CreateState) -> dict[str, Any]:
        """Pause the whole graph. The checkpointer saves it and the SSE stream closes. Nothing
        else runs in this node, because it runs again from the top on resume."""
        answers = interrupt({"questions": state["questions"]})
        return {"answers": answers, "questions": []}

    # ---- [4] code guard -----------------------------------------------------------------------
    async def code_guard(state: CreateState) -> dict[str, Any]:
        emit(status_event(state["run_id"], "compiling"))
        trace = deps.tracer.get(state["run_id"])
        card = _card(deps, state)
        with trace.span("code_guard", inputs={"mode": s.compile_mode}) as span:
            compiled = build_prompt(
                card,
                Brief.model_validate(state["brief"]),
                state.get("final_prompt"),
                mode=s.compile_mode,
                aspect_ratio=state["settings"]["aspect_ratio"],
                resolution=GENERATE_RESOLUTION,
                user_prompt=state["prompt"],  # weak words the user typed are kept
            )
            llm_prompt = state.get("final_prompt") or ""
            span.log(
                llm_prompt=llm_prompt,  # what the enhancer wrote
                final_prompt=compiled.prompt,  # what the guard sends
                prompt_changed=compiled.prompt != llm_prompt.strip(),
                negative_prompt=compiled.negative_prompt,
                words=len(compiled.prompt.split()),
                max_words=card.prompt_max_words,
                size=f"{compiled.width}x{compiled.height}",
                model=card.id,
            )
        # The parent span of every variation. The critic step closes it.
        generate = trace.start(
            "generate",
            inputs={"n": state["settings"]["n"], "model": card.id},
            run_type="chain",
        )
        generate.meta(model=card.id, provider=card.providers_in(deps.enabled_providers)[0].name)
        emit(status_event(state["run_id"], "creating", progress=f"0/{state['settings']['n']}"))
        return {"compiled": asdict(compiled)}

    def fan_out(state: CreateState) -> list[Send] | str:
        if state.get("error"):
            return "fail"
        return [Send("generate_one", {**state, "index": i}) for i in range(state["settings"]["n"])]

    # ---- [5] generate: one parallel branch per variation --------------------------------------
    async def generate_one_node(state: dict[str, Any]) -> dict[str, Any]:
        run_id, index = state["run_id"], state["index"]
        trace = deps.tracer.get(run_id)
        card = _card(deps, state)  # type: ignore[arg-type]
        compiled = CompiledPrompt(**state["compiled"])
        span = trace.start(
            "generate_image",
            parent="generate",
            key=f"generate_image_{index}",
            inputs={"index": index, "final_prompt": compiled.prompt},
        )
        try:
            settings = state["settings"]
            options = options_for(
                card,
                RequestExtras(
                    brand_colors=tuple(settings.get("brand_colors") or ()),
                    transparent_bg=bool(settings.get("transparent_bg")),
                ),
            )
            saved = await generate_one(
                deps.gateway,
                deps.store,
                card,
                compiled,
                run_id=run_id,
                index=index,
                timeout=s.image_timeout_s,
                span=span,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001 - one bad variation never kills the others
            span.end(error=type(exc).__name__)
            log.warning(
                "generate_failed",
                run_id=run_id,
                index=index,
                model=card.id,
                error=type(exc).__name__,
                detail=str(exc)[:200],  # status text only. Provider errors never hold secrets
            )
            return {"results": [{"ok": False, "index": index, "code": getattr(exc, "code", "")}]}
        # Safety first, on the 256 px thumbnail. Unsafe or unchecked images are deleted and never
        # shown or charged (fail closed, per image).
        thumb = deps.store.read_bytes(run_id, saved.thumb)
        verdict = Safety.FAILED
        if thumb is not None:
            checked = await check_image_detail(
                deps.vision, thumb, compiled.prompt, timeout=s.llm_timeout_s
            )
            verdict = checked.verdict
            trace.record(  # timed inside the helper, so the latency is the real one
                "safety",
                parent=f"generate_image_{index}",
                key=f"safety_{index}",
                started_at=checked.started_at,
                ended_at=checked.ended_at,
                inputs={"thumb": saved.thumb, "prompt": compiled.prompt},
                outputs={
                    "verdict": verdict.value,
                    "model": checked.model,
                    "provider": "cloudflare",
                    "p_unsafe": checked.p_unsafe,
                    "neurons": checked.neurons,
                    "error": checked.error,
                },
            )
        span.log(safety=verdict.value)
        if verdict is not Safety.SAFE:
            deps.store.delete(run_id, saved.file)
            deps.store.delete(run_id, saved.thumb)
            log.warning("image_dropped_by_safety", run_id=run_id, index=index, safety=verdict.value)
            span.end()
            emit(RunEvent(type="generated", run_id=run_id))  # it counts as done for progress
            return {
                "results": [
                    {"ok": False, "index": index, "unsafe": verdict is Safety.UNSAFE, "code": ""}
                ]
            }
        span.end()
        # Not shown yet. The image is released to the user only after the run is saved.
        emit(RunEvent(type="generated", run_id=run_id))
        return {"results": [{"ok": True, **asdict(saved)}]}

    # ---- [6] critic --------------------------------------------------------------------------
    async def critic(state: CreateState) -> dict[str, Any]:
        run_id = state["run_id"]
        trace = deps.tracer.get(run_id)
        results = sorted(state.get("results", []), key=lambda r: r["index"])
        ok = [r for r in results if r["ok"]]
        gen = trace.span_named("generate")
        if gen is not None:
            gen.log(delivered=len(ok), failed=len(results) - len(ok))
            gen.end()
        if gen is not None:
            gen.log(unsafe_dropped=sum(1 for r in results if r.get("unsafe")))
        if not ok:
            if any(r.get("unsafe") for r in results):  # every image that was made was unsafe
                log.error("guardrail_miss", run_id=run_id, reason="all output images unsafe")
                raise OutputBlocked()
            raise ProviderDown("We couldn't create your image. Please try again.")
        emit(status_event(run_id, "final"))
        base = deps.prompts.get("critic")
        trace.set_metadata(critic_prompt_version=base.version)
        thumbs: list[tuple[str, bytes]] = []
        for r in ok:
            data = deps.store.read_bytes(run_id, r["thumb"])
            if data is None:
                raise InvalidModelOutput("a thumbnail is missing")
            thumbs.append((critic_worker.image_id(r["index"]), data))
        with trace.span(
            "critic",
            inputs={
                "images": [
                    {"id": i, "thumb": r["thumb"]} for (i, _), r in zip(thumbs, ok, strict=True)
                ]
            },
            run_type="llm",
        ) as span:
            span.meta(prompt_version=base.version, model=deps.critic.model, provider="cloudflare")
            usage: dict[str, Any] = {}
            review = await _review_with_retry(deps, base.text, state, thumbs, usage)
            span.log(  # scores stay in the backend. They are traced, never streamed
                verdicts={k: asdict(v) for k, v in review.verdicts.items()},
                best=review.best,
                critic_failed=review.failed,
                **usage,  # prompt_tokens, completion_tokens, neurons
            )
        return {
            "review": {
                "verdicts": {k: asdict(v) for k, v in review.verdicts.items()},
                "best": review.best,
                "highlight": review.highlight,
                "failed": review.failed,
            }
        }

    # ---- [7] settle --------------------------------------------------------------------------
    async def settle(state: CreateState) -> dict[str, Any]:
        trace = deps.tracer.get(state["run_id"])
        with trace.span("settle") as span:
            out = await settle_run(deps, state, span)
        return out

    async def fail(state: CreateState) -> dict[str, Any]:
        err = state.get("error") or {"code": "internal_error", "message": "Something went wrong."}
        if state.get("hold_id"):
            await deps.credits.release(state["hold_id"])  # nothing is charged on a failed run
        deps.store.purge(state["run_id"])  # and nothing from a failed run stays fetchable
        emit(
            RunEvent(type="error", run_id=state["run_id"], code=err["code"], message=err["message"])
        )
        return {}

    def _next(target: str) -> Callable[[CreateState], str]:
        return lambda st: "fail" if st.get("error") else target

    def after_enhancer(state: CreateState) -> str:
        if state.get("error"):
            return "fail"
        return "ask" if state.get("questions") else "code_guard"

    g = StateGraph(CreateState)
    g.add_node("code_checks", safe(code_checks))
    g.add_node("router", safe(router))
    g.add_node("enhancer", safe(enhancer))
    g.add_node("ask", ask)  # not wrapped: interrupt() must reach the checkpointer
    g.add_node("code_guard", safe(code_guard))
    g.add_node("generate_one", generate_one_node)  # type: ignore[type-var]
    g.add_node("critic", safe(critic))
    g.add_node("settle", safe(settle))
    g.add_node("fail", fail)
    g.add_edge(START, "code_checks")
    g.add_conditional_edges("code_checks", _next("router"), ["router", "fail"])
    g.add_conditional_edges("router", _next("enhancer"), ["enhancer", "fail"])
    g.add_conditional_edges("enhancer", after_enhancer, ["ask", "code_guard", "fail"])
    g.add_edge("ask", "enhancer")
    g.add_conditional_edges("code_guard", fan_out, ["generate_one", "fail"])
    g.add_edge("generate_one", "critic")
    g.add_conditional_edges("critic", _next("settle"), ["settle", "fail"])
    g.add_conditional_edges("settle", _next(END), [END, "fail"])
    g.add_edge("fail", END)
    return g


async def _review_with_retry(
    deps: CreateDeps,
    system: str,
    state: CreateState,
    thumbs: list[tuple[str, bytes]],
    usage: dict[str, Any] | None = None,
) -> critic_worker.Review:
    """One retry. If the critic still fails it falls OPEN: the images already passed the safety
    check, so they go out in generation order with no recommendation."""
    for attempt in range(2):
        try:
            return await critic_worker.review(
                deps.critic,
                system=system,
                brief=state["brief"],
                thumbs=thumbs,
                timeout=deps.settings.llm_timeout_s,
                user_prompt=state["prompt"],
                usage=usage,
            )
        except (InvalidModelOutput, ProviderCallFailed) as exc:
            log.warning("critic_failed", attempt=attempt, error=type(exc).__name__)
    return critic_worker.Review(verdicts={}, best=None, highlight="", failed=True)


def compile_txt2img(
    deps: CreateDeps,
) -> CompiledStateGraph[CreateState, None, CreateState, CreateState]:
    return build_txt2img_graph(deps).compile()
