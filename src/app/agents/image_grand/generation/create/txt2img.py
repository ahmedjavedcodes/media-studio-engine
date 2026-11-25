"""txt2img sub-agent. Spec TXT2IMG-AGENT-SPEC.md section 4.

code_checks -> router -> art_director (-> interrupt -> art_director) -> prompt_writer
-> code_guard -> generate xN
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
from app.agents.image_grand.workers import art_director as director
from app.agents.image_grand.workers import critic as critic_worker
from app.agents.image_grand.workers import plan_rules, prompt_writer
from app.agents.image_grand.workers.image_gen import generate_one
from app.agents.image_grand.workers.safety import Safety, check_image_detail
from app.agents.image_grand.workers.shot_plan import ShotPlan
from app.agents.image_grand.workers.subject_type import guess_subject_type
from app.contracts.events import RunEvent, status_event
from app.core.errors import InvalidModelOutput, OutputBlocked, ProviderCallFailed, ProviderDown
from app.core.plans import Plan
from app.core.user_errors import to_user_error
from app.gateway.compiler import CompiledPrompt, build_prompt
from app.gateway.options import RequestExtras, options_for
from app.gateway.pricing import credits_for_run, credits_per_image
from app.gateway.registry import ModelCard
from app.gateway.router import RouteRequest, classify_prompt, route_auto

log = structlog.get_logger("txt2img")

GENERATE_RESOLUTION = "1K"  # every image is 1K for now. 2K comes from the upscale tool later
DIRECTOR_MIN_TOKENS = 2500  # the plan and the medium reasoning need room, whatever the setting says
WRITER_MIN_TOKENS = 1200

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
            content_category = classify_prompt(state["prompt"])  # product, photo, people ...
            fixed_look = deps.looks.pick(state["settings"].get("preset", "auto"))
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
                    content_category=content_category,
                    subject_guess=guess_subject_type(state["prompt"]),
                    preset_requested=state["settings"].get("preset", "auto"),
                    look_fixed=fixed_look.id if fixed_look else None,
                )
                return {
                    "category": category,
                    "content_category": content_category,
                    "preset": fixed_look.id if fixed_look else "",
                    "model_id": card.id,
                    "hold_id": hold_id,
                }
            except BaseException:
                await deps.credits.release(hold_id)
                raise

    # ---- [3] art director: ANY prompt -> a ShotPlan (translate, missing check, the 6 steps) -----
    async def art_director(state: CreateState) -> dict[str, Any]:
        emit(status_event(state["run_id"], "thinking"))
        trace = deps.tracer.get(state["run_id"])
        answers = state.get("answers") or {}
        asked = bool(answers)
        prompt = state["prompt"]
        base = deps.prompts.get("art_director")
        trace.set_metadata(art_director_prompt_version=base.version)
        refs = state.get("n_refs", 0) > 0
        n_images = state["settings"]["n"]
        subject_guess = guess_subject_type(prompt)
        shots = deps.plans.similar(prompt, subject_type=subject_guess, k=2)
        project_id = state.get("project_id") or ""
        project_style = deps.plans.project_style(project_id) if project_id else None
        fixed_look = deps.looks.pick(state["settings"].get("preset", "auto"))
        with trace.span(
            "art_director",
            inputs={"prompt": prompt, "answers": answers, "asked_before": asked, "n": n_images},
            run_type="llm",
        ) as span:
            span.meta(
                prompt_version=base.version,
                model=deps.enhancer.model,
                provider=getattr(deps.enhancer, "provider", "cloudflare"),
            )
            usage: dict[str, Any] = {}
            report: dict[str, Any] = {}
            plan = await director.direct(
                deps.enhancer,
                base_prompt=base.text,
                prompt=prompt,
                aspect_ratio=state["settings"]["aspect_ratio"],
                resolution=GENERATE_RESOLUTION,
                answers=answers,
                allow_questions=not asked,  # at most 1 round
                n=n_images,
                looks=deps.looks,
                max_tokens=max(s.enhancer_max_tokens, DIRECTOR_MIN_TOKENS),
                timeout=s.llm_timeout_s,
                fixed_look=fixed_look,
                shots=shots,
                project_style=project_style,
                has_references=refs,
                usage=usage,
                report=report,
            )
            span.log(
                status="missing" if plan.missing else "ready",
                shot_plan=plan.compact(),  # traced (LangSmith, the dev panel). Never sent over SSE
                look=plan.look,
                subject_type=plan.subject_type,
                subject_guess=subject_guess,
                retrieved=[
                    {"id": x.id, "aesthetic": x.aesthetic, "prompt": x.prompt} for x in shots
                ],
                project_style=project_style,
                **report,  # attempts, retry_reason and the rule violations that were left
            )
            span.log(**usage)  # prompt_tokens, completion_tokens, neurons
        if plan.missing:
            emit(status_event(state["run_id"], "need_detail"))
        return {
            "shot_plan": plan.model_dump(mode="json"),
            "subject_type": plan.subject_type or subject_guess,
            "questions": [q.model_dump() for q in plan.missing],
        }

    # ---- [3b] prompt writer: the plan -> one final prompt per image ---------------------------
    async def prompt_writer_node(state: CreateState) -> dict[str, Any]:
        trace = deps.tracer.get(state["run_id"])
        card = _card(deps, state)
        plan = ShotPlan.model_validate(state["shot_plan"])
        n_images = state["settings"]["n"]
        with trace.span(
            "prompt_writer", inputs={"model": card.id, "n": n_images}, run_type="llm"
        ) as span:
            report: dict[str, Any] = {}
            usage: dict[str, Any] = {}
            if s.compile_mode == "code" or deps.writer is None:
                prompts = prompt_writer.render_all(card, plan, n_images)
                report = {"source": "template", "attempts": 0}
            else:
                base = deps.prompts.get("prompt_writer")
                trace.set_metadata(prompt_writer_prompt_version=base.version)
                span.meta(
                    prompt_version=base.version,
                    model=deps.writer.model,
                    provider=getattr(deps.writer, "provider", "cloudflare"),
                )
                prompts = await prompt_writer.write(
                    deps.writer,
                    card,
                    base_prompt=base.text,
                    plan=plan,
                    user_prompt=state["prompt"],
                    aspect_ratio=state["settings"]["aspect_ratio"],
                    n=n_images,
                    max_tokens=max(s.enhancer_max_tokens // 2, WRITER_MIN_TOKENS),
                    timeout=s.llm_timeout_s,
                    has_references=state.get("n_refs", 0) > 0,
                    usage=usage,
                    report=report,
                )
            span.log(
                prompts=prompts,
                target_words=list(plan_rules.word_range(card.prompt_max_words)),
                words=[len(p.split()) for p in prompts],
                **report,
            )
            span.log(**usage)
        return {"prompts": prompts}

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
        n_images = state["settings"]["n"]
        written = state.get("prompts") or []
        with trace.span("code_guard", inputs={"mode": s.compile_mode, "images": n_images}) as span:
            plan = ShotPlan.model_validate(state["shot_plan"])
            axes = plan.variation_axes or ["camera angle", "time of day", "prop layout"]
            compiled_list = [
                build_prompt(
                    card,
                    plan,
                    written[i] if i < len(written) else None,
                    mode=s.compile_mode,
                    aspect_ratio=state["settings"]["aspect_ratio"],
                    resolution=GENERATE_RESOLUTION,
                    user_prompt=state["prompt"],  # weak words the user typed are kept
                    change=prompt_writer.axis_change(axes[i % len(axes)], i)
                    if n_images > 1
                    else "",
                )  # each variation is guarded on its own: words, weak words, blocklist, size
                for i in range(n_images)
            ]
            compiled = compiled_list[0]
            llm_prompt = written[0] if written else ""
            span.log(
                llm_prompt=llm_prompt,  # what the prompt writer wrote
                final_prompt=compiled.prompt,  # what the guard sends
                prompt_changed=compiled.prompt != llm_prompt.strip(),
                negative_prompt=compiled.negative_prompt,
                words=len(compiled.prompt.split()),
                max_words=card.prompt_max_words,
                size=f"{compiled.width}x{compiled.height}",
                native_size=bool(card.native_sizes),  # True = snapped to the model's own size
                model=card.id,
                variations=[
                    {"index": i, "prompt": c.prompt, "words": len(c.prompt.split())}
                    for i, c in enumerate(compiled_list)
                ]
                if n_images > 1
                else None,
            )
        # The parent span of every variation. The critic step closes it.
        generate = trace.start(
            "generate",
            inputs={"n": n_images, "model": card.id},
            run_type="chain",
        )
        generate.meta(model=card.id, provider=card.providers_in(deps.enabled_providers)[0].name)
        emit(status_event(state["run_id"], "creating", progress=f"0/{state['settings']['n']}"))
        return {"compiled": asdict(compiled), "variants": [asdict(c) for c in compiled_list]}

    def fan_out(state: CreateState) -> list[Send] | str:
        if state.get("error"):
            return "fail"
        total = state["settings"]["n"]
        return [Send("generate_one", {**state, "index": i}) for i in range(total)]

    # ---- [5] generate: one parallel branch per variation --------------------------------------
    async def generate_one_node(state: dict[str, Any]) -> dict[str, Any]:
        run_id, index = state["run_id"], state["index"]
        trace = deps.tracer.get(run_id)
        card = _card(deps, state)  # type: ignore[arg-type]
        variants = state.get("variants") or [state["compiled"]]
        compiled = CompiledPrompt(**variants[min(index, len(variants) - 1)])
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
                    plan=Plan(state["plan"]),
                    manual=settings["model"] != "auto",
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
                    "provider": getattr(deps.vision, "provider", "cloudflare"),
                    "p_unsafe": checked.p_unsafe,
                    "neurons": checked.neurons,
                    "error": checked.error,
                },
            )
        span.log(safety=verdict.value)
        # Every image that was made is delivered. Only a check that POSITIVELY found it unsafe
        # withholds it. A check that could not run (a provider limit) never costs the user an image.
        if verdict is Safety.UNSAFE:
            deps.store.delete(run_id, saved.file)
            deps.store.delete(run_id, saved.thumb)
            log.warning("image_dropped_by_safety", run_id=run_id, index=index, safety=verdict.value)
            span.end()
            emit(RunEvent(type="generated", run_id=run_id))  # it counts as done for progress
            return {"results": [{"ok": False, "index": index, "unsafe": True, "code": ""}]}
        span.end()
        # Not shown yet. The image is released to the user only after the run is saved.
        emit(RunEvent(type="generated", run_id=run_id))
        return {"results": [{"ok": True, **asdict(saved), "prompt": compiled.prompt}]}

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
            span.meta(
                prompt_version=base.version,
                model=deps.critic.model,
                provider=getattr(deps.critic, "provider", "cloudflare"),
            )
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

    def after_director(state: CreateState) -> str:
        if state.get("error"):
            return "fail"
        return "ask" if state.get("questions") else "prompt_writer"

    g = StateGraph(CreateState)
    g.add_node("code_checks", safe(code_checks))
    g.add_node("router", safe(router))
    g.add_node("art_director", safe(art_director))
    g.add_node("prompt_writer", safe(prompt_writer_node))
    g.add_node("ask", ask)  # not wrapped: interrupt() must reach the checkpointer
    g.add_node("code_guard", safe(code_guard))
    g.add_node("generate_one", generate_one_node)  # type: ignore[type-var]
    g.add_node("critic", safe(critic))
    g.add_node("settle", safe(settle))
    g.add_node("fail", fail)
    g.add_edge(START, "code_checks")
    g.add_conditional_edges("code_checks", _next("router"), ["router", "fail"])
    g.add_conditional_edges("router", _next("art_director"), ["art_director", "fail"])
    g.add_conditional_edges("art_director", after_director, ["ask", "prompt_writer", "fail"])
    g.add_edge("ask", "art_director")
    g.add_conditional_edges("prompt_writer", _next("code_guard"), ["code_guard", "fail"])
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
                brief=state["shot_plan"],
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
