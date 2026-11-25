"""Runs the graph in the background and keeps each run's events, so a closed or re-opened SSE
stream loses nothing. One RunService per process.

A run starts at POST (after the guardrail). Its events go into a log. `GET .../events` reads the
log. When the graph asks a question the phase ends and the stream closes. `resume` starts the
next phase on the same checkpointed thread. Traces, scores and provider errors never enter the
log (see core/user_errors.py).

States: running -> waiting (a question) -> resuming -> running -> done | failed, or waiting ->
expired. Every change out of `waiting` happens under the run's lock, so a double tap or a timeout
that fires during a resume cannot start the graph twice or run it without credits.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import structlog
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.api.checkpoint_cleanup import forget_thread
from app.contracts.events import OptionOut, QuestionOut, RunEvent, status_event
from app.contracts.request import GenerationRequest
from app.core.user_errors import INTERNAL_MESSAGE
from app.gateway.options import RequestExtras, options_for

log = structlog.get_logger("runs")

Status = Literal["running", "waiting", "resuming", "done", "failed", "expired"]
MAX_KEPT_RUNS = 500


class RunNotWaiting(Exception):
    """The run is not paused at a question (already resumed, finished or timed out)."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass
class RunRecord:
    run_id: str
    user_id: str
    request: GenerationRequest
    status: Status = "running"
    events: list[RunEvent] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timer: asyncio.TimerHandle | None = None
    task: asyncio.Task[None] | None = None
    expire_task: asyncio.Task[None] | None = None  # kept so it is not garbage-collected
    images_seen: int = 0
    hold_id: str | None = None
    crashed: bool = False  # the driver raised. The run must never be resumed

    @property
    def phase_over(self) -> bool:
        return self.status in ("waiting", "done", "failed", "expired")


def _has_news(record: RunRecord, seen: int) -> bool:
    return seen < len(record.events) or record.phase_over


class RunService:
    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any], deps: CreateDeps) -> None:
        self._graph = graph
        self._deps = deps
        self.runs: dict[str, RunRecord] = {}

    # ---- start and resume --------------------------------------------------------------------

    def preflight(self, request: GenerationRequest) -> None:
        """Cheap checks before anything runs (and before any cost). A picked model must take this
        many reference images, this quality, transparent background and brand colors. Raises
        BadInput, which is a 400. `auto` is checked by the router, which only picks a model that
        fits."""
        cfg = request.settings
        card = self._deps.registry.get(cfg.model) if cfg.model != "auto" else None
        if card is None:
            return
        options_for(
            card,
            RequestExtras(
                references=("x",) * len(request.images),
                brand_colors=tuple(cfg.brand_colors),
                transparent_bg=cfg.transparent_bg,
            ),
        )

    def start(self, run_id: str, user_id: str, request: GenerationRequest) -> RunRecord:
        self._prune()
        meta = {
            "run_id": run_id,
            "user_id": user_id,
            "plan": request.plan.value,
            "model": request.settings.model,  # becomes the real model after the router
            "provider": None,
            "compile_mode": self._deps.settings.compile_mode,
        }
        self._deps.tracer.start_run(
            run_id,
            metadata=meta,
            inputs={"prompt": request.prompt, "settings": request.settings.model_dump()},
        )
        record = RunRecord(run_id=run_id, user_id=user_id, request=request)
        self.runs[run_id] = record
        graph_input: dict[str, Any] = {
            "run_id": run_id,
            "user_id": user_id,
            "prompt": request.prompt,
            "plan": request.plan.value,
            "settings": request.settings.model_dump(),
            "has_images": bool(request.images),
            "n_refs": len(request.images),
            "results": [],
        }
        record.task = asyncio.create_task(self._drive(record, graph_input))
        return record

    async def resume(self, run_id: str, answers: dict[str, str]) -> None:
        """Raises RunNotWaiting unless the run is paused at a question right now."""
        record = self.runs[run_id]
        async with record.lock:
            if record.status != "waiting":  # re-checked here, after any slow guardrail call
                raise RunNotWaiting(record.status)
            async with record.cond:
                # One step: a reader that connects now sees no old question and waits for the
                # new phase. A second resume sees `resuming` and is refused.
                record.status = "resuming"
                record.events.clear()
                record.questions = []
            if record.timer is not None:
                record.timer.cancel()
        try:
            if record.task is not None:
                await record.task  # the paused phase must be fully closed first
        except BaseException:  # for example a cancel at shutdown
            await self._abandon(record, error="resume interrupted")
            raise
        if record.crashed:  # it crashed while we were waiting for it. Do not run on without a hold
            await self._push(record, self._internal_error(record))
            async with record.cond:
                record.status = "failed"
                record.cond.notify_all()
            self._deps.tracer.get(run_id).finish(error="crashed")
            self._deps.tracer.drop(run_id)
            await self._forget(record)
            raise RunNotWaiting("failed")
        span = self._deps.tracer.get(run_id).span_named("interrupt")
        if span is not None:
            span.log(answers=answers)
            span.end()
        async with record.cond:
            record.status = "running"
        record.task = asyncio.create_task(self._drive(record, Command(resume=answers)))

    # ---- the driver --------------------------------------------------------------------------

    async def _drive(self, record: RunRecord, graph_input: Any) -> None:
        config = {"configurable": {"thread_id": record.run_id}}
        try:
            # subgraphs=True: the txt2img sub-agent's events reach this stream too.
            stream = self._graph.astream(
                graph_input,
                config,
                stream_mode=["custom", "updates"],
                subgraphs=True,
            )  # type: ignore[call-overload]
            interrupted = False
            async with contextlib.aclosing(stream):
                async for _ns, mode, chunk in stream:
                    if mode == "updates" and isinstance(chunk, dict):
                        if "router" in chunk:  # credits are held in the router step
                            record.hold_id = (chunk["router"] or {}).get("hold_id")
                        if "__interrupt__" in chunk and not interrupted:
                            interrupted = True  # the parent and the sub-agent both report it
                            await self._on_interrupt(record, chunk["__interrupt__"])
                    elif mode == "custom":
                        await self._on_event(record, RunEvent.model_validate(chunk))
        except Exception as exc:  # noqa: BLE001 - the stream must always end cleanly
            log.error("run_crashed", run_id=record.run_id, error=type(exc).__name__)
            record.crashed = True
            async with record.lock:
                # A crash after a question must not leave a resumable run with no hold.
                if record.timer is not None:
                    record.timer.cancel()
                resuming = record.status == "resuming"  # resume() will push the error itself
                if record.status == "waiting":
                    record.status = "running"  # `_finish` below ends it as failed
            await self._release_hold(record)
            self._deps.store.purge(record.run_id)
            if not resuming:
                await self._push(record, self._internal_error(record))
        await self._finish(record)

    async def _on_event(self, record: RunRecord, event: RunEvent) -> None:
        if event.type == "generated":  # one image is done. It is not shown until it is released
            record.images_seen += 1
            n = record.request.settings.n
            await self._push(
                record,
                status_event(record.run_id, "creating", progress=f"{record.images_seen}/{n}"),
            )
            return
        await self._push(record, event)

    async def _on_interrupt(self, record: RunRecord, interrupts: Any) -> None:
        value = interrupts[0].value
        record.questions = value["questions"]
        questions = [
            QuestionOut(
                id=q["id"],
                text=q["text"],
                options=[OptionOut(**o) for o in q["options"]],
            )
            for q in record.questions
        ]
        self._deps.tracer.get(record.run_id).start(
            "interrupt", inputs={"questions": [q["text"] for q in record.questions]}
        )
        async with record.cond:  # the question and the new status become visible together
            record.events.append(
                RunEvent(type="question", run_id=record.run_id, questions=questions)
            )
            record.status = "waiting"
            record.cond.notify_all()
        loop = asyncio.get_running_loop()
        record.timer = loop.call_later(
            self._deps.settings.draft_timeout_s, partial(self._start_expire, record)
        )

    def _start_expire(self, record: RunRecord) -> None:
        record.expire_task = asyncio.create_task(self._expire(record))

    async def _expire(self, record: RunRecord) -> None:
        """Nobody answered in time. The run becomes a draft and nothing is charged."""
        async with record.lock:
            if record.status != "waiting":  # a resume got there first. Do nothing
                return
            record.status = "expired"
        await self._release_hold(record)
        self._deps.store.purge(record.run_id)
        async with record.cond:
            record.events.append(
                RunEvent(
                    type="expired",
                    run_id=record.run_id,
                    code="draft_expired",
                    message="This request timed out waiting for your answer. Tap to continue.",
                )
            )
            record.cond.notify_all()
        self._deps.tracer.get(record.run_id).finish(error="draft timeout")
        self._deps.tracer.drop(record.run_id)
        await self._forget(record)

    async def _release_hold(self, record: RunRecord) -> None:
        if record.hold_id:
            await self._deps.credits.release(record.hold_id)  # safe to call twice

    async def _finish(self, record: RunRecord) -> None:
        if record.status != "running":  # paused at a question (or already resuming or expired)
            async with record.cond:  # the run is not over. The trace stays open
                record.cond.notify_all()
            return
        last = record.events[-1] if record.events else None
        if last is None or last.type not in ("ready", "error"):
            # A phase that ends with only status events is a bug, not a success.
            log.error("run_ended_without_result", run_id=record.run_id)
            await self._release_hold(record)
            self._deps.store.purge(record.run_id)
            last = self._internal_error(record)
            await self._push(record, last)
        failed = last.type == "error"
        async with record.cond:
            record.status = "failed" if failed else "done"
        trace = self._deps.tracer.get(record.run_id)
        trace.finish(
            outputs={"status": record.status, "images": record.images_seen},
            error=(last.code if failed and last else None),
        )
        self._deps.tracer.drop(record.run_id)
        await self._forget(record)  # done or failed: the saved graph state is not needed any more
        async with record.cond:
            record.cond.notify_all()

    async def _forget(self, record: RunRecord) -> None:
        """The run is over. Its saved graph state is no longer needed."""
        await forget_thread(getattr(self._graph, "checkpointer", None), record.run_id)

    async def _abandon(self, record: RunRecord, *, error: str) -> None:
        """End a run that cannot continue: failed, hold released, files gone, trace closed."""
        await self._release_hold(record)
        self._deps.store.purge(record.run_id)
        async with record.cond:
            record.status = "failed"
            record.cond.notify_all()
        self._deps.tracer.get(record.run_id).finish(error=error)
        self._deps.tracer.drop(record.run_id)
        await self._forget(record)

    @staticmethod
    def _internal_error(record: RunRecord) -> RunEvent:
        return RunEvent(
            type="error", run_id=record.run_id, code="internal_error", message=INTERNAL_MESSAGE
        )

    async def _push(self, record: RunRecord, event: RunEvent) -> None:
        async with record.cond:
            record.events.append(event)
            record.cond.notify_all()

    def _prune(self) -> None:
        """Forget the oldest finished runs. Their saved results stay on disk."""
        if len(self.runs) < MAX_KEPT_RUNS:
            return
        for run_id in [
            r for r, rec in self.runs.items() if rec.phase_over and rec.status != "waiting"
        ]:
            if len(self.runs) < MAX_KEPT_RUNS:
                break
            del self.runs[run_id]

    # ---- reading -----------------------------------------------------------------------------

    async def stream(self, run_id: str) -> AsyncIterator[RunEvent]:
        """Every event of the current phase, then the stream ends (a question, ready or error)."""
        record = self.runs[run_id]
        seen = 0
        while True:
            async with record.cond:
                await record.cond.wait_for(partial(_has_news, record, seen))
                batch = record.events[seen:]
                over = record.phase_over
            for event in batch:
                yield event
            seen += len(batch)
            if over and seen >= len(record.events):
                return

    async def wait(self, run_id: str) -> None:
        """Wait for the current phase to finish. For tests."""
        task = self.runs[run_id].task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
