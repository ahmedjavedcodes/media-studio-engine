"""TODO(backend): DEV ONLY. A throwaway FastAPI shim of the API contract in
project-status/ARCHITECTURE.md, section 8, so the frontend can be run against the real harness.

The harness is a library, and an API server is a non-goal of it: this file lives OUTSIDE
`src/media_engine`, nothing in the library imports it, and the teammate's backend replaces it.
It keeps everything in memory (runs, events) and stores images in ./.artifacts/.

    uv sync --group server                 # fastapi + uvicorn: an optional group, dev only
    DEV_FAKE=1 PYTHONPATH=src uvicorn --app-dir scripts dev_server:app --port 8000
    # then, in frontend/:  npm run dev:api     (NEXT_PUBLIC_MOCK=0)

DEBUG_TRACE=1 also serves GET /runs/:id/trace (the developer inspector channel; 404 without it)
and the run history: every finished run is saved as JSON in ./.runs/ (the last HISTORY_KEEP),
listed by GET /history and read back, in full, by GET /history/:id. That is the shim's local
stand-in for the real store of record (LangSmith); it is developer data, never product data.

DEV_FAKE=1 answers every model call from a scripted fake (scripts/dev_fake_backend.py): no .env,
no real calls. WITHOUT it this makes REAL provider calls with the keys in `.env`: agents on Groq
(free tier) first, images on Cloudflare Workers AI (free tier) first, as gateway/chains.yaml
orders them; paid providers stay off unless ALLOW_PAID=true is set in the environment.
Note: n is capped per plan (free 2, paid 4); create_images refuses more before any call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from media_engine.contracts import AuthContext, Envelope
from media_engine.harness.trace import Trace
from media_engine.ports import ArtifactNotFound, InMemoryRunStore, LocalArtifactStore
from media_engine.services.catalog import catalog
from media_engine.services.create.entry import create_images

# Tests inject a fake gateway here. None = the shared real Gateway (real provider calls).
GATEWAY: Any = None
if os.environ.get("DEV_FAKE") == "1":
    from dev_fake_backend import FakeBackend

    GATEWAY = FakeBackend()
RUN_STORE = InMemoryRunStore()  # the backend's cancel flag lives here (request_cancel)
ARTIFACTS = LocalArtifactStore("./.artifacts")
POLL_SECONDS = 0.1
# GET /runs/:id/trace (the developer inspector channel) is served only when DEBUG_TRACE=1.
DEBUG_TRACE = os.environ.get("DEBUG_TRACE") == "1"
# Run history (developer view): one JSON file per finished run, the newest HISTORY_KEEP kept.
HISTORY_DIR = Path(os.environ.get("RUN_HISTORY_DIR", "./.runs"))
HISTORY_KEEP = 50


class Run:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.task: asyncio.Task[None] | None = None
        self.trace = Trace()  # developer trace for this run, readable while it is running


RUNS: dict[str, Run] = {}


class Sink:
    """EventSink for one run: sync and fire-and-forget, as the port requires."""

    def __init__(self, run: Run) -> None:
        self._run = run

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        self._run.events.append(event)


class CreateBody(BaseModel):
    prompt: str = Field(min_length=1)
    plan: Literal["free", "pro", "max", "ultra"] = "free"
    mode: Literal["auto", "manual"] = "auto"
    n: int = Field(default=1, ge=1, le=4)
    settings: dict[str, str] = Field(default_factory=dict)
    # An explicit operation (a button or a slash command); None: inferred, today always create.
    operation: Literal["create", "edit", "merge", "restyle"] | None = None
    # A catalogue option id from GET /catalog (e.g. "openai/gpt-image-2@low"); None = auto.
    model: str | None = None


def to_result(run_id: str, env: Envelope) -> dict[str, Any]:
    """The Envelope as the contract's RunResult. Images become URLs, never bytes."""
    out = env.output
    ids: list[str] = list(out.get("image_ids", []))
    scores = out.get("quality_scores")
    if scores is None:  # one image: a single score, for the best image only
        scores = [out.get("quality_score")] + [None] * max(0, len(ids) - 1)
    return {
        "run_id": run_id,
        "status": env.state.status.value,
        "summary": env.summary,
        "reason": env.state.reason,
        "images": [
            {"image_id": i, "url": f"/images/{i}", "quality_score": s}
            for i, s in zip(ids, scores, strict=False)
        ],
        "issues": list(out.get("issues", [])),
        "questions": list(out.get("questions", [])),
        "assumptions": list(out.get("assumptions", [])),
    }


def running(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "running",
        "summary": "",
        "reason": None,
        "images": [],
        "issues": [],
        "questions": [],
        "assumptions": [],
    }


async def execute(run_id: str, body: CreateBody, run: Run) -> None:
    try:
        env = await create_images(
            body.prompt,
            auth=AuthContext(plan=body.plan, execution_mode=body.mode, model_demand="balanced"),
            n=body.n,
            settings=body.settings,
            run_id=run_id,
            gateway=GATEWAY,
            run_store=RUN_STORE,
            artifacts=ARTIFACTS,
            events=Sink(run),
            trace=run.trace,
            operation=body.operation,
            model=body.model,
        )
        run.result = to_result(run_id, env)
        output: dict[str, Any] = dict(env.output)
    except Exception:  # noqa: BLE001 - a dev shim: report any crash as a failed run
        run.result = {
            **running(run_id),
            "status": "failed",
            "summary": "The image could not be created.",
        }
        output = {}
    if DEBUG_TRACE:
        save_history(run_id, body, run, output)


def save_history(run_id: str, body: CreateBody, run: Run, output: dict[str, Any]) -> None:
    """Persist one finished run (input, result, the envelope's output with its documents, and
    the full developer trace), then drop the oldest beyond HISTORY_KEEP. Best effort: a disk
    problem never fails a run."""
    record = {
        "run_id": run_id,
        "request": body.model_dump(),
        "result": run.result,
        "output": output,
        "trace": run.trace.snapshot(),
    }
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (HISTORY_DIR / f"{run_id}.json").write_text(json.dumps(record, default=str))
        files = sorted(HISTORY_DIR.glob("run_*.json"), key=lambda f: f.stat().st_mtime)
        for old in files[:-HISTORY_KEEP]:
            with contextlib.suppress(OSError):
                old.unlink()
    except OSError as err:  # pragma: no cover - best effort
        print(f"dev_server: could not save run history: {err}", file=sys.stderr)


def history_row(record: dict[str, Any]) -> dict[str, Any]:
    """One line of the history list: what was asked, what came back, what it cost."""
    trace, request, result = record["trace"], record["request"], record["result"] or {}
    totals = trace.get("totals") or {}
    served = [
        c["model"] for c in trace.get("calls", []) if c.get("kind") == "image" and c.get("model")
    ]
    return {
        "run_id": record["run_id"],
        "created_at": trace.get("created_at"),
        "prompt": request.get("prompt"),
        "plan": request.get("plan"),
        "operation": request.get("operation") or "create",
        "n": request.get("n"),
        "model_pick": request.get("model") or "auto",
        "models": sorted(set(served)),
        "status": result.get("status"),
        "cost_usd": totals.get("cost_usd", 0.0),
        "neurons": totals.get("neurons", 0.0),
        "llm_calls": totals.get("llm_calls", 0),
        "wall_ms": trace.get("wall_ms"),
    }


app = FastAPI(title="media-studio dev shim (TODO(backend))")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_run(run_id: str) -> Run:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    return run


@app.get("/catalog")
async def get_catalog(plan: Literal["free", "pro", "max", "ultra"] = "free") -> dict[str, Any]:
    """What the prompt bar may offer this plan (services/catalog.py). TODO(backend): the real
    endpoint is the backend's; locks are UX only, the gateway enforces the plan on every call."""
    return catalog(plan).model_dump()


@app.post("/runs", status_code=202)
async def post_run(body: CreateBody) -> dict[str, str]:
    run_id = f"run_{uuid.uuid4().hex}"
    run = RUNS[run_id] = Run()
    run.task = asyncio.create_task(execute(run_id, body, run))
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
async def get_run_result(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    return run.result or running(run_id)


@app.get("/runs/{run_id}/trace")
async def get_trace(run_id: str) -> dict[str, Any]:
    """Developer inspector data: every model call, every decision, steps, attempts, budget.
    A SEPARATE channel from the events and the result: 404 unless DEBUG_TRACE=1."""
    if not DEBUG_TRACE:
        raise HTTPException(404, "tracing is off (set DEBUG_TRACE=1)")
    return get_run(run_id).trace.snapshot()


def _history_on() -> None:
    if not DEBUG_TRACE:
        raise HTTPException(404, "run history is off (set DEBUG_TRACE=1)")


@app.get("/history")
async def list_history() -> list[dict[str, Any]]:
    """Recent finished runs, newest first (developer view: 404 unless DEBUG_TRACE=1)."""
    _history_on()
    if not HISTORY_DIR.exists():
        return []
    files = sorted(HISTORY_DIR.glob("run_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    rows = []
    for file in files[:HISTORY_KEEP]:
        try:
            rows.append(history_row(json.loads(file.read_text())))
        except (OSError, ValueError, KeyError):
            continue  # a half-written or foreign file: skip it
    return rows


@app.get("/history/{run_id}")
async def get_history(run_id: str) -> dict[str, Any]:
    """One finished run in full: request, result, output (documents), trace."""
    _history_on()
    if not run_id.startswith("run_") or "/" in run_id or ".." in run_id:
        raise HTTPException(404, "unknown run")
    file = HISTORY_DIR / f"{run_id}.json"
    if not file.exists():
        raise HTTPException(404, "unknown run")
    record: dict[str, Any] = json.loads(file.read_text())
    return record


@app.get("/runs/{run_id}/events")
async def get_events(run_id: str) -> StreamingResponse:
    run = get_run(run_id)

    async def stream() -> AsyncIterator[str]:
        sent = 0
        while True:
            while sent < len(run.events):  # replayed from the start for a late subscriber
                yield f"event: step\ndata: {json.dumps(run.events[sent])}\n\n"
                sent += 1
            if run.result is not None and sent >= len(run.events):
                yield f"event: result\ndata: {json.dumps(run.result)}\n\n"
                return
            await asyncio.sleep(POLL_SECONDS)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.post("/runs/{run_id}/cancel", status_code=202)
async def cancel(run_id: str) -> dict[str, Any]:
    get_run(run_id)
    await RUN_STORE.request_cancel(run_id)  # the backend's side of the store; never via save()
    return {}


@app.get("/images/{image_id}")
async def get_image(image_id: str) -> FileResponse:
    try:
        path = Path(unquote(urlparse(await ARTIFACTS.url(image_id)).path))
    except (ValueError, ArtifactNotFound):
        raise HTTPException(404, "unknown image") from None
    return FileResponse(path)


__all__ = ["app"]
