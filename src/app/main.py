"""FastAPI app."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.grand.graph import build_grand_graph
from app.agents.image_grand.generation.create.factory import build_create_deps
from app.api import catalog, events, health, run_service, runs
from app.api.checkpoint_cleanup import sweep_old_threads
from app.core.config import get_settings
from app.core.errors import EngineError
from app.core.logging import setup_logging
from app.core.telemetry import setup_telemetry
from app.guardrails.deps import close_guard_clients, get_guard_deps
from app.guardrails.warmup import warm_up

settings = get_settings()
setup_logging(settings.log_level)
setup_telemetry(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.guard_warmup:  # open warm connections to Jev and vision before the first request
        await warm_up(get_guard_deps())
    deps, http = build_create_deps(settings)
    await deps.prompts.load()  # pull the system prompts from the Hub (falls back to the files)
    # The checkpointer saves a run that paused at a question, so a later request can resume it.
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        await sweep_old_threads(saver)  # threads older than 24 h belong to runs that are gone
        app.state.create_deps = deps
        app.state.run_service = run_service.RunService(build_grand_graph(deps, saver), deps)
        yield
    deps.tracer.flush()
    await http.aclose()
    await close_guard_clients()


app = FastAPI(title="Media Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


@app.exception_handler(EngineError)
async def engine_error_handler(_: Request, exc: EngineError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content=exc.body(), headers=exc.headers())


app.include_router(health.router)
app.include_router(runs.router)
app.include_router(catalog.router)
app.include_router(events.router)
