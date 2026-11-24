"""FastAPI app."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import events, health, runs
from app.core.config import get_settings
from app.core.errors import EngineError
from app.core.logging import setup_logging
from app.core.telemetry import setup_telemetry

settings = get_settings()
setup_logging(settings.log_level)
setup_telemetry(settings)

app = FastAPI(title="Media Engine")
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
app.include_router(events.router)
