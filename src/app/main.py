"""FastAPI app."""

from fastapi import FastAPI

from app.api import events, health, runs
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.telemetry import setup_telemetry

settings = get_settings()
setup_logging(settings.log_level)
setup_telemetry(settings)

app = FastAPI(title="Media Engine")
app.include_router(health.router)
app.include_router(runs.router)
app.include_router(events.router)
