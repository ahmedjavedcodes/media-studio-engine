"""LangSmith setup. Tracing turns on only when a key is set."""

import os

from app.core.config import Settings


def setup_telemetry(settings: Settings) -> bool:
    enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return enabled
