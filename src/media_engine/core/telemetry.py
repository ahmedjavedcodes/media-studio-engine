"""LangSmith tracing.

Tracing is off unless `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` is set. Callers pass
`process_inputs` / `process_outputs` so prompts, images and keys never leave the process
in a trace unless a caller explicitly chooses to include them.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Awaitable, Callable
from typing import Any, Literal, ParamSpec, TypeVar, cast

from media_engine.core.config import Settings, get_settings

P = ParamSpec("P")
R = TypeVar("R")

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]

_enabled = False


def init_telemetry(settings: Settings | None = None) -> bool:
    """Configure LangSmith from settings. Returns whether tracing is active."""
    global _enabled
    settings = settings or get_settings()
    key = settings.langsmith_api_key
    _enabled = bool(settings.langsmith_tracing and key)
    if _enabled and key is not None:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
    return _enabled


def tracing_enabled() -> bool:
    return _enabled


def traced(
    name: str,
    *,
    run_type: RunType = "chain",
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    process_outputs: Callable[[Any], dict[str, Any]] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Trace an async function in LangSmith when tracing is enabled; otherwise a no-op."""

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _enabled:
                return await fn(*args, **kwargs)
            from langsmith import traceable

            wrapped = traceable(
                run_type,
                name=name,
                process_inputs=process_inputs,
                process_outputs=process_outputs,
            )(fn)
            call = cast("Callable[..., Awaitable[R]]", wrapped)
            return await call(*args, **kwargs)

        return wrapper

    return decorator
