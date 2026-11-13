"""One Gateway.chat call with LangSmith tracing on, so you can confirm the trace.

    uv run python playground/trace_check.py

Prints the LangSmith project name and the run id to look for. Prompts, responses and keys
are never sent in the trace (only task name and call summary).
"""

from __future__ import annotations

import asyncio
import uuid

from media_engine.core.config import get_settings
from media_engine.core.logging import configure_logging
from media_engine.core.telemetry import init_telemetry
from media_engine.gateway.router import Gateway
from media_engine.gateway.types import LLMRequest


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    enabled = init_telemetry(settings)
    print(f"tracing_enabled={enabled} project={settings.langsmith_project}")
    if not enabled:
        print("Tracing is OFF: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env")
        return

    run_id = uuid.uuid4()
    gateway = Gateway(settings)
    try:
        result = await gateway.chat(
            "fast_llm",
            LLMRequest(messages=[{"role": "user", "content": "ping"}], max_tokens=32),
            langsmith_extra={"run_id": run_id},  # type: ignore[call-arg]
        )
    finally:
        await gateway.aclose()

    from langsmith import Client

    Client().flush()  # make sure the background trace upload finishes before exit
    print(f"call ok: {result.provider}/{result.model} in {result.latency_ms:.0f} ms")
    print(f"LangSmith project: {settings.langsmith_project}")
    print(f"run name: gateway.chat   run id: {run_id}")


if __name__ == "__main__":
    asyncio.run(main())
