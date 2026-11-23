"""Run the dummy graph once and send a trace to LangSmith.

Usage: PYTHONPATH=src uv run python playground/trace_check.py
"""

import asyncio
import sys

from app.agents.grand.graph import build_dummy_graph
from app.core.config import get_settings
from app.core.telemetry import setup_telemetry


async def main() -> int:
    settings = get_settings()
    if not setup_telemetry(settings):
        print("Tracing is OFF. Set LANGSMITH_API_KEY in .env (and LANGSMITH_TRACING=true).")
        return 1
    config = {
        "run_name": "trace_check",
        "tags": ["agent:dummy", "tier:dev", "model:none", "cost:0"],
        "metadata": {"provider_mode": settings.provider_mode},
    }
    out = await build_dummy_graph().ainvoke({"prompt": "trace check", "reply": ""}, config=config)
    print("reply:", out["reply"])
    print(f"Check the '{settings.langsmith_project}' project in LangSmith.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
