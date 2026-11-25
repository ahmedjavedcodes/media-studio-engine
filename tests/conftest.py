"""Test-wide setup. Runs before any test module imports the app.

Tests must never call a real provider or post traces to LangSmith. Environment variables beat
`.env` in pydantic-settings, so an empty key here turns tracing off even when `.env` has one. (The
guardrail agent's own `trace()` calls and the app's telemetry both read this.)
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# The owner's dev switches in `.env` (paid providers, testing models, trace panel) must not leak
# into tests. Tests that need one turn it on themselves.
for _flag in (
    "OPENROUTER_IMAGES",
    "FAL_IMAGES",
    "GROQ_LLM",
    "GROQ_FALLBACK",
    "ALLOW_TESTING_MODELS",
    "TRACE_UI",
):
    os.environ[_flag] = "false"

# The OpenRouter adapter pauses a second before its one retry. Tests do not wait.
from app.gateway.providers import openrouter_images  # noqa: E402

openrouter_images.RETRY_PAUSE_S = 0.0
