"""TODO(backend): DEV ONLY. A scripted stand-in for the provider Gateway, so the frontend can be
used against `dev_server.py` with NO .env and NO real model calls (`DEV_FAKE=1`).

It answers the router, enhancer, critic and lever calls with canned JSON, and draws a placeholder
image. Words in the prompt pick the outcome:

    "complex"  -> the router says complex  -> escalate
    "reject"   -> the critic blocks         -> rejected
    "fail"     -> no image provider         -> failed
    "question" -> the enhancer asks "Which city should the street be in?" (use manual mode)
    "issues"   -> the critic scores 4       -> retries, then done_with_issues
    anything else                           -> done (score 9)

Lives outside `src/media_engine`; nothing in the library imports it.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from typing import Any

from PIL import Image, ImageDraw

from media_engine.gateway.errors import AllProvidersFailed
from media_engine.gateway.types import CallResult, ImageResponse, LLMResponse

QUESTION = "Which city should the street be in?"
_REQUEST = re.compile(r"<user_request>\s*(.*?)\s*</user_request>", re.DOTALL)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


class FakeBackend:
    """Both halves of the Gateway the harness needs: `chat` and `image`."""

    def __init__(self, delay: float = 0.6) -> None:
        self.delay = delay  # per call, so the progress stepper is visible

    async def chat(self, task: str, request: Any) -> CallResult:
        await asyncio.sleep(self.delay)
        system = request.messages[0]["content"]
        user = _text(request.messages[1]["content"])
        found = _REQUEST.search(user)
        prompt = (found.group(1) if found else user).strip()
        low = prompt.lower()

        if "request classifier" in system:
            reply: dict[str, Any] = {"route": "complex" if "complex" in low else "simple"}
        elif "retry planner" in system:
            reply = {"lever": "reseed", "reason": "dev fake"}
        elif "reviewer for an image generation studio" in system:
            blocked = "reject" in low
            reply = {
                "quality_score": 4 if "issues" in low else 9,
                "safety_verdict": "block" if blocked else "allow",
                "safety_reason": "dev fake block" if blocked else None,
                "critique": "The lamps are slightly blown out.",
                "summary": "A placeholder image.",
            }
        else:  # the enhancer
            asked = "question" in low and QUESTION not in user
            reply = {
                "enhanced_prompt": f"{prompt}, detailed, well lit",
                "must_have": [" ".join(prompt.split()[:3])],
                "missing_info": [{"kind": "question", "question": QUESTION}] if asked else [],
                "text_wanted": [],
                "summary": "A detailed prompt.",
            }
        body = json.dumps(reply)
        usage = {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
        return CallResult(
            output=LLMResponse(text=body, usage=usage),
            task=task,
            provider="dev-fake",
            model="dev-fake",
            latency_ms=1.0,
            is_fallback=False,
            neurons=2.0,
        )

    async def image(self, task: str, request: Any, **kwargs: Any) -> CallResult:
        await asyncio.sleep(self.delay)
        if "fail" in request.prompt.lower():
            raise AllProvidersFailed(task, [])
        digest = hashlib.sha256(f"{request.prompt}{request.seed}".encode()).digest()
        color = (80 + digest[0] % 150, 80 + digest[1] % 150, 80 + digest[2] % 150)
        picture = Image.new("RGB", (512, 512), color)
        draw = ImageDraw.Draw(picture)
        draw.ellipse((156, 106, 356, 306), fill=(255, 255, 255))
        draw.text((150, 400), f"dev fake  seed {request.seed}", fill=(255, 255, 255))
        buffer = io.BytesIO()
        picture.save(buffer, "PNG")
        return CallResult(
            output=ImageResponse(images=[buffer.getvalue()]),
            task=task,
            provider="dev-fake",
            model="dev-fake",
            latency_ms=1.0,
            is_fallback=False,
            neurons_est=57.6,
        )
