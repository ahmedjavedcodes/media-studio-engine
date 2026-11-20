"""TODO(backend): DEV ONLY. A scripted stand-in for the provider Gateway, so the frontend can be
used against `dev_server.py` with NO .env and NO real model calls (`DEV_FAKE=1`).

It answers the router, enhancer, critic and lever calls with canned JSON, and draws a placeholder
image. Words in the prompt pick the outcome:

    "complex"  -> the gate says complex    -> recorded only: still done (escalate needs an
                                           explicit stub operation, e.g. "operation": "edit")
    "unsafe"   -> the gate blocks           -> rejected before any image
    "borderline" -> the gate says review, the reviewer allows -> continues
    "reject"   -> the critic blocks         -> rejected
    "fail"     -> no image provider         -> failed
    "question" -> the enhancer asks "Which city should the street be in?" (use manual mode)
    "issues"   -> the critic scores 4       -> retries, then done_with_issues
    anything else                           -> done (score 9)

Each call is reported as served by the FIRST model of its real chain (gateway/chains.yaml), with
" (fake)" after the provider name, so the inspector's provider split looks like production (Groq
agents, Cloudflare images) while making clear nothing real was called. Neurons are reported only
for Cloudflare, as in production.

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
from media_engine.gateway.router import load_chains
from media_engine.gateway.types import CallResult, ImageResponse, LLMResponse, ModelRef

QUESTION = "Which city should the street be in?"
_REQUEST = re.compile(r"<user_request>\s*(.*?)\s*</user_request>", re.DOTALL)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


_CHAINS = load_chains()


def _served_by(task: str) -> ModelRef:
    """The model production would try first for this chain, marked as fake."""
    first = next((r for r in _CHAINS.get(task, []) if not r.is_placeholder), None)
    if first is None:
        return ModelRef("dev-fake", "dev-fake")
    return ModelRef(f"{first.provider} (fake)", first.model)


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

        if "intake and safety classifier" in system:
            safety = "block" if "unsafe" in low else "review" if "borderline" in low else "allow"
            reply: dict[str, Any] = {
                "safety": safety,
                "safety_categories": ["dev fake"] if safety != "allow" else [],
                "route": "complex" if "complex" in low else "simple",
                "task_type": "create",
                "prompt_is_detailed": False,
            }
        elif "safety policy reviewer" in system:
            reply = {"safety": "allow", "safety_categories": ["dev fake"]}
        elif "images" in ((request.response_schema or {}).get("properties") or {}):  # the Critic
            blocked = "reject" in low
            score = 4 if "issues" in low else 9
            count = sum(1 for p in request.messages[1]["content"] if p.get("type") == "image_url")
            reply = {
                "images": [
                    {
                        "index": i,
                        "prompt_adherence": score,
                        "craft": score,
                        "visual_interest": score,
                        "technical": score,
                        "safety_verdict": "block" if blocked else "allow",
                        "safety_reason": "dev fake block" if blocked else None,
                        "rank": None if blocked else i,
                        "critique": "The lamps are slightly blown out." if score < 6 else "",
                    }
                    for i in range(1, count + 1)
                ]
            }
        else:  # the enhancer
            asked = "question" in low and QUESTION not in user
            reply = {
                "subject": prompt,
                "lighting": "soft window light from the left",
                "detail_anchors": ["wooden table", "warm colours"],
                "must_have": [" ".join(prompt.split()[:3])],
                "missing_info": [{"kind": "question", "question": QUESTION}] if asked else [],
                "text_wanted": [],
                "summary": "A detailed prompt.",
            }
        body = json.dumps(reply)
        usage = {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
        ref = _served_by(task)
        return CallResult(
            output=LLMResponse(text=body, usage=usage),
            task=task,
            provider=ref.provider,
            model=ref.model,
            latency_ms=1.0,
            is_fallback=False,
            neurons=2.0 if ref.provider.startswith("cloudflare") else None,
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
        ref = _served_by(task)
        return CallResult(
            output=ImageResponse(images=[buffer.getvalue()], media_types=["image/png"]),
            task=task,
            provider=ref.provider,
            model=ref.model,
            latency_ms=1.0,
            is_fallback=False,
            neurons_est=57.6 if ref.provider.startswith("cloudflare") else None,
        )
