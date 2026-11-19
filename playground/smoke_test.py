"""Real calls to confirm model IDs, credentials and request formats.

    uv run python playground/smoke_test.py                       # free checks only
    uv run python playground/smoke_test.py --only cf-flux2-merge
    uv run python playground/smoke_test.py --only openrouter-image   # COSTS REAL MONEY (tiny)

Calls providers directly (not through the gateway) on purpose: the gateway would hide a wrong
model ID behind a fallback. Playground script, never run by tests. The paid OpenRouter check
only runs when asked for by name. Prints model, latency, status, cost and response field
NAMES; never keys or full responses.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from media_engine.core.config import get_settings
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.providers.base import Provider
from media_engine.gateway.providers.cloudflare import CloudflareProvider
from media_engine.gateway.providers.groq import GroqProvider
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.types import ImageRequest, LLMRequest

ASSETS = Path(__file__).parent / "assets"
PROMPT = [{"role": "user", "content": "Reply with exactly: pong"}]

CF_LLM = "@cf/openai/gpt-oss-20b"
CF_IMG = "@cf/black-forest-labs/flux-1-schnell"
CF_FLUX2 = "@cf/black-forest-labs/flux-2-klein-4b"
CF_VISION_SCOUT = "@cf/meta/llama-4-scout-17b-16e-instruct"
CF_VISION_GEMMA = "@cf/google/gemma-3-12b-it"
GROQ_LLM = "openai/gpt-oss-20b"
OR_IMG = "google/gemini-2.5-flash-image"


@dataclass
class Check:
    name: str
    label: str
    provider: Provider
    model: str
    run: Callable[[], Awaitable[str]]
    paid: bool = False


async def _execute(check: Check) -> bool:
    if not check.provider.is_configured():
        print(f"SKIP  {check.label:<24} {check.model}  (missing key in .env)")
        return False
    start = time.perf_counter()
    try:
        detail = await check.run()
    except ProviderError as err:
        ms = (time.perf_counter() - start) * 1000
        print(
            f"FAIL  {check.label:<24} {check.model}  {ms:7.0f} ms  "
            f"status={err.status} code={err.code} timeout={err.timeout} :: {err.message[:200]}"
        )
        return False
    ms = (time.perf_counter() - start) * 1000
    print(f"OK    {check.label:<24} {check.model}  {ms:7.0f} ms  {detail}")
    return True


def _image_detail(r: object, status: str = "200") -> str:
    images = r.images  # type: ignore[attr-defined]
    cost = getattr(r, "cost_usd", None)
    extra = f" cost_usd={cost}" if cost is not None else ""
    return f"status={status} images={len(images)} bytes={len(images[0])}{extra}"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated check names (see --list)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    async with httpx.AsyncClient() as client:
        cf = CloudflareProvider(client, settings)
        groq = GroqProvider(client, settings)
        orr = OpenRouterProvider(client, settings)

        async def cf_chat() -> str:
            r = await cf.chat(CF_LLM, LLMRequest(messages=PROMPT, max_tokens=64))
            return f"reply={r.text.strip()[:40]!r} usage={r.usage}"

        async def cf_image() -> str:
            r = await cf.generate_image(
                CF_IMG, ImageRequest(prompt="a red apple on a white table", steps=4)
            )
            return f"image_bytes={len(r.images[0])}"

        async def cf_edit() -> str:
            # multipart form data: prompt + width/height + ONE input image (input_image_0)
            r = await cf.generate_image(
                CF_FLUX2,
                ImageRequest(
                    prompt="make the background light blue",
                    width=512,
                    height=512,
                    input_images=[(ASSETS / "test.png").read_bytes()],
                ),
            )
            return _image_detail(r) + " response_fields=result.image"

        async def cf_merge() -> str:
            # TWO input images (input_image_0, input_image_1) + a combine prompt
            r = await cf.generate_image(
                CF_FLUX2,
                ImageRequest(
                    prompt=(
                        "combine the two images into one picture: "
                        "a grey square and a red square side by side"
                    ),
                    width=512,
                    height=512,
                    input_images=[
                        (ASSETS / "test.png").read_bytes(),
                        (ASSETS / "test2.png").read_bytes(),
                    ],
                ),
            )
            return _image_detail(r) + " response_fields=result.image"

        def vision_check(model: str) -> Callable[[], Awaitable[str]]:
            # OpenAI-style multimodal message: text part + image_url part with a data URL.
            # test2.png is a solid red 256x256 square, so the right answer is unambiguous.
            data = base64.b64encode((ASSETS / "test2.png").read_bytes()).decode()

            async def run() -> str:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What colour is this image? Reply with JSON only: "
                                '{"colour": "<one word>"}',
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{data}"},
                            },
                        ],
                    }
                ]
                r = await cf.chat(model, LLMRequest(messages=messages, max_tokens=64))
                return f"reply={r.text.strip()[:60]!r} usage={r.usage}"

            return run

        async def cf_seed() -> str:
            # Same prompt + same seed twice. Compare decoded PIXELS, not file bytes, so encoder
            # metadata can't hide a match. Identical pixels => the seed is honoured.
            from PIL import Image

            req = ImageRequest(prompt="a red apple on a white table", steps=4, seed=42)
            digests = []
            for _ in range(2):
                r = await cf.generate_image(CF_IMG, req)
                with Image.open(io.BytesIO(r.images[0])) as img:
                    digests.append(hashlib.sha256(img.convert("RGB").tobytes()).hexdigest())
            same = digests[0] == digests[1]
            return f"seed=42 x2 pixels_identical={same} sha={digests[0][:12]},{digests[1][:12]}"

        async def groq_chat() -> str:
            r = await groq.chat(
                GROQ_LLM, LLMRequest(messages=PROMPT, max_tokens=64, reasoning_effort="low")
            )
            return f"reply={r.text.strip()[:40]!r} usage={r.usage}"

        async def or_image() -> str:
            # Tiny on purpose: n=1, 1:1, simple prompt (gemini-2.5-flash-image has no resolution
            # parameter and a fixed output size). Costs real money.
            r = await orr.generate_image(
                OR_IMG,
                ImageRequest(
                    prompt="a single red dot on a white background", aspect_ratio="1:1", n=1
                ),
            )
            return _image_detail(r) + " response_fields=data[].b64_json,usage.cost"

        checks = [
            Check("cf-llm", "cloudflare LLM", cf, CF_LLM, cf_chat),
            Check("cf-image", "cloudflare image", cf, CF_IMG, cf_image),
            Check("cf-flux2-edit", "cloudflare FLUX.2 edit", cf, CF_FLUX2, cf_edit),
            Check("cf-flux2-merge", "cloudflare FLUX.2 merge", cf, CF_FLUX2, cf_merge),
            Check(
                "cf-vision-scout",
                "cloudflare vision",
                cf,
                CF_VISION_SCOUT,
                vision_check(CF_VISION_SCOUT),
            ),
            Check(
                "cf-vision-gemma",
                "cloudflare vision",
                cf,
                CF_VISION_GEMMA,
                vision_check(CF_VISION_GEMMA),
            ),
            Check("cf-seed", "cloudflare image seed", cf, CF_IMG, cf_seed),
            Check("groq-llm", "groq LLM", groq, GROQ_LLM, groq_chat),
            Check("openrouter-image", "openrouter image (PAID)", orr, OR_IMG, or_image, paid=True),
        ]
        if args.list:
            for c in checks:
                print(f"{c.name:<18}{'PAID, opt-in only' if c.paid else ''}")
            return
        if args.only:
            wanted = {n.strip() for n in args.only.split(",")}
            unknown = wanted - {c.name for c in checks}
            if unknown:
                raise SystemExit(f"unknown check(s): {sorted(unknown)}")
            selected = [c for c in checks if c.name in wanted]
        else:
            selected = [c for c in checks if not c.paid]

        results = [await _execute(c) for c in selected]
    print(f"\n{sum(results)}/{len(results)} succeeded")


if __name__ == "__main__":
    asyncio.run(main())
