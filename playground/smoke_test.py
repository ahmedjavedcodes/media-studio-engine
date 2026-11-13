"""ONE real call per provider/model, to confirm model IDs and credentials.

    uv run python playground/smoke_test.py

Calls providers directly (not through the gateway) on purpose: the gateway would hide a wrong
model ID behind a fallback. This is a playground script, not agent code, and it is never run
by tests. It spends a little of the free quota (one short LLM call each, one 4-step image).
Prints model, latency and success only; never prints keys or full responses.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx

from media_engine.core.config import get_settings
from media_engine.gateway.errors import ProviderError
from media_engine.gateway.providers.base import Provider
from media_engine.gateway.providers.cloudflare import CloudflareProvider
from media_engine.gateway.providers.groq import GroqProvider
from media_engine.gateway.types import ImageRequest, LLMRequest

TEST_IMAGE = Path(__file__).parent / "assets" / "test.png"
PROMPT = [{"role": "user", "content": "Reply with exactly: pong"}]


async def _check(label: str, provider: Provider, model: str, call) -> bool:  # type: ignore[no-untyped-def]
    if not provider.is_configured():
        print(f"SKIP  {label:<24} {model}  (missing key in .env)")
        return False
    start = time.perf_counter()
    try:
        detail = await call()
    except ProviderError as err:
        ms = (time.perf_counter() - start) * 1000
        print(
            f"FAIL  {label:<24} {model}  {ms:7.0f} ms  "
            f"status={err.status} code={err.code} timeout={err.timeout} :: {err.message[:160]}"
        )
        return False
    ms = (time.perf_counter() - start) * 1000
    print(f"OK    {label:<24} {model}  {ms:7.0f} ms  {detail}")
    return True


async def main() -> None:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        cloudflare = CloudflareProvider(client, settings)
        groq = GroqProvider(client, settings)

        cf_llm = "@cf/openai/gpt-oss-20b"
        cf_img = "@cf/black-forest-labs/flux-1-schnell"
        cf_flux2 = "@cf/black-forest-labs/flux-2-klein-4b"
        groq_llm = "openai/gpt-oss-20b"

        async def cf_chat() -> str:
            r = await cloudflare.chat(cf_llm, LLMRequest(messages=PROMPT, max_tokens=64))
            return f"reply={r.text.strip()[:40]!r} usage={r.usage}"

        async def cf_image() -> str:
            r = await cloudflare.generate_image(
                cf_img, ImageRequest(prompt="a red apple on a white table", steps=4)
            )
            return f"image_bytes={len(r.images[0])}"

        async def cf_flux2_edit() -> str:
            # multipart form data: prompt + width/height + ONE input image (input_image_0)
            r = await cloudflare.generate_image(
                cf_flux2,
                ImageRequest(
                    prompt="make the background light blue",
                    width=512,
                    height=512,
                    input_images=[TEST_IMAGE.read_bytes()],
                ),
            )
            return f"status=200 response_fields=result.image images={len(r.images)} bytes={len(r.images[0])}"

        async def groq_chat() -> str:
            r = await groq.chat(
                groq_llm,
                LLMRequest(messages=PROMPT, max_tokens=64, reasoning_effort="low"),
            )
            return f"reply={r.text.strip()[:40]!r} usage={r.usage}"

        results = [
            await _check("cloudflare LLM", cloudflare, cf_llm, cf_chat),
            await _check("cloudflare image", cloudflare, cf_img, cf_image),
            await _check("cloudflare FLUX.2 edit", cloudflare, cf_flux2, cf_flux2_edit),
            await _check("groq LLM", groq, groq_llm, groq_chat),
        ]
    print(f"\n{sum(results)}/{len(results)} succeeded")


if __name__ == "__main__":
    asyncio.run(main())
