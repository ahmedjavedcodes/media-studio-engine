"""E7 (PAID, ~$0.05): one call with n=4, or four separate calls? openai/gpt-image-2, quality low.

    PYTHONPATH=src:playground .venv/bin/python playground/e7_n_per_call.py --confirm-paid     (NOT yet run)

Cost: 8 images x $0.006 = about $0.048 (worst case checked against the ledger first; every call
is written to .cache/paid_spend.jsonl). Compares, for the two ways of getting 4 images: billed
cost (usage.cost), wall time, and how different the 4 images are from each other (mean absolute
pixel difference between pairs: higher = more variation). If one call wins, the parallel-branch
design could simplify.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import itertools
import json
import time

import httpx
from eval_common import MEASURE, log_paid, require_paid_ok
from PIL import Image, ImageChops

from media_engine.core.config import get_settings
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.types import ImageRequest

MODEL = "openai/gpt-image-2"
PROMPT = "a red fox trotting across fresh snow in a pine forest at dawn"
WORST_CASE_USD = 0.06  # 8 images at $0.006 with headroom


def variation(images: list[bytes]) -> float:
    """Mean absolute pixel difference (0-255) over all pairs, at 128 px."""
    small = [Image.open(io.BytesIO(b)).convert("RGB").resize((128, 128)) for b in images]
    diffs = []
    for a, b in itertools.combinations(small, 2):
        d = ImageChops.difference(a, b)
        diffs.append(sum(sum(p) for p in d.getdata()) / (128 * 128 * 3))
    return round(sum(diffs) / max(1, len(diffs)), 2)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid", action="store_true")
    require_paid_ok(parser.parse_args().confirm_paid, WORST_CASE_USD)
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")
    out: dict[str, object] = {}
    async with httpx.AsyncClient() as client:
        provider = OpenRouterProvider(client, settings)
        started = time.perf_counter()
        one = await provider.generate_image(MODEL, ImageRequest(prompt=PROMPT, n=4, quality="low"))
        log_paid(MODEL, one.cost_usd, "E7 n=4 in one call")
        out["one_call_n4"] = {
            "images": len(one.images),
            "cost_usd": one.cost_usd,
            "wall_s": round(time.perf_counter() - started, 1),
            "variation": variation(one.images),
        }
        started = time.perf_counter()
        parts = await asyncio.gather(
            *[
                provider.generate_image(MODEL, ImageRequest(prompt=PROMPT, n=1, quality="low"))
                for _ in range(4)
            ]
        )
        for p in parts:
            log_paid(MODEL, p.cost_usd, "E7 four calls")
        out["four_calls"] = {
            "images": sum(len(p.images) for p in parts),
            "cost_usd": round(sum(p.cost_usd or 0 for p in parts), 6),
            "wall_s": round(time.perf_counter() - started, 1),
            "variation": variation([p.images[0] for p in parts]),
        }
    (MEASURE / "e7.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
