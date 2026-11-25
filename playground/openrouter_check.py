"""Verify OpenRouter image models through OUR adapter, with a hard spend cap.

    uv run python playground/openrouter_check.py --spend --cap 0.30 gpt-image-2 recraft-v4-1

PAID. It refuses to run without --spend. Each call is reserved against the cap first, and the real
`usage.cost` is added up and printed. It makes one call per model id you name, and saves the
image next to the others in --out. It does not change models.yaml: a person sets `status: live`
for the models that passed.
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.gateway.budget import SpendGuard
from app.gateway.image_gateway import ImageGateway
from app.gateway.options import RequestExtras, options_for
from app.gateway.providers.openrouter_images import OpenRouterImages
from app.gateway.registry import load_registry
from app.tools.imaging import make_thumbnail, sniff
from app.tools.sizes import dimensions

PROMPT = "a red fox sitting in the snow, studio light, sharp focus"
LOGO_PROMPT = "a simple two-color logo of a coffee cup"
POSTER_PROMPT = "a flat minimal summer sale poster of a coffee cup"

# model id -> (prompt, extras). Anything not listed uses the default prompt and no extras.
SCENARIOS = {
    "recraft-v4-1": (POSTER_PROMPT, RequestExtras(brand_colors=("#DC143C", "#FFFFFF"))),
    "recraft-v4-1-vector": (LOGO_PROMPT, RequestExtras()),
    "gpt-image-2-6-flare": (PROMPT, RequestExtras(complexity=2)),
}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--spend", action="store_true", help="yes, make PAID calls")
    ap.add_argument("--cap", type=float, default=0.30, help="stop after this many USD")
    ap.add_argument("--out", default="outputs/openrouter-check")
    args = ap.parse_args()
    if not args.spend:
        print("This makes PAID OpenRouter calls. Add --spend to confirm.")
        return 2
    s = get_settings()
    if not s.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set")
        return 2
    cap = min(args.cap, s.spend_cap_usd)
    registry = load_registry()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(cap, ledger=Path(".cache/paid_spend.jsonl"))
    failures = 0
    async with httpx.AsyncClient() as http:
        provider = OpenRouterImages(
            http, api_key=s.openrouter_api_key, base_url=s.openrouter_base_url, guard=guard
        )
        gateway = ImageGateway({"openrouter": provider}, ["openrouter"], paid={"openrouter"})
        for model_id in args.models:
            card = registry.get(model_id)
            if card is None:
                print(f"{model_id}: not in the registry")
                failures += 1
                continue
            prompt, extras = SCENARIOS.get(model_id, (PROMPT, RequestExtras()))
            w, h = dimensions("1:1", "1K", card.size_multiple)
            started = time.monotonic()
            try:
                result = await gateway.generate(
                    card,
                    prompt=prompt,
                    negative_prompt="",
                    width=w,
                    height=h,
                    aspect_ratio="1:1",
                    seed=42,
                    timeout=150,
                    options=options_for(card, extras),
                )
            except Exception as exc:  # noqa: BLE001 - report and carry on with the next model
                print(f"{model_id}: FAILED {type(exc).__name__}: {exc}")
                failures += 1
                continue
            took = time.monotonic() - started
            info = sniff(result.image.data, result.image.media_type)
            thumb = make_thumbnail(result.image.data)
            path = out_dir / f"{model_id}.{info.ext}"
            path.write_bytes(result.image.data)
            est = card.cost_per_image
            print(
                f"{model_id}: OK via {card.endpoint} | {info.mime} {info.width}x{info.height} | "
                f"{len(result.image.data) // 1024} KB | thumb {len(thumb) // 1024} KB | "
                f"cost ${result.image.cost_usd} (registry estimate ${est}) | {took:.1f}s | "
                f"seed_sent={result.seed_sent} | {path}"
            )
    print(f"\nspent this run: ${guard.spent_usd:.4f} of cap ${cap}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
