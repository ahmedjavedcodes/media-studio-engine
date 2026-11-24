"""Does a model honour the seed? Two calls, same prompt and seed, then compare the pixels.

    uv run python playground/seed_check.py --model phoenix-1
    uv run python playground/seed_check.py --model phoenix-1 --size 512   # cheaper

Real Cloudflare calls (build mode, free neurons). It stops on any error and never uses a paid
provider. A seed is "honoured" when both images are the same (a near-zero pixel difference).
"""

import argparse
import asyncio
import hashlib
import io

import httpx
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.gateway.image_gateway import ImageGateway
from app.gateway.providers.cloudflare import CloudflareClient
from app.gateway.providers.cloudflare_create import CloudflareImages
from app.gateway.registry import load_registry

PROMPT = "a red fox sitting in the snow, studio lighting, sharp focus"
SEED = 123456789


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phoenix-1")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    s = get_settings()
    if s.openrouter_images or s.fal_images:
        print("a paid provider flag is on. Refusing.")
        return 2
    card = load_registry().get(args.model)
    if card is None or not card.supports_seed:
        print(f"{args.model} is not a seed-capable model in the registry")
        return 2
    async with httpx.AsyncClient() as http:
        cf = CloudflareClient(
            http, account_id=s.cloudflare_account_id, api_token=s.cloudflare_api_token
        )
        gw = ImageGateway({"cloudflare": CloudflareImages(cf, http)}, ["cloudflare"])
        shots = []
        for i in (1, 2):
            res = await gw.generate(
                card,
                prompt=PROMPT,
                negative_prompt="",
                width=args.size,
                height=args.size,
                aspect_ratio="1:1",
                seed=SEED,
            )
            shots.append(res.image.data)
            digest = hashlib.sha256(res.image.data).hexdigest()[:16]
            print(
                f"call {i}: {len(res.image.data)} bytes, sha256 {digest}, seed_sent={res.seed_sent}"
            )
    a, b = (np.asarray(Image.open(io.BytesIO(x)).convert("RGB"), dtype=float) for x in shots)
    diff = float(np.abs(a - b).mean())
    same_bytes = shots[0] == shots[1]
    print(
        f"identical bytes: {same_bytes}; mean pixel difference: {diff:.3f} (0 = same, 0-255 scale)"
    )
    print("RESULT seed_honoured =", same_bytes or diff < 1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
