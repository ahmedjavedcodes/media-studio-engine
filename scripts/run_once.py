"""Smoke test for the image-creation harness: one prompt in, an image file out.

    PYTHONPATH=src .venv/bin/python scripts/run_once.py "a cozy coffee shop"
    PYTHONPATH=src .venv/bin/python scripts/run_once.py "a red fox" --plan pro --n 2

Makes REAL provider calls (Cloudflare Workers AI, free tier) with the keys in `.env`; it makes
none, and says so, when a key is missing. Paid providers stay off unless ALLOW_PAID=true is set
in the environment, which this script never sets. Keys are never printed.
Images are written to ./.artifacts/.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_engine.contracts import AuthContext
from media_engine.core.config import get_settings
from media_engine.ports import LocalArtifactStore
from media_engine.services.create.entry import aclose_gateway, create_images

REQUIRED_KEYS = {
    "CLOUDFLARE_ACCOUNT_ID": "cloudflare_account_id",
    "CLOUDFLARE_API_TOKEN": "cloudflare_api_token",
}


def missing_keys() -> list[str]:
    settings = get_settings()
    return [env for env, attr in REQUIRED_KEYS.items() if not getattr(settings, attr)]


async def main(args: argparse.Namespace) -> int:
    missing = missing_keys()
    if missing:
        print(f"missing key in .env: {', '.join(missing)} (see .env.example). No calls were made.")
        return 2

    artifacts = LocalArtifactStore("./.artifacts")
    auth = AuthContext(plan=args.plan, execution_mode="auto", model_demand="balanced")
    try:
        env = await create_images(args.prompt, auth=auth, n=args.n, artifacts=artifacts)
    finally:
        await aclose_gateway()

    print(f"status:  {env.state.status.value}")
    if env.state.reason:
        print(f"reason:  {env.state.reason}")
    for image_id in env.output.get("image_ids", []):
        path = unquote(urlparse(await artifacts.url(image_id)).path)
        print(f"image:   {image_id}  ->  {path}")
    for issue in env.output.get("issues", []):
        print(f"issue:   {issue}")
    spend = env.state.spend
    print(
        f"spend:   {spend.llm_calls} llm calls, {spend.neurons:.1f} neurons, ${spend.cost_usd:.4f}"
    )
    return 0 if env.output.get("image_ids") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt")
    parser.add_argument("--plan", default="free", choices=["free", "pro", "max", "ultra"])
    parser.add_argument("--n", type=int, default=1)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
