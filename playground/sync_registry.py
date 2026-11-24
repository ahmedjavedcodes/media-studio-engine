"""Print what a provider's catalogue says about image models. A free metadata read.

    uv run python playground/sync_registry.py                      # both
    uv run python playground/sync_registry.py --provider fal
    uv run python playground/sync_registry.py --provider openrouter

It edits nothing and generates nothing. A person updates src/app/gateway/models.yaml from the
report. fal's list is public. Its prices need FAL_KEY (the key is only sent to fal's own API).
"""

import argparse
import asyncio

import httpx

from app.core.config import get_settings
from app.gateway.registry import load_registry
from app.gateway.sync import SyncReport, sync_fal, sync_openrouter

SHOW = 40


def show(title: str, report: SyncReport) -> None:
    print(f"\n=== {title} ===")
    priced = sorted(
        (m for m in report.new_models if m["cost"] is not None), key=lambda m: m["cost"]
    )
    unpriced = [m for m in report.new_models if m["cost"] is None]
    print(f"new image models: {len(report.new_models)} ({len(priced)} with a price)")
    for m in priced[:SHOW]:
        print(f"  ${m['cost']:<8.4f} {m['slug']:<48} {m['name']}")
    if len(priced) > SHOW:
        print(f"  ... {len(priced) - SHOW} more priced models")
    for m in unpriced[:SHOW]:
        print(f"  (no price) {m['slug']:<45} {m['name']}")
    print("price changes:", report.price_changes)
    print("missing from the provider:", report.missing)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openrouter", "fal", "all"], default="all")
    args = ap.parse_args()
    s = get_settings()
    registry = load_registry()
    async with httpx.AsyncClient() as client:
        if args.provider in ("openrouter", "all"):
            report = await sync_openrouter(
                client,
                base_url=s.openrouter_base_url,
                registry=registry,
                api_key=s.openrouter_api_key,
            )
            show("OpenRouter", report)
        if args.provider in ("fal", "all"):
            show("fal", await sync_fal(client, registry=registry, api_key=s.fal_key))


if __name__ == "__main__":
    asyncio.run(main())
