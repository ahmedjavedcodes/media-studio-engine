"""PAID (~$0.028): does OpenRouter's FLUX.2 Klein 4B reproduce an image from the same seed?

    PYTHONPATH=src .venv/bin/python playground/klein_seed_check.py --confirm-paid

Two calls, same prompt and seed; the decoded PIXELS are compared (not file bytes). It is the only
candidate for reproducible evals: no free model honours a seed (IMAGE-CREATE-OPTIMIZE §6).
Same spend safety as record_openrouter_response.py: refuses without --confirm-paid, checks the
ledger (.cache/paid_spend.jsonl) against the $5 cap first, and writes every call to it.
Prints the verdict, the cost of each call and the running total; never keys or image data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from PIL import Image

from media_engine.core.config import get_settings
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.types import ImageRequest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".cache" / "paid_spend.jsonl"
MODEL = "black-forest-labs/flux.2-klein-4b"
SPEND_CAP_USD = 5.00
MAX_CALL_USD = 0.05  # assumed ceiling per call (listed ~$0.014)


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    return round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6)


def log_call(cost: float | None, status: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "status": status,
            "cost_usd": cost,
            "purpose": "klein seed reproducibility",
        }
        f.write(json.dumps(row) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid", action="store_true")
    if not parser.parse_args().confirm_paid:
        raise SystemExit("PAID call: re-run with --confirm-paid")
    spent = ledger_total()
    if spent + 2 * MAX_CALL_USD > SPEND_CAP_USD:
        raise SystemExit(f"refused: ${spent:.4f} spent, cap ${SPEND_CAP_USD:.2f}")
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")

    digests: list[str] = []
    async with httpx.AsyncClient() as client:
        provider = OpenRouterProvider(client, settings)
        request = ImageRequest(prompt="a red apple on a white table, studio photo", seed=42)
        for call in (1, 2):
            try:
                response = await provider.generate_image(MODEL, request)
            except Exception as err:  # noqa: BLE001 - record the attempt, then stop
                log_call(None, f"error {type(err).__name__}")
                raise SystemExit(f"call {call} failed: {type(err).__name__}: {str(err)[:200]}")
            log_call(response.cost_usd, "200")
            with Image.open(io.BytesIO(response.images[0])) as img:
                digests.append(hashlib.sha256(img.convert("RGB").tobytes()).hexdigest())
            print(f"call {call}: cost ${response.cost_usd}, pixels {digests[-1][:12]}")
    print(f"same seed -> {'IDENTICAL' if digests[0] == digests[1] else 'DIFFERENT'} pixels")
    print(f"ledger total: ${ledger_total():.4f} of ${SPEND_CAP_USD:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
