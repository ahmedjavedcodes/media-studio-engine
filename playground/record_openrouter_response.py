"""ONE real, PAID OpenRouter image call, recorded as a test fixture (review item 3).

    PYTHONPATH=src .venv/bin/python playground/record_openrouter_response.py --confirm-paid

Why: the response field names (image payload, media type, usage.cost) were assumed from the
docs. This pins them to what OpenRouter actually returns. Approved by the owner: one call, on the
cheapest model (openai/gpt-image-2 at quality low, measured $0.006), inside the spend cap.

Spend safety, all checked BEFORE the call:
- refuses without --confirm-paid;
- a persistent ledger (.cache/paid_spend.jsonl, one line per paid call) holds every real paid
  call; the call is refused if ledger total + MAX_CALL_USD would exceed SPEND_CAP_USD. The
  ceiling is a fixed constant, NOT gateway/budget.estimate_image_usd, which is not enforcing;
- the request body is built by the production adapter (OpenRouterProvider.build_image_payload)
  and validated against the catalogue cache, so quality is sent explicitly as `low`.

Writes tests/fixtures/openrouter/gpt-image-2.images.json: the request body, HTTP status, response
headers (cookies dropped) and the FULL response body. Never prints keys or image data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from media_engine.core.config import get_settings
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.types import ImageRequest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".cache" / "paid_spend.jsonl"
FIXTURE = ROOT / "tests" / "fixtures" / "openrouter" / "gpt-image-2.images.json"
MODEL = "openai/gpt-image-2"
PROMPT = "a red apple on a white table, studio photo"
SPEND_CAP_USD = 5.00  # the owner's cap across ALL real paid calls
MAX_CALL_USD = 0.05  # hard ceiling assumed for this one call (measured price is $0.006)
DROP_HEADERS = {"set-cookie", "cookie", "authorization"}


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid", action="store_true")
    args = parser.parse_args()
    if not args.confirm_paid:
        raise SystemExit("PAID call: re-run with --confirm-paid")

    spent = ledger_total()
    if spent + MAX_CALL_USD > SPEND_CAP_USD:
        raise SystemExit(f"refused: ${spent:.4f} spent, cap ${SPEND_CAP_USD:.2f}")
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")

    async with httpx.AsyncClient() as client:
        provider = OpenRouterProvider(client, settings)
        request = ImageRequest(prompt=PROMPT, width=1024, height=1024)
        payload = provider.build_image_payload(MODEL, request)
        provider._validate(MODEL, payload)  # same local check as production, before paying
        print(f"request body: {json.dumps(payload)}")
        print(f"ledger before: ${spent:.4f} of ${SPEND_CAP_USD:.2f}")
        key = settings.openrouter_api_key.get_secret_value()
        response = await client.post(
            f"{settings.openrouter_base_url.rstrip('/')}/images",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=180,
        )

    try:
        body = response.json()
    except ValueError:
        body = {"_raw_text": response.text}
    usage = body.get("usage") if isinstance(body, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    now = datetime.now(UTC).isoformat()

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:  # recorded even on an error status
        f.write(
            json.dumps(
                {
                    "ts": now,
                    "model": MODEL,
                    "status": response.status_code,
                    "cost_usd": cost,
                    "purpose": "review item 3 fixture",
                }
            )
            + "\n"
        )

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    headers = {k: v for k, v in response.headers.items() if k.lower() not in DROP_HEADERS}
    FIXTURE.write_text(
        json.dumps(
            {
                "recorded_at": now,
                "endpoint": "POST /api/v1/images",
                "request": payload,
                "status": response.status_code,
                "headers": headers,
                "body": body,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    def shape(value: object) -> object:  # field names and types, never image data
        if isinstance(value, dict):
            return {k: shape(v) for k, v in value.items()}
        if isinstance(value, list):
            return [shape(value[0])] if value else []
        if isinstance(value, str) and len(value) > 80:
            return f"<str, {len(value)} chars>"
        return value

    print(f"status: {response.status_code}")
    print(f"body shape: {json.dumps(shape(body), indent=1)}")
    print(f"usage.cost: {cost}")
    print(f"ledger after: ${ledger_total():.4f} of ${SPEND_CAP_USD:.2f}")
    print(f"fixture: {FIXTURE.relative_to(ROOT)} ({FIXTURE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
