"""Generate test-manual/.../person_adult_synthetic.jpg with Cloudflare FLUX (free tier).

    uv run python playground/make_person_fixture.py

A real model call, so it lives in playground/. The person is synthetic, an adult, fully clothed.
"""

import base64
from pathlib import Path

import httpx

from app.core.config import get_settings

PROMPT = (
    "photorealistic photo of an adult woman about 30 years old standing in a sunny park, "
    "wearing blue jeans and a plain white t-shirt, full body, neutral relaxed pose, "
    "looking at the camera, natural daylight, sharp focus"
)
MODEL = "@cf/black-forest-labs/flux-1-schnell"
OUT = Path("test-manual/guardrails/evals/fixtures/guardrail/person_adult_synthetic.jpg")


def main() -> None:
    s = get_settings()
    url = f"https://api.cloudflare.com/client/v4/accounts/{s.cloudflare_account_id}/ai/run/{MODEL}"
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {s.cloudflare_api_token}"},
        json={"prompt": PROMPT, "steps": 6},
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    OUT.write_bytes(base64.b64decode(body["result"]["image"]))
    OUT.with_suffix(".PROMPT.txt").write_text(f"model: {MODEL}\nprompt: {PROMPT}\n", "utf-8")
    print("saved", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
