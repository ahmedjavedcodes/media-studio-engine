"""Refresh the local OpenRouter model cache and check chains.yaml model IDs against it.

    uv run python playground/sync_openrouter_models.py

Makes only free listing calls (GET /images/models, per-model /endpoints for models used in
chains.yaml, and GET /models for chat models). No generation, no cost. Writes
.cache/openrouter_image_models.json. Prints model IDs and parameter NAMES only, never keys.
"""

from __future__ import annotations

import asyncio

import httpx

from media_engine.core.config import get_settings
from media_engine.gateway.openrouter_models import BASE_URL, ModelCache, sync_image_models
from media_engine.gateway.router import load_chains, task_kind


async def main() -> None:
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")
    key = settings.openrouter_api_key.get_secret_value()

    chains = load_chains()
    wanted: dict[str, list[str]] = {"image": [], "llm": []}
    for task, refs in chains.items():
        for ref in refs:
            if ref.provider == "openrouter" and not ref.is_placeholder:
                wanted[task_kind(task)].append(ref.model)

    cache = ModelCache()
    async with httpx.AsyncClient() as client:
        models = await sync_image_models(client, key, cache, price_for=wanted["image"])
        chat = await client.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {key}"})
        chat.raise_for_status()
        chat_ids = {m["id"] for m in chat.json().get("data", [])}

    image_ids = {m.id for m in models}
    print(f"image models in list: {len(image_ids)}   chat models in list: {len(chat_ids)}")
    print(f"cache written: {cache.path}\n")

    print("== chains.yaml OpenRouter IDs ==")
    for model_id in sorted(set(wanted["image"])):
        info = next((m for m in models if m.id == model_id), None)
        if info is None:
            print(f"MISSING  image  {model_id}  (not in /images/models)")
            continue
        params = ", ".join(sorted(info.supported_parameters))
        prices = [f"{p.get('billable')}/{p.get('unit')}=${p.get('cost_usd')}" for p in info.pricing]
        print(f"OK       image  {model_id}")
        print(f"           input={info.input_modalities} output={info.output_modalities}")
        print(f"           params: {params}")
        print(f"           pricing: {prices or 'not available'}")
        for name in ("aspect_ratio", "resolution", "n", "input_references"):
            d = info.supported_parameters.get(name)
            if d:
                print(f"           {name}: {d}")
    for model_id in sorted(set(wanted["llm"])):
        status = "OK      " if model_id in chat_ids else "MISSING "
        print(
            f"{status} chat   {model_id}" + ("" if model_id in chat_ids else "  (not in /models)")
        )


if __name__ == "__main__":
    asyncio.run(main())
