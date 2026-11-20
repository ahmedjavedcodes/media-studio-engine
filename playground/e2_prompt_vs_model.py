"""E2 (PAID, ~$0.16): separate prompt quality from model quality (IMAGE-CREATE-OPTIMIZE §11).

    PYTHONPATH=src:playground .venv/bin/python playground/e2_prompt_vs_model.py --confirm-paid   (NOT yet run)

Four generations, same prompt and size:
                    baseline prompt (the user's words)   compiled prompt (Brief + Compiler)
    FLUX.2 Klein 4B   $0.014                              $0.014
    Nano Banana 2     $0.067                              $0.067          total about $0.162

The Brief comes from the real enhancer (a free Groq call); the compiled prompt uses each model's
own style and word budget from the registry. The four images are then ranked by the real Critic
(free Groq, one call) so the result is a score table, plus the images themselves in
.cache/measure/e2/ for the human eye. Worst case is checked against the ledger and every paid
call is written to it. Neither model reproduces from a seed (measured), so each cell is ONE
sample: read the table as a hint, not a verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
from eval_common import AUTH, MEASURE, CriticRig, log_paid, require_paid_ok

from media_engine.agents.base import AgentConfig
from media_engine.agents.prompt.enhancer import PromptEnhancer
from media_engine.contracts import Goal
from media_engine.core.config import get_settings
from media_engine.gateway.budget import estimate_image_usd
from media_engine.gateway.compiler import compile_prompt
from media_engine.gateway.image_registry import image_registry
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.providers.openrouter import OpenRouterProvider
from media_engine.gateway.router import Gateway
from media_engine.gateway.types import ImageRequest
from media_engine.ports import ListEventSink, LocalArtifactStore
from media_engine.schemas import EnhancerOutput
from media_engine.services.create.loader import load_create_config

MODELS = {
    "klein": "black-forest-labs/flux.2-klein-4b",
    "nano_banana_2": "google/gemini-3.1-flash-image",
}
PROMPT = "a red fox trotting across fresh snow in a pine forest at dawn"
HEADROOM = 1.25


def planned_cost() -> float:
    reg = image_registry()
    one = sum(
        estimate_image_usd(reg.get(slug), ImageRequest(prompt="x")) or 0 for slug in MODELS.values()
    )
    return round(2 * one, 4)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid", action="store_true")
    confirm = parser.parse_args().confirm_paid
    cost = planned_cost()
    print(f"planned cost: ${cost:.4f} (4 generations)")
    require_paid_ok(confirm, cost * HEADROOM)
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is not set in .env")

    cfg = load_create_config()
    gateway = Gateway()
    enhancer = PromptEnhancer(
        AgentConfig.from_policy("prompt_enhancer", cfg.agent("prompt_enhancer"), EnhancerOutput),
        LLMGateway(gateway, chain_resolver(cfg.agents)),
        ListEventSink(),
    )
    stop = cfg.policy.stopping_condition("enhance", "free", "a brief")
    env = await enhancer.run(
        Goal(task="enhance", instructions="brief"), stop, AUTH, {"prompt": PROMPT}
    )
    brief = env.output["brief"]
    print("brief:", json.dumps(brief))

    out_dir = MEASURE / "e2"
    store = LocalArtifactStore(out_dir / "artifacts")
    rows: list[dict[str, object]] = []
    ids: list[str] = []
    async with httpx.AsyncClient() as client:
        provider = OpenRouterProvider(client, settings)
        for name, slug in MODELS.items():
            spec = image_registry().get(slug)
            assert spec is not None
            for arm in ("baseline", "compiled"):
                text = (
                    PROMPT
                    if arm == "baseline"
                    else compile_prompt(
                        brief, spec.prompt_style, spec.prompt_max_words, fallback=PROMPT
                    ).text
                )
                resp = await provider.generate_image(
                    slug, ImageRequest(prompt=text, width=1024, height=1024)
                )
                log_paid(slug, resp.cost_usd, f"E2 {name} {arm}")
                image_id = await store.put(resp.images[0], "image/png")
                ids.append(image_id)
                rows.append(
                    {
                        "model": name,
                        "arm": arm,
                        "prompt": text,
                        "cost_usd": resp.cost_usd,
                        "image_id": image_id,
                    }
                )
                print(f"{name:14} {arm:9} ${resp.cost_usd} {len(text.split())} words")
    rig = CriticRig(store)
    try:
        reviews, usage = await rig.rank(ids, PROMPT)
    finally:
        await rig.close()
        await gateway.aclose()
    by_id = {r["image_id"]: r for r in reviews}
    for row in rows:
        r = by_id.get(str(row["image_id"]), {})
        row.update(score=r.get("quality_score"), rank=r.get("rank"), scores=r.get("scores"))
    (out_dir / "e2.json").write_text(
        json.dumps({"brief": brief, "rows": rows, "critic": usage}, indent=1)
    )
    for row in rows:
        print(row["model"], row["arm"], "score", row.get("score"), "rank", row.get("rank"))


if __name__ == "__main__":
    asyncio.run(main())
