"""Run the txt2img sub-agent for real from the terminal (build mode: Cloudflare only).

    uv run python playground/create.py "iced coffee ad for summer" --plan pro --n 2
    uv run python playground/create.py "a red fox" --model flux-2-klein-4b --ratio 4:5

It skips the HTTP guardrail (so it needs no TypeSafe key) and calls Cloudflare for the enhancer,
the image models and the critic. Each run uses a few of the 10,000 free daily neurons.
If Cloudflare says the limit is used up, it stops. It never switches to a paid provider.
Answers to a question are asked on the terminal.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.grand.graph import build_grand_graph
from app.agents.image_grand.generation.create.factory import build_create_deps
from app.api.run_service import RunService
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.config import get_settings
from app.core.plans import Plan


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--plan", default="basic", choices=[p.value for p in Plan])
    ap.add_argument("--model", default="auto")
    ap.add_argument("--ratio", default="1:1")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    s = get_settings()
    if not (s.cloudflare_account_id and s.cloudflare_api_token):
        print("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not set in .env")
        return 2
    if s.openrouter_images:
        print("OPENROUTER_IMAGES is on: paid calls are possible. Cap:", s.spend_cap_usd, "USD")

    deps, http = build_create_deps(s)
    await deps.prompts.load()
    service = RunService(build_grand_graph(deps, InMemorySaver()), deps)
    req = GenerationRequest(
        prompt=args.prompt,
        plan=Plan(args.plan),
        settings=ImageSettings(model=args.model, aspect_ratio=args.ratio, n=args.n),
    )
    run_id = "play-" + str(abs(hash(args.prompt)))[:8]
    service.start(run_id, "playground", req)
    code = 0
    while True:
        last = None
        async for ev in service.stream(run_id):
            last = ev
            detail = ev.label or ev.message or ""
            if ev.progress:
                detail += f" ({ev.progress})"
            if ev.type == "image" and ev.image:
                detail = ev.image.url
            print(f"[{ev.type}] {detail}")
        if last is not None and last.type == "question":
            answers: dict[str, str] = {}
            for q in last.questions:
                print(q.text)
                for o in q.options:
                    print(f"  {o.id}) {o.label}{' *' if o.recommended else ''}")
                answers[q.id] = input("> ").strip() or "a"
            labels = {
                q["id"]: {o["id"]: o["label"] for o in q["options"]}
                for q in service.runs[run_id].questions
            }
            await service.resume(run_id, {k: labels[k].get(v, v) for k, v in answers.items()})
            continue
        if last is not None and last.type == "ready":
            print("\nRecommendation:", last.recommendation)
            for img in last.images:
                print("  ", Path(s.local_output_dir) / run_id / img.url.rsplit("/", 1)[-1])
        else:
            code = 1
        break
    print("Credits left:", await deps.credits.balance("playground"))
    deps.tracer.flush()  # the trace is sent in the background. Send it before we exit
    await http.aclose()
    return code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
