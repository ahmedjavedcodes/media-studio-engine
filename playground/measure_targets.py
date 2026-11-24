"""Measure the spec section 9 targets on real Cloudflare (build mode, free neurons).

    uv run python playground/measure_targets.py --model flux-2-klein-4b --n 1 --runs 5
    uv run python playground/measure_targets.py --model flux-2-klein-4b --n 4 --runs 5

For each run it records the time to the FIRST progress event and the time to `ready`.
It stops at once if a run fails (for example when the free daily limit is used up), and says so.
It never turns on a paid provider.
"""

import argparse
import asyncio
import statistics
import time

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.grand.graph import build_grand_graph
from app.agents.image_grand.generation.create.factory import build_create_deps
from app.api.run_service import RunService
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.config import get_settings
from app.core.plans import Plan

PROMPTS = [
    "a red fox in the snow",
    "a ceramic mug of coffee on a wooden table",
    "a lighthouse on a cliff at sunset",
    "a bowl of fresh strawberries",
    "a bicycle leaning on a brick wall",
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux-2-klein-4b")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--plan", default="max")
    args = ap.parse_args()

    s = get_settings()
    if s.openrouter_images or s.fal_images:
        print("a paid provider flag is on. Refusing to measure.")
        return 2
    deps, http = build_create_deps(s)
    await deps.prompts.load()
    service = RunService(build_grand_graph(deps, InMemorySaver()), deps)
    first: list[float] = []
    ready: list[float] = []
    for i in range(args.runs):
        run_id = f"measure-{args.model}-{args.n}-{i}-{int(time.time())}"
        req = GenerationRequest(
            prompt=PROMPTS[i % len(PROMPTS)],
            plan=Plan(args.plan),
            settings=ImageSettings(model=args.model, n=args.n),
        )
        t0 = time.monotonic()
        service.start(run_id, "measure", req)
        t_first: float | None = None
        last = None
        async for ev in service.stream(run_id):
            if t_first is None:
                t_first = time.monotonic() - t0
            last = ev
        t_ready = time.monotonic() - t0
        await service.wait(run_id)
        if last is None or last.type != "ready":
            print(
                f"run {i + 1}: FAILED ({last.type if last else 'no events'}: "
                f"{last.message if last else ''}). Stopping. Is the free limit used up?"
            )
            deps.tracer.flush()
            await http.aclose()
            return 1
        first.append(t_first or 0.0)
        ready.append(t_ready)
        print(
            f"run {i + 1}: first progress {t_first:.2f}s, ready {t_ready:.1f}s "
            f"({len(last.images)} images)"
        )
    print(f"\nmodel={args.model} n={args.n} runs={args.runs}")
    print(f"first progress: median {statistics.median(first):.2f}s, max {max(first):.2f}s")
    print(
        f"ready:          median {statistics.median(ready):.1f}s, max {max(ready):.1f}s, "
        f"min {min(ready):.1f}s"
    )
    deps.tracer.flush()
    await http.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
