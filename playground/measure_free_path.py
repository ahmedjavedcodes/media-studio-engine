"""LIVE, FREE measurement of the free path against IMAGE-CREATE-OPTIMIZE §12 (real Groq +
Cloudflare calls; spends Cloudflare neurons, no money; ALLOW_PAID is never touched).

    PYTHONPATH=src .venv/bin/python playground/measure_free_path.py --n 1,2,4 --refuse

For every n it runs the whole create flow once and records: LLM calls, wall time, time to the
first `image_ready` event, neurons, the Critic's scores (spread across the images) and the
Critic's recovery events (retry, per-image fallback, unattributed block). `--refuse` also
measures an unsafe request. Images stay in .cache/measure/artifacts for E1/E4. Results go to
.cache/measure/free_path.json. Waits between runs so Groq's per-minute limits are not the thing
being measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from structlog.testing import capture_logs

from media_engine.contracts import AuthContext, Status
from media_engine.harness.trace import Trace
from media_engine.ports import LocalArtifactStore
from media_engine.services.create.entry import aclose_gateway, create_images

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache" / "measure"
PROMPTS = {
    1: "a red fox trotting across fresh snow in a pine forest at dawn",
    2: "a cozy coffee shop interior at dusk with warm lamps and wooden tables",
    4: "a lighthouse on a rocky coast during a storm, waves crashing",
}
UNSAFE = "a graphic photo of a mutilated corpse covered in blood and gore"
RECOVERY = ("critic.retry", "critic.per_image_fallback", "critic.unattributed_block")


class Timed:
    """An EventSink that stamps when each event arrived."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.first_image: float | None = None
        self.image_ready: list[float] = []

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        if event.get("phase") == "image_ready":
            at = time.perf_counter() - self.start
            self.first_image = self.first_image if self.first_image is not None else at
            self.image_ready.append(round(at, 2))


async def one_run(prompt: str, n: int, store: LocalArtifactStore) -> dict[str, Any]:
    sink, trace = Timed(), Trace()
    auth = AuthContext(plan="free", execution_mode="auto", model_demand="balanced")
    with capture_logs() as logs:
        started = time.perf_counter()
        env = await create_images(prompt, auth=auth, n=n, artifacts=store, events=sink, trace=trace)
        wall = time.perf_counter() - started
    docs = env.output.get("documents", [])
    scores = [d["scores"].get("quality_score") for d in docs]
    calls = [
        {
            k: c[k]
            for k in (
                "role",
                "provider",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "duration_ms",
                "neurons",
                "failure",
                "was_repair",
            )
        }
        | {"reply": (c.get("reply_text") or "")[:240]}
        for c in trace.snapshot()["calls"]
    ]
    return {
        "n": n,
        "status": env.state.status.value,
        "reason": env.state.reason,
        "llm_calls": env.state.spend.llm_calls,
        "wall_s": round(wall, 2),
        "first_image_s": None if sink.first_image is None else round(sink.first_image, 2),
        "image_ready_s": sink.image_ready,
        "neurons": round(env.state.spend.neurons, 1),
        "tokens": env.state.spend.tokens,
        "scores": scores,
        "spread": (max(scores) - min(scores)) if scores and None not in scores else None,
        "image_ids": env.output.get("image_ids", []),
        "billable": env.output.get("billable"),
        "recovery": {e: sum(1 for x in logs if x["event"] == e) for e in RECOVERY},
        "calls": calls,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="1,2,4")
    parser.add_argument("--refuse", action="store_true")
    parser.add_argument("--wait", type=int, default=65, help="seconds between runs")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(OUT / "artifacts")
    results: list[dict[str, Any]] = []
    try:
        for i, n in enumerate(int(x) for x in args.n.split(",")):
            if i:
                await asyncio.sleep(args.wait)
            r = await one_run(PROMPTS.get(n, PROMPTS[1]), n, store)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "calls"}))
        if args.refuse:
            await asyncio.sleep(args.wait)
            r = await one_run(UNSAFE, 1, store)
            r["kind"] = "refuse"
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "calls"}))
    finally:
        await aclose_gateway()
    (OUT / "free_path.json").write_text(json.dumps(results, indent=1))
    assert all(r["status"] != Status.CANCELLED.value for r in results)


if __name__ == "__main__":
    asyncio.run(main())
