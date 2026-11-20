"""E4 (FREE: Groq vision, no images generated): is the ranking Critic's order stable when the
SAME images arrive in a different order? One image set, 3 shuffled orders, one Critic call each.

    PYTHONPATH=src:playground .venv/bin/python playground/e4_position_bias.py [--images 3] [--wait 65]

Reads the images already in .cache/measure/artifacts (run measure_free_path.py first). Reports,
per image, its rank and mean score under each order, the pairwise rank correlation between
orders, whether the same image won every time, and whether the position of an image predicts
its rank (a bias) rather than the image itself.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random

from eval_common import MEASURE, CriticRig, spearman, stored_image_ids

from media_engine.ports import LocalArtifactStore

PROMPT = "a lighthouse on a rocky coast during a storm, waves crashing"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--wait", type=int, default=65)  # Groq qwen: ~7K input tokens a minute
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()
    ids = stored_image_ids()[-args.images :]
    if len(ids) < 2:
        raise SystemExit("need at least 2 stored images: run measure_free_path.py first")
    orders = [ids, ids[::-1], random.Random(7).sample(ids, len(ids))]
    rig = CriticRig(LocalArtifactStore(MEASURE / "artifacts"))
    runs: list[dict[str, dict[str, float]]] = []
    try:
        for i, order in enumerate(orders):
            if i:
                await asyncio.sleep(args.wait)
            reviews, usage = await rig.rank(order, args.prompt)
            if usage["status"] != "done":
                print("order", i, "failed:", usage)
                continue
            runs.append(
                {
                    r["image_id"]: {
                        "rank": r.get("rank"),
                        "score": r.get("quality_score"),
                        "position": order.index(r["image_id"]) + 1,
                    }
                    for r in reviews
                }
            )
            print(f"order {i}: {usage['providers']} prompt_tokens={usage['prompt_tokens']}")
    finally:
        await rig.close()

    report: dict[str, object] = {"orders": len(runs), "images": len(ids), "runs": runs}
    corr = []
    for a, b in itertools.combinations(runs, 2):
        common = [k for k in a if k in b and a[k]["rank"] and b[k]["rank"]]
        corr.append(spearman([a[k]["rank"] for k in common], [b[k]["rank"] for k in common]))
    winners = {min(r, key=lambda k: r[k]["rank"] or 99) for r in runs if r}
    report["pairwise_rank_correlation"] = corr
    report["same_winner_every_time"] = len(winners) == 1
    # a position bias shows as rank following position instead of the image
    pos = [(v["position"], v["rank"]) for r in runs for v in r.values() if v["rank"]]
    report["position_vs_rank_correlation"] = spearman(
        [float(p) for p, _ in pos], [float(k) for _, k in pos]
    )
    per_image = {}
    for k in ids:
        ranks = [r[k]["rank"] for r in runs if k in r and r[k]["rank"]]
        per_image[k[-6:]] = {"ranks": ranks, "spread": (max(ranks) - min(ranks)) if ranks else None}
    report["per_image"] = per_image
    (MEASURE / "e4.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
