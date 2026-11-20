"""E5 (FREE: Cloudflare neurons for images + Groq vision): Critic calibration against hand scores.
Two steps (NOT yet run):

  1. make:    generate N images on the free path, have the Critic score them (batches of 4), and
              write .cache/measure/e5.csv with EMPTY hand-score columns, plus the images.
        PYTHONPATH=src:playground .venv/bin/python playground/e5_calibration.py make --images 50
  2. report:  after you fill the hand_* columns (0-10 each) in the CSV:
        PYTHONPATH=src:playground .venv/bin/python playground/e5_calibration.py report

`report` prints, per dimension, the mean absolute error and rank correlation between the Critic
and you, then suggests the pass threshold on the Critic's mean score that best separates your
"acceptable" (mean hand score >= 6) from "weak" images (max F1). Thresholds per tier are then
your decision: a higher tier can simply demand a higher cut.

Cost: $0 in money. `make --images 50` spends about 50 x 57.6 = 2,880 Cloudflare neurons (29% of the
10,000/day free pool, shared with every other free call) and about 13 Groq vision calls
(~80K of the 200K tokens/day), ~15 minutes because of Groq's 7K input tokens a minute.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json

from eval_common import MEASURE, CriticRig, spearman

from media_engine.gateway.router import Gateway
from media_engine.gateway.types import ImageRequest
from media_engine.ports import LocalArtifactStore

CSV = MEASURE / "e5.csv"
DIMS = ("prompt_adherence", "craft", "visual_interest", "technical")
# varied on purpose: easy subjects, and ones image models usually get wrong (hands, crowds, text)
PROMPTS = [
    "a red apple on a white table", "a close-up of a person's hands holding a coffee cup",
    "a crowded street market with many people", "a shop sign with the words OPEN 24 HOURS",
    "a cat sleeping on a sofa", "a bowl of ramen with chopsticks", "a sailing boat at sunset",
    "a woman playing a violin on stage", "a plate of sushi on a wooden board",
    "a futuristic city skyline at night", "a child blowing out birthday candles",
    "a glass of water with ice cubes", "a mountain lake with a small cabin",
    "a pair of running shoes on a track", "a chessboard in the middle of a game",
    "a bicycle leaning on a brick wall", "a bakery window full of pastries",
    "a astronaut floating beside a space station", "a wedding cake with three tiers",
    "a hiker standing on a ridge above clouds", "a vintage typewriter with paper in it",
    "a rainy window with city lights behind", "a basket of colourful fruit",
    "a guitar leaning on an amp in a garage", "a horse running through a field",
]  # fmt: skip


async def make(n: int, wait: int) -> None:
    store = LocalArtifactStore(MEASURE / "e5_artifacts")
    gateway = Gateway()
    rows: list[dict[str, object]] = []
    try:
        for i in range(n):
            prompt = PROMPTS[i % len(PROMPTS)]
            res = await gateway.image(
                "image_create", ImageRequest(prompt=prompt, seed=i), plan="free"
            )
            image_id = await store.put(res.output.images[0], "image/png")  # type: ignore[union-attr]
            rows.append({"image_id": image_id, "prompt": prompt})
        rig = CriticRig(store)
        try:
            for start in range(0, len(rows), 4):
                batch = rows[start : start + 4]
                if start:
                    await asyncio.sleep(wait)
                reviews, _ = await rig.rank([str(r["image_id"]) for r in batch], batch[0]["prompt"])  # type: ignore[arg-type]
                for r in reviews:
                    row = next(x for x in batch if x["image_id"] == r["image_id"])
                    row["critic_verdict"] = r["verdict"]
                    for d in DIMS:
                        row[f"critic_{d}"] = (r.get("scores") or {}).get(d)
                    row["critic_mean"] = r.get("quality_score")
        finally:
            await rig.close()
    finally:
        await gateway.aclose()
    cols = [
        "image_id",
        "prompt",
        "critic_verdict",
        *[f"critic_{d}" for d in DIMS],
        "critic_mean",
        *[f"hand_{d}" for d in DIMS],
    ]
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV}: fill the hand_* columns (0-10), then run `report`")


def report() -> None:
    rows = [
        r
        for r in csv.DictReader(CSV.open())
        if all(r.get(f"hand_{d}") for d in DIMS) and r.get("critic_mean")
    ]
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} fully hand-scored rows: score more")
    out: dict[str, object] = {"rows": len(rows), "dimensions": {}}
    for d in DIMS:
        c = [float(r[f"critic_{d}"]) for r in rows]
        h = [float(r[f"hand_{d}"]) for r in rows]
        out["dimensions"][d] = {  # type: ignore[index]
            "mean_abs_error": round(
                sum(abs(a - b) for a, b in zip(c, h, strict=True)) / len(rows), 2
            ),
            "rank_correlation": spearman(c, h),
        }
    crit = [float(r["critic_mean"]) for r in rows]
    hand = [sum(float(r[f"hand_{d}"]) for d in DIMS) / 4 for r in rows]
    best = (0.0, 0)
    for t in range(1, 10):
        tp = sum(1 for c, h in zip(crit, hand, strict=True) if c >= t and h >= 6)
        fp = sum(1 for c, h in zip(crit, hand, strict=True) if c >= t and h < 6)
        fn = sum(1 for c, h in zip(crit, hand, strict=True) if c < t and h >= 6)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        best = max(best, (round(f1, 3), t))
    out["suggested_pass_threshold"] = {"critic_mean_at_least": best[1], "f1": best[0]}
    out["score_spread"] = {"critic": max(crit) - min(crit), "hand": round(max(hand) - min(hand), 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["make", "report"])
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--wait", type=int, default=65)
    a = p.parse_args()
    asyncio.run(make(a.images, a.wait)) if a.step == "make" else report()
