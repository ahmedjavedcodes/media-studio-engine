"""E1 (FREE: Groq vision): does the Critic still agree at smaller thumbnails?
Scores the SAME images at 512, 384 and 256 px and compares to 512.

    PYTHONPATH=src:playground .venv/bin/python playground/e1_thumbnail_size.py [--images 3]

The spec's version scores 50 images against HAND scores; this measures agreement with the 512 px
reading (no hand scores yet) and tokens per image, which is what sets free-tier throughput.
Reports per size: prompt tokens per image, mean absolute score difference from 512, rank
correlation with 512, and whether the winner changed. Add hand scores later to the same output.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from eval_common import MEASURE, CriticRig, spearman, stored_image_ids

from media_engine.ports import LocalArtifactStore

PROMPT = "a lighthouse on a rocky coast during a storm, waves crashing"
SIZES = (512, 384, 256)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--wait", type=int, default=65)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()
    ids = stored_image_ids()[-args.images :]
    store = LocalArtifactStore(MEASURE / "artifacts")
    by_size: dict[int, dict[str, dict[str, float]]] = {}
    tokens: dict[int, float] = {}
    for i, size in enumerate(SIZES):
        if i:
            await asyncio.sleep(args.wait)
        rig = CriticRig(store, size=size)
        try:
            reviews, usage = await rig.rank(ids, args.prompt)
        finally:
            await rig.close()
        if usage["status"] != "done":
            print(size, "failed:", usage)
            continue
        by_size[size] = {
            r["image_id"]: {"score": r.get("quality_score"), "rank": r.get("rank")}
            for r in reviews
            if "quality_score" in r
        }
        tokens[size] = round(usage["prompt_tokens"] / len(ids), 1)
        print(f"{size}px: {tokens[size]} prompt tokens per image via {usage['providers']}")
    base = by_size.get(512, {})
    report: dict[str, object] = {"images": len(ids), "tokens_per_image": tokens, "sizes": {}}
    for size, got in by_size.items():
        common = [k for k in got if k in base]
        report["sizes"][str(size)] = {  # type: ignore[index]
            "mean_abs_score_diff_vs_512": round(
                sum(abs(got[k]["score"] - base[k]["score"]) for k in common) / max(1, len(common)),
                2,
            ),
            "rank_correlation_vs_512": spearman(
                [got[k]["rank"] for k in common], [base[k]["rank"] for k in common]
            ),
            "same_winner_as_512": bool(common)
            and min(got, key=lambda k: got[k]["rank"]) == min(base, key=lambda k: base[k]["rank"]),
            "scores": {k[-6:]: got[k]["score"] for k in got},
        }
    (MEASURE / "e1.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
