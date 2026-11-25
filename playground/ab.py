"""A/B: is a weak image the PROMPT or the MODEL?

    uv run python playground/ab.py "perfume bottle on marble" \
        --models seedream-5-flash seedream-5-pro nano-banana-2-lite --final-raw
    uv run python playground/ab.py "perfume bottle on marble" --models seedream-5-flash --run

Arms per model: `pipeline` (the full engine), `raw` (your words straight to the model) and, with
--final-raw, `final_raw` (the engine's final prompt sent raw). A dry run (the default) only prints
the paid estimate. --run calls providers, so the estimate must be under --cap (default 0.50 USD).
Your .env decides what can run (paid models need OPENROUTER_IMAGES=true). LangSmith tracing is
forced off, prompts come from the repo, and the traces are saved next to the images:
outputs/ab/<name>/ has the images, side_by_side.png, report.md/json and one trace per pipeline run.
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `evals`

from evals import ab_compare as ab  # noqa: E402

from app.core.plans import Plan  # noqa: E402
from app.gateway.registry import load_registry  # noqa: E402


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--final-raw", action="store_true", help="also send the final prompt raw")
    ap.add_argument("--plan", default="pro", choices=[p.value for p in Plan])
    ap.add_argument("--ratio", default="1:1")
    ap.add_argument("--cap", type=float, default=0.50, help="max paid USD for this run")
    ap.add_argument("--run", action="store_true", help="really run (default is a dry run)")
    ap.add_argument("--out", type=Path)
    return ap.parse_args()


async def main() -> int:
    args = parse()
    arms = ["pipeline", "raw"] + (["final_raw"] if args.final_raw else [])
    reg = load_registry(run_testing=True)
    cards = [c for m in args.models if (c := reg.get(m))]
    if len(cards) != len(args.models):
        print("Unknown model id in --models")
        return 2
    est = ab.estimate_usd(cards, ["openrouter", "cloudflare"], {"openrouter", "fal"}, arms)
    print(
        f"{len(cards)} models x {len(arms)} arms. Estimated paid cost ${est:.3f} (cap ${args.cap})."
    )
    if not args.run:
        print("Dry run. Add --run to make the images.")
        return 0
    if est > args.cap:
        print("Refused: the estimate is over the cap. Ask the owner, then re-run with --cap.")
        return 3
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["PROMPT_SOURCE"] = "local"
    os.environ["TRACE_UI"] = "true"
    os.environ["SPEND_CAP_USD"] = str(args.cap)
    from app.agents.image_grand.generation.create.factory import build_create_deps
    from app.core.config import get_settings

    deps, http = build_create_deps(get_settings())
    await deps.prompts.load()
    slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40]
    out = args.out or Path("outputs/ab") / slug
    results = await ab.run_ab(
        deps,
        args.prompt,
        cards,
        out_dir=out,
        arms=tuple(arms),
        plan=Plan(args.plan),
        ratio=args.ratio,
    )
    print("Report:", ab.write_report(results, out, args.prompt))
    await http.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
