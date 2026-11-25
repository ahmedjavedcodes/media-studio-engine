"""txt2img eval: the golden prompts on the curated models, the critic's scores, a winner table.

    uv run python playground/eval.py                       # dry run: what a full run would cost
    uv run python playground/eval.py --run --cap 2 --models seedream-5-flash --per-category 1
    uv run python playground/eval.py --report              # per-category, per-plan winner table
    uv run python playground/eval.py --report --write-rank # ...and write `auto_rank` to models.yaml
    uv run python playground/eval.py --push-dataset        # create the LangSmith dataset once

Nothing is called unless you pass --run. A run refuses an estimate above --cap (default 2 USD):
the full 60 prompts x 14 models sweep costs far more, so it needs the owner's go-ahead and a higher
--cap. The run uses your .env as it is: paid models only run when OPENROUTER_IMAGES=true, and
`testing` models only with --allow-testing. LangSmith tracing is forced off here, and prompts come
from the repo (prompts/create/txt2img/), not the Hub, so the run measures the files on disk.

Human ratings: put {"<prompt id>|<model id>": 1 to 5} in evals/results/ratings.json, then re-run
--report. A rating counts as much as the critic.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `evals`

from evals import txt2img_eval as ev  # noqa: E402

from app.gateway.registry import load_registry  # noqa: E402


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", action="store_true", help="really run (default is a dry run)")
    ap.add_argument("--report", action="store_true", help="build the winner table from results")
    ap.add_argument("--write-rank", action="store_true", help="with --report: write auto_rank")
    ap.add_argument("--push-dataset", action="store_true", help="create the LangSmith dataset")
    ap.add_argument("--models", nargs="*", help="model ids (default: the 14 curated models)")
    ap.add_argument("--categories", nargs="*", choices=ev.CATEGORIES)
    ap.add_argument("--per-category", type=int, help="use only the first N prompts of a category")
    ap.add_argument("--cap", type=float, default=2.0, help="max paid USD for this run")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--allow-testing", action="store_true", help="let `testing` models run")
    ap.add_argument("--results", nargs="*", type=Path, help="result files for --report")
    ap.add_argument("--redo", type=Path, help="re-run only the cells that failed in this file")
    return ap.parse_args()


def dry_run(args: argparse.Namespace) -> int:
    """No keys, no calls: what a run would cost if every chosen model could run."""
    reg = load_registry(run_testing=True)
    cards = [c for m in (args.models or []) if (c := reg.get(m))] or ev.curated_models(reg)
    cells = ev.make_cells(
        ev.load_dataset(),
        cards,
        categories=tuple(args.categories or ()) or None,
        per_category=args.per_category,
    )
    plan = ev.plan_run(cells, reg, ["openrouter", "cloudflare"], {"openrouter", "fal"})
    print(f"{len(cells)} cells = {len({c.prompt_id for c in cells})} prompts x {len(cards)} models")
    print(
        f"Estimated PAID cost if every model runs: ${plan.estimate_usd:.2f} (cap ${args.cap:.2f})"
    )
    for model_id, usd in sorted(plan.per_model_usd.items(), key=lambda kv: -kv[1]):
        print(f"  {model_id:28s} ${usd:7.3f}")
    print("Free Cloudflare neurons are used too: the enhancer, the critic and the safety check.")
    over = plan.estimate_usd > args.cap
    print("OVER THE CAP: ask the owner before running this." if over else "Within the cap.")
    return 0


async def real_run(args: argparse.Namespace) -> int:
    os.environ["LANGSMITH_TRACING"] = "false"  # the monthly trace quota is not for evals
    os.environ["PROMPT_SOURCE"] = "local"  # measure the prompt files, not an older Hub tag
    if args.allow_testing:
        os.environ["ALLOW_TESTING_MODELS"] = "true"
    from app.agents.image_grand.generation.create.factory import build_create_deps
    from app.core.config import get_settings

    s = get_settings()
    deps, http = build_create_deps(s)
    paid = set(getattr(deps.gateway, "_paid", set()))
    cards = [c for m in (args.models or []) if (c := deps.registry.get(m))] or ev.curated_models(
        deps.registry
    )
    cells = ev.make_cells(
        ev.load_dataset(),
        cards,
        categories=tuple(args.categories or ()) or None,
        per_category=args.per_category,
    )
    if args.redo:
        failed = {
            (r["prompt_id"], r["model_id"])
            for r in ev.load_rows(args.redo)
            if r["status"] in ("error", "no_image")
        }
        cells = [c for c in cells if (c.prompt_id, c.model_id) in failed]
    plan = ev.plan_run(cells, deps.registry, deps.enabled_providers, paid)
    print(f"{len(plan.cells)} cells, est. paid cost ${plan.estimate_usd:.2f}, cap ${args.cap:.2f}")
    if plan.skipped_models:
        print("Not run (provider off or model still `testing`):", ", ".join(plan.skipped_models))
    if plan.estimate_usd > args.cap:
        print("Refused: the estimate is over the cap. Ask the owner, then re-run with --cap.")
        await http.aclose()
        return 3
    if not plan.cells:
        await http.aclose()
        return 2
    await deps.prompts.load()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = ev.RESULTS_DIR / f"txt2img-{stamp}.jsonl"
    rows = await ev.run_eval(
        deps,
        plan.cells,
        cap_usd=args.cap,
        out_file=out_file,
        images_dir=ev.RESULTS_DIR / f"images-{stamp}",
        concurrency=args.concurrency,
    )
    spent = sum(r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), int | float))
    print(f"Done. {len(rows)} rows in {out_file}. Billed ${spent:.4f}.")
    await http.aclose()
    return 0


def report(args: argparse.Namespace) -> int:
    files = args.results or sorted(ev.RESULTS_DIR.glob("txt2img-*.jsonl"))
    if not files:
        print("No result files yet. Run with --run first.")
        return 2
    reg = load_registry(run_testing=True)
    rows = ev.load_rows(*files)
    table = ev.score_table(rows, ev.load_ratings())
    ranks = ev.winners(table, reg)
    print("Mean aesthetic and realism (critic, 0 to 10):\n")
    print(ev.render_aesthetic(ev.aesthetic_report(rows)))
    if not ranks:
        print(f"No model has {ev.MIN_CELLS} scored cells in a category yet (no winner table).")
        return 0
    print(ev.render_table(ranks, table))
    if args.write_rank:
        ev.write_auto_rank(ev.MODELS_YAML, ranks)
        print(f"\nWrote auto_rank to {ev.MODELS_YAML}")
    return 0


def push_dataset() -> int:
    from langsmith import Client

    from app.core.config import get_settings

    key = get_settings().langsmith_api_key
    if not key:
        print("LANGSMITH_API_KEY is not set.")
        return 2
    result = ev.push_dataset(Client(api_key=key), ev.load_dataset())
    print(f"LangSmith dataset {ev.DATASET_NAME!r}: {result}")
    return 0


def main() -> int:
    args = parse()
    if args.push_dataset:
        return push_dataset()
    if args.report:
        return report(args)
    if args.run:
        return asyncio.run(real_run(args))
    return dry_run(args)


if __name__ == "__main__":
    sys.exit(main())
