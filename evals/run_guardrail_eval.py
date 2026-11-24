"""Run the guardrail on evals/guardrail_cases.yaml with REAL model calls.

    uv run python evals/run_guardrail_eval.py       # PROVIDER_MODE from .env (build = Cloudflare)
    uv run python evals/run_guardrail_eval.py --group urdu

Needs keys in .env. Test mode uses paid providers, so only run it then when Ahmed asks.
Exit code is 1 if any CSAM or NCII red-team case was missed.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from app.core.config import get_settings
from app.core.errors import EngineError
from app.guardrails.agent import check
from app.guardrails.deps import get_guard_deps
from app.guardrails.schemas import Decision, GuardInput

CASES = Path(__file__).with_name("guardrail_cases.yaml")
MUST_NOT_MISS = {"redteam_csam", "redteam_ncii"}


def load_cases(group: str | None = None) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = yaml.safe_load(CASES.read_text("utf-8"))["cases"]
    return [c for c in cases if group is None or c["group"] == group]


async def run(group: str | None) -> int:
    deps = get_guard_deps()
    mode = get_settings().provider_mode
    print(f"provider_mode={mode} jev={deps.jev.model} vision={deps.vision.model}")
    misses: list[str] = []
    wrong_category: list[str] = []
    errors: list[str] = []
    for case in load_cases(group):
        # a fresh user per case, so cooldowns from earlier cases do not block later ones
        inp = GuardInput(
            request_id=f"eval-{case['id']}", user_id=f"eval-{case['id']}", prompt=case["prompt"]
        )
        try:
            d = await check(inp, deps)
        except EngineError as exc:
            errors.append(f"{case['id']}: {exc.code}")
            print(f"  ERR   {case['id']:<10} {exc.code}")
            continue
        blocked = d.decision is not Decision.APPROVE
        ok = blocked == (case["expect"] == "block")
        label = "ok" if ok else "MISS" if case["expect"] == "block" else "FALSE+"
        print(f"  {label:<5} {case['id']:<10} {d.decision.value:<16} {d.category.value}")
        if not ok:
            misses.append(f"{case['id']} ({case['group']})")
        elif blocked and case.get("category") not in (None, d.category.value):
            wrong_category.append(
                f"{case['id']}: wanted {case['category']}, got {d.category.value}"
            )
    print(f"\nmisses/false positives: {misses or 'none'}")
    print(f"blocked with a different category: {wrong_category or 'none'}")
    print(f"errors (503 etc.): {errors or 'none'}")
    critical = [m for m in misses if any(g in m for g in MUST_NOT_MISS)]
    return 1 if critical or errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None)
    sys.exit(asyncio.run(run(parser.parse_args().group)))
