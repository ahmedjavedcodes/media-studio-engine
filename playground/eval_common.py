"""Shared helpers for the E1-E7 experiment scripts (IMAGE-CREATE-OPTIMIZE §11). Playground code:
real calls happen only when a script is run by hand. Paid scripts also refuse without
--confirm-paid and check the ledger (.cache/paid_spend.jsonl) against the owner's $5 cap first."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from media_engine.agents.base import AgentConfig
from media_engine.agents.image_generation.create.critic import Critic
from media_engine.contracts import AuthContext, Goal
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.router import Gateway
from media_engine.harness import trace as tracing
from media_engine.ports import ListEventSink, LocalArtifactStore
from media_engine.schemas import CriticOutput
from media_engine.services.create.loader import load_create_config

ROOT = Path(__file__).resolve().parents[1]
MEASURE = ROOT / ".cache" / "measure"
LEDGER = ROOT / ".cache" / "paid_spend.jsonl"
SPEND_CAP_USD = 5.00
AUTH = AuthContext(plan="free", execution_mode="auto", model_demand="balanced")


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    return round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6)


def require_paid_ok(confirm: bool, worst_case_usd: float) -> None:
    """Refuse a paid script unless confirmed AND the worst case fits under the cap."""
    if not confirm:
        raise SystemExit(f"PAID (~${worst_case_usd:.3f}): re-run with --confirm-paid")
    spent = ledger_total()
    if spent + worst_case_usd > SPEND_CAP_USD:
        raise SystemExit(f"refused: ${spent:.4f} spent + ${worst_case_usd:.3f} > ${SPEND_CAP_USD}")


def log_paid(model: str, cost: float | None, purpose: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "model": model,
        "status": "200",
        "cost_usd": cost,
        "purpose": purpose,
    }
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def stored_image_ids(store_dir: Path | None = None) -> list[str]:
    """Originals only (never thumbnails), oldest first."""
    folder = store_dir or MEASURE / "artifacts"
    files = [p for p in folder.glob("img_*") if ".thumb" not in p.name]
    return [p.stem for p in sorted(files, key=lambda p: p.stat().st_mtime)]


class CriticRig:
    """The real Critic on the real gateway, over a local store. `size` overrides the thumbnail
    size it reads (E1); a size with no stored thumbnail is made by downscaling the original."""

    def __init__(self, store: LocalArtifactStore, size: int | None = None) -> None:
        self.cfg = load_create_config()
        self.gateway = Gateway()
        policy = self.cfg.agent("critic")
        if size is not None:
            policy = policy.model_copy(update={"params": {**policy.params, "thumbnail_size": size}})
        self.critic = Critic(
            AgentConfig.from_policy("critic", policy, CriticOutput),
            LLMGateway(self.gateway, chain_resolver(self.cfg.agents)),
            ListEventSink(),
            artifacts=store,
        )
        self.stop = self.cfg.policy.stopping_condition("critique", "free", "a review per image")

    async def rank(
        self, ids: list[str], prompt: str, must_have: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """(reviews in rank order, usage). Usage is read from the call records of this call."""
        trace = tracing.Trace()
        with tracing.scope(trace, "critique", 1, 0):
            env = await self.critic.run(
                Goal(task="critique", instructions="Review and rank the images."),
                self.stop,
                AUTH,
                {"image_ids": ids, "prompt": prompt, "must_have": must_have or [prompt]},
            )
        calls = trace.snapshot()["calls"]
        usage = {
            "status": env.state.status.value,
            "reason": env.state.reason,
            "calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "providers": sorted({f"{c['provider']}/{c['model']}" for c in calls}),
            "neurons": round(sum(c["neurons"] for c in calls), 1),
            # what each call said and whether it was usable: shows WHY a repair was needed
            "replies": [
                {
                    "model": c["model"],
                    "failure": c.get("failure"),
                    "was_repair": c.get("was_repair"),
                    "reply": (c.get("reply_text") or "")[:240],
                }
                for c in calls
            ],
        }
        return list(env.output.get("reviews", [])), usage

    async def close(self) -> None:
        await self.gateway.aclose()


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation of two equal-length score lists (ties get average ranks)."""
    n = len(a)
    if n < 2 or n != len(b):
        return None

    def ranks(x: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: x[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return None if den == 0 else round(num / den, 3)
