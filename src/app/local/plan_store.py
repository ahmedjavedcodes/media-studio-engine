"""The plan store (dev). Good ShotPlans, kept to teach the Art Director, and project memory.

Every delivered image stores its ShotPlan with its subject type and the critic's aesthetic score.
For a new prompt the store returns the top 2 similar plans scored 8 or more, and the Art Director
sees them as few-shot examples. It starts with a seed set of hand-checked plans (seed_plans.json).

Similarity is lexical (words in common), plus a bonus when the subject type matches, so it needs no
embedding call and works offline. A project can pin a style: its first approved plan gives the
look, palette and props that later requests reuse (a hook, not wired to a screen yet).
"""

import json
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("plan_store")

SEEDS = Path(__file__).with_name("seed_plans.json")
MIN_AESTHETIC = 8.0  # only plans the critic scored this high are shown to the Art Director
TYPE_BONUS = 0.35
_TOKENS = re.compile(r"[a-z]+|[؀-ۿ]+")
_STOP = frozenset("a an the of on in with and at to for by is are it its from into over".split())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
  id TEXT PRIMARY KEY, source TEXT NOT NULL, project_id TEXT, subject_type TEXT NOT NULL,
  look TEXT NOT NULL, prompt TEXT NOT NULL, plan TEXT NOT NULL,
  aesthetic REAL, realism REAL, approved INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
)"""


@dataclass(frozen=True)
class StoredPlan:
    id: str
    source: str  # "seed" or "run"
    subject_type: str
    look: str
    prompt: str
    plan: dict[str, Any]
    aesthetic: float | None
    realism: float | None


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKENS.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _norm(text: str) -> str:
    return " ".join(_TOKENS.findall(text.lower()))


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


class PlanStore:
    def __init__(self, path: str | Path = ":memory:", *, seeds: Path | None = SEEDS) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(_SCHEMA)
            self._db.commit()
        if seeds is not None and seeds.exists():
            self.load_seeds(seeds)

    # ---- writing -----------------------------------------------------------------------------

    def load_seeds(self, path: Path) -> int:
        """Insert the hand-checked seed plans that are not there yet. Returns how many are new."""
        added = 0
        for item in json.loads(path.read_text("utf-8")):
            added += self._insert(
                plan_id=f"seed-{item['id']}",
                source="seed",
                project_id=None,
                subject_type=item["subject_type"],
                look=item["look"],
                prompt=item["prompt"],
                plan=item["plan"],
                aesthetic=float(item.get("aesthetic", 9.0)),
                realism=float(item.get("realism", 9.0)),
            )
        return added

    def add(
        self,
        *,
        run_id: str,
        index: int,
        subject_type: str,
        look: str,
        prompt: str,
        plan: dict[str, Any],
        aesthetic: float | None,
        realism: float | None,
        project_id: str | None = None,
    ) -> str:
        """Store the plan of one delivered image, with the critic's scores."""
        plan_id = f"{run_id}-{index}"
        self._insert(
            plan_id=plan_id,
            source="run",
            project_id=project_id,
            subject_type=subject_type or "other",
            look=look,
            prompt=prompt,
            plan=plan,
            aesthetic=aesthetic,
            realism=realism,
        )
        return plan_id

    def _insert(self, *, plan_id: str, plan: dict[str, Any], **row: Any) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO plans (id, source, project_id, subject_type, look, prompt,"
                " plan, aesthetic, realism, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    plan_id,
                    row["source"],
                    row["project_id"],
                    row["subject_type"],
                    row["look"],
                    row["prompt"],
                    json.dumps(plan, ensure_ascii=False),
                    row["aesthetic"],
                    row["realism"],
                    time.time(),
                ),
            )
            self._db.commit()
            return cur.rowcount

    # ---- reading -----------------------------------------------------------------------------

    def similar(
        self,
        prompt: str,
        *,
        subject_type: str = "",
        k: int = 2,
        min_aesthetic: float = MIN_AESTHETIC,
    ) -> list[StoredPlan]:
        """The top `k` stored plans for a prompt, best first. Only plans scored `min_aesthetic` or
        more. Words in common plus a bonus for the same subject type. [] when nothing is close."""
        want = _tokens(prompt)
        with self._lock:
            rows = self._db.execute(
                "SELECT id, source, subject_type, look, prompt, plan, aesthetic, realism "
                "FROM plans WHERE aesthetic >= ?",
                (min_aesthetic,),
            ).fetchall()
        scored: list[tuple[float, float, str, StoredPlan]] = []
        for rid, source, stype, look, text, raw, aes, real in rows:
            if _norm(text) == _norm(prompt):
                continue  # teach by analogy: a plan for this very prompt would just be copied
            plan = json.loads(raw)
            have = _tokens(f"{text} {plan.get('subject', '')} {plan.get('world', '')}")
            score = _similarity(want, have)
            if subject_type and subject_type != "other" and stype == subject_type:
                score += TYPE_BONUS
            if score > 0:
                stored = StoredPlan(rid, source, stype, look, text, plan, aes, real)
                scored.append((-score, -(aes or 0.0), rid, stored))
        scored.sort(key=lambda t: t[:3])
        return [s[3] for s in scored[:k]]

    def count(self, source: str | None = None) -> int:
        with self._lock:
            if source is None:
                return int(self._db.execute("SELECT COUNT(*) FROM plans").fetchone()[0])
            return int(
                self._db.execute("SELECT COUNT(*) FROM plans WHERE source=?", (source,)).fetchone()[
                    0
                ]
            )

    # ---- project memory (a hook) ------------------------------------------------------------

    def approve(self, plan_id: str) -> bool:
        """Mark a plan as approved. The first approved plan of a project sets its style."""
        with self._lock:
            cur = self._db.execute("UPDATE plans SET approved=1 WHERE id=?", (plan_id,))
            self._db.commit()
            return cur.rowcount > 0

    def project_style(self, project_id: str) -> dict[str, Any] | None:
        """The look, palette and props of the project's FIRST approved plan, or None."""
        with self._lock:
            row = self._db.execute(
                "SELECT look, plan FROM plans WHERE project_id=? AND approved=1 "
                "ORDER BY created, id LIMIT 1",
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        plan = json.loads(row[1])
        return {"look": row[0], "palette": plan.get("palette", []), "props": plan.get("props", [])}

    def close(self) -> None:
        with self._lock:
            self._db.close()
