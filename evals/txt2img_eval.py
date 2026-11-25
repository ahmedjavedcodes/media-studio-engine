"""Eval harness for txt2img (the logic behind playground/eval.py).

A cell is one golden prompt run on one model. A run makes ONE image per cell at 1K on the Pro
plan (every tier allowed, GPT Image at medium), keeps the critic's scores, and leaves a slot for a
human 1 to 5 rating. The report ranks the models per category and per plan and writes the result
into models.yaml as `auto_rank`, which the router reads first.

Money: a paid cell is real OpenRouter spend. `plan_run` estimates the cost before anything runs,
`run_eval` stops at the cap, and the CLI refuses an estimate above the cap. Nothing here switches a
provider on: the owner's OPENROUTER_IMAGES / ALLOW_TESTING_MODELS settings decide what can run.
The critic is a small free vision model, so its scores are a first filter, not the last word:
add human ratings (evals/results/ratings.json) and re-run the report.
"""

import asyncio
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.generation.create.txt2img import build_txt2img_graph
from app.agents.image_grand.workers import plan_rules
from app.agents.image_grand.workers.shot_plan import ShotPlan
from app.contracts.request import ImageSettings
from app.core.plans import Plan, tier_allowed
from app.gateway import registry as registry_module
from app.gateway.options import quality_for
from app.gateway.registry import ModelCard, Registry, load_registry

ROOT = Path(__file__).parent
DATASET = ROOT / "datasets" / "txt2img_golden.json"
RESULTS_DIR = ROOT / "results"
RATINGS_FILE = RESULTS_DIR / "ratings.json"
MODELS_YAML = Path(registry_module.__file__).with_name("models.yaml")
DATASET_NAME = "txt2img-golden"

CATEGORIES = ("product", "photo", "people", "text_in_image", "design", "general")
PLANS = (Plan.BASIC, Plan.PRO, Plan.ULTRA, Plan.MAX)
EVAL_PLAN = Plan.PRO  # every tier is allowed, and GPT Image runs at its medium quality
HUMAN_WEIGHT = 0.5  # a human 1 to 5 rating counts as much as the critic once it exists
MIN_CELLS = 3  # a model with fewer scored cells in a category is not ranked there

RANK_BEGIN = "# BEGIN auto_rank (written by playground/eval.py --write-rank, do not edit by hand)"
RANK_END = "# END auto_rank"


# ---- the dataset --------------------------------------------------------------------------------


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text("utf-8"))["examples"])


@dataclass(frozen=True)
class Cell:
    prompt_id: str
    category: str  # the router's category (auto_rank groups by it)
    lang: str
    prompt: str
    model_id: str
    subject_type: str = ""  # product, person, animal, place, food, fantasy, poster, abstract

    @property
    def key(self) -> str:
        return f"{self.prompt_id}|{self.model_id}"


def curated_models(registry: Registry) -> list[ModelCard]:
    """The 14 curated models: the ones the picker shows."""
    return [c for c in registry.all() if c.picker]


def make_cells(
    dataset: list[dict[str, Any]],
    cards: list[ModelCard],
    *,
    categories: tuple[str, ...] | None = None,
    per_category: int | None = None,
) -> list[Cell]:
    taken: dict[str, int] = defaultdict(int)
    cells: list[Cell] = []
    for ex in dataset:
        cat = ex["category"]
        if categories and cat not in categories:
            continue
        if per_category is not None and taken[cat] >= per_category:
            continue
        taken[cat] += 1
        cells += [
            Cell(ex["id"], cat, ex["lang"], ex["prompt"], c.id, ex.get("subject_type", ""))
            for c in cards
        ]
    return cells


# ---- the cost -----------------------------------------------------------------------------------


def cell_cost_usd(card: ModelCard, enabled: list[str], paid: set[str]) -> float:
    """Estimated PAID cost of one cell: the cheapest enabled provider, if it is a paid one. A model
    that has no enabled provider cannot run, so it costs nothing and is skipped by the runner."""
    entries = card.providers_in(enabled)
    if not entries or entries[0].name not in paid:
        return 0.0
    return card.cost_for(entries[0], quality_for(card, EVAL_PLAN, manual=False))


def runnable(card: ModelCard, registry: Registry, enabled: list[str]) -> bool:
    return card in registry.live(enabled)


@dataclass(frozen=True)
class RunPlan:
    cells: list[Cell]
    skipped_models: list[str]  # no enabled provider, or not allowed to run (still `testing`)
    estimate_usd: float
    per_model_usd: dict[str, float]


def plan_run(cells: list[Cell], registry: Registry, enabled: list[str], paid: set[str]) -> RunPlan:
    by_id = {c.model_id for c in cells}
    skipped = sorted(
        m
        for m in by_id
        if (card := registry.get(m)) is None or not runnable(card, registry, enabled)
    )
    keep = [c for c in cells if c.model_id not in skipped]
    per_model: dict[str, float] = defaultdict(float)
    for c in keep:
        card = registry.get(c.model_id)
        assert card is not None
        per_model[c.model_id] += cell_cost_usd(card, enabled, paid)
    return RunPlan(keep, skipped, round(sum(per_model.values()), 4), dict(per_model))


# ---- one cell -----------------------------------------------------------------------------------


def _rule_codes(prompt: str, state: dict[str, Any], card: ModelCard) -> list[str]:
    """The art-director rules the plan and the written prompt broke (the CI check's rules, run on
    real output)."""
    try:
        plan = ShotPlan.model_validate(state.get("shot_plan") or {})
    except ValueError:
        return ["unreadable_plan"]
    found = plan_rules.check_plan(prompt, plan, valid_looks=None)
    for written in (state.get("prompts") or [])[:1]:
        found += plan_rules.check_prompt(prompt, plan, written, max_words=card.prompt_max_words)
    return sorted({v.code for v in found})


async def run_cell(
    deps: CreateDeps, cell: Cell, *, images_dir: Path | None = None
) -> dict[str, Any]:
    """Run one cell through the real graph. Never raises: a failure is a row with a status."""
    card = deps.registry.get(cell.model_id)
    assert card is not None
    run_id = f"eval-{cell.prompt_id}-{cell.model_id}"
    settings = ImageSettings(model=cell.model_id, aspect_ratio="1:1", n=1).model_dump()
    state_in = {
        "run_id": run_id,
        "user_id": run_id,  # one user per cell, so the fake credits never run out
        "prompt": cell.prompt,
        "plan": EVAL_PLAN.value,
        "settings": settings,
        "has_images": False,
        "n_refs": 0,
        "results": [],
    }
    row: dict[str, Any] = {
        **asdict(cell),
        "plan": EVAL_PLAN.value,
        "status": "error",
        "code": "",
        "human_rating": None,  # fill in 1 to 5 in ratings.json, then re-run --report
    }
    started = time.monotonic()
    deps.tracer.start_run(run_id, metadata={"eval": True, "model": cell.model_id}, inputs=state_in)
    try:
        graph = build_txt2img_graph(deps).compile(checkpointer=InMemorySaver())
        out = await graph.ainvoke(state_in, {"configurable": {"thread_id": run_id}})
    except Exception as exc:  # noqa: BLE001 - one bad cell never stops the run
        row["code"] = type(exc).__name__
        return row
    finally:
        row["wall_ms"] = int((time.monotonic() - started) * 1000)
    if "__interrupt__" in out:
        return {**row, "status": "asked"}  # the enhancer wanted a question answered
    if out.get("error"):
        return {**row, "code": out["error"].get("code", "error")}
    ok = [r for r in out.get("results", []) if r.get("ok")]
    if not ok:
        return {**row, "status": "no_image"}
    image = ok[0]
    verdict = (out.get("review", {}).get("verdicts") or {}).get("img_1") or {}
    scored = verdict.get("mean") is not None
    row.update(
        status="ok" if scored else "unscored",
        provider=image.get("provider"),
        cost_usd=image.get("cost_usd"),
        latency_ms=image.get("latency_ms"),
        final_prompt=(out.get("prompts") or [None])[0],
        look=(out.get("shot_plan") or {}).get("look"),
        subject_type=out.get("subject_type") or cell.subject_type,
        content_category=out.get("content_category"),
        rule_violations=_rule_codes(cell.prompt, out, card),
        critic={
            k: verdict.get(k)
            for k in (
                "match",
                "craft",
                "interest",
                "technical",
                "aesthetic",
                "realism",
                "mean",
                "flags",
            )
        },
    )
    if images_dir is not None:
        save_cell_report(deps, run_id, cell, row, images_dir)
        data = deps.store.read_bytes(run_id, image["file"])
        if data is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            name = f"{cell.model_id}__{cell.prompt_id}.{image['file'].rsplit('.', 1)[-1]}"
            (images_dir / name).write_bytes(data)
            row["image"] = f"{images_dir.name}/{name}"
    return row


class CachingLLM:
    """Wraps the Art Director's model for an eval. Its system prompt and input do not depend on the
    image model, so one prompt is planned ONCE and every model gets the same ShotPlan. That makes
    the comparison fair and keeps Groq's per-minute token limit out of the way."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model = inner.model
        self.provider = getattr(inner, "provider", "cloudflare")
        self._done: dict[tuple[str, str, str], asyncio.Future[Any]] = {}

    async def complete_json(self, **kw: Any) -> Any:
        key = (kw["system"], kw["user"], kw["schema"].__name__)
        fut = self._done.get(key)
        if fut is not None:
            return (await fut).model_copy(deep=True)  # a cache hit costs no tokens
        fut = asyncio.get_running_loop().create_future()
        self._done[key] = fut
        try:
            out = await self._inner.complete_json(**kw)
        except BaseException as exc:
            del self._done[key]  # a failure is not cached
            fut.set_exception(exc)
            fut.exception()  # mark it retrieved
            raise
        fut.set_result(out)
        return out


def save_cell_report(
    deps: CreateDeps, run_id: str, cell: Cell, row: dict[str, Any], folder: Path
) -> None:
    """One markdown per cell, input to output: every step with its time, model and cost, each
    agent's system prompt and output, the final prompt, the critic's scores. Plus the raw trace."""
    trace = deps.tracer.trace_for(run_id)
    if trace is None:
        return
    snap = trace.snapshot(row["status"])
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{cell.model_id}__{cell.prompt_id}"
    (folder / f"{stem}.trace.json").write_text(json.dumps(snap, indent=1, ensure_ascii=False))
    spans = {s["name"]: s for s in snap["spans"]}
    t = snap.get("totals", {})
    lines = [
        f"# {cell.prompt_id} on {cell.model_id}",
        f"**Input:** {cell.prompt}  ({cell.lang}, {cell.subject_type})",
        f"**Total:** {t.get('duration_ms')} ms, ${t.get('total_usd')} "
        f"(image ${t.get('images_usd')}, LLM ${t.get('llm_usd_estimate')})",
        "",
        "| step | ms | model | detail |",
        "|---|---|---|---|",
    ]
    for s in snap["spans"]:
        o = s.get("outputs") or {}
        detail = o.get("cost_usd") or o.get("verdict") or o.get("look") or ""
        lines.append(
            f"| {s['name']} | {s.get('duration_ms')} | {s.get('model') or ''} | {detail} |"
        )
    for name in ("art_director", "prompt_writer"):
        o = (spans.get(name) or {}).get("outputs") or {}
        lines += ["", f"## {name}", "### system prompt", "```", str(o.get("system_prompt")), "```"]
        lines += [
            "### input",
            "```json",
            json.dumps(o.get("user_input"), ensure_ascii=False),
            "```",
        ]
        out = o.get("shot_plan") if name == "art_director" else o.get("prompts")
        lines += ["### output", "```json", json.dumps(out, indent=1, ensure_ascii=False), "```"]
        lines.append(
            f"attempts {o.get('attempts')}, retry {o.get('retry_reason')}, "
            f"violations {o.get('violations')}"
        )
    g = (spans.get("code_guard") or {}).get("outputs") or {}
    lines += [
        "",
        "## sent to the image model",
        g.get("final_prompt") or "",
        f"size {g.get('size')}",
        "",
        "## critic",
        json.dumps(row.get("critic")),
    ]
    (folder / f"{stem}.md").write_text("\n".join(lines), "utf-8")


async def run_eval(
    deps: CreateDeps,
    cells: list[Cell],
    *,
    cap_usd: float,
    out_file: Path,
    images_dir: Path | None = None,
    concurrency: int = 2,
    say: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run the cells, append every row to `out_file`, stop before the cap would be passed."""
    paid = set(getattr(deps.gateway, "_paid", set()))
    deps.enhancer = CachingLLM(deps.enhancer)  # one plan per prompt, shared by every model
    spent = 0.0
    rows: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    async def one(cell: Cell) -> None:
        nonlocal spent
        card = deps.registry.get(cell.model_id)
        assert card is not None
        estimate = cell_cost_usd(card, deps.enabled_providers, paid)
        async with sem:
            async with lock:  # reserve the estimate, so parallel cells cannot pass the cap
                if spent + estimate > cap_usd:
                    row = {**asdict(cell), "status": "skipped_cap", "human_rating": None}
                    rows.append(row)
                    return
                spent += estimate
            row = await run_cell(deps, cell, images_dir=images_dir)
            async with lock:
                billed = row.get("cost_usd")
                if isinstance(billed, int | float):
                    spent += float(billed) - estimate  # the real price replaces the estimate
                rows.append(row)
                with out_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                say(f"[{len(rows)}/{len(cells)}] {cell.model_id} {cell.prompt_id}: {row['status']}")

    await asyncio.gather(*(one(c) for c in cells))
    return rows


# ---- the report ---------------------------------------------------------------------------------


def load_rows(*files: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in files:
        for line in f.read_text("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_ratings(path: Path = RATINGS_FILE) -> dict[str, float]:
    """{"<prompt id>|<model id>": 1 to 5}. A missing file means no human ratings yet."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text("utf-8"))
    return {k: float(v) for k, v in raw.items() if isinstance(v, int | float) and 1 <= v <= 5}


def cell_score(row: dict[str, Any], ratings: dict[str, float]) -> float | None:
    """0 to 10: the critic's mean, blended with the human rating (1 to 5, doubled) if one exists."""
    critic = (row.get("critic") or {}).get("mean")
    if row.get("status") != "ok" or critic is None:
        return None
    human = ratings.get(f"{row['prompt_id']}|{row['model_id']}", row.get("human_rating"))
    if isinstance(human, int | float) and 1 <= human <= 5:
        return (1 - HUMAN_WEIGHT) * float(critic) + HUMAN_WEIGHT * float(human) * 2
    return float(critic)


@dataclass(frozen=True)
class ModelScore:
    model_id: str
    score: float  # mean over scored cells, 0 to 10
    cells: int
    failures: int  # cells that made no usable image (errors, unsafe, no image)
    cost_usd: float
    rule_breaks: int  # cells where the enhancer broke an art-director rule


def score_table(
    rows: list[dict[str, Any]], ratings: dict[str, float] | None = None
) -> dict[str, dict[str, ModelScore]]:
    ratings = ratings or {}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "skipped_cap":
            buckets[(r["category"], r["model_id"])].append(r)
    table: dict[str, dict[str, ModelScore]] = defaultdict(dict)
    for (cat, model_id), items in buckets.items():
        scores = [s for r in items if (s := cell_score(r, ratings)) is not None]
        if not scores:
            continue
        costs = [r["cost_usd"] for r in items if isinstance(r.get("cost_usd"), int | float)]
        table[cat][model_id] = ModelScore(
            model_id=model_id,
            score=round(sum(scores) / len(scores), 3),
            cells=len(scores),
            failures=sum(1 for r in items if r.get("status") not in ("ok", "unscored")),
            cost_usd=round(sum(costs) / len(costs), 5) if costs else 0.0,
            rule_breaks=sum(1 for r in items if r.get("rule_violations")),
        )
    return dict(table)


def winners(
    table: dict[str, dict[str, ModelScore]], registry: Registry, *, min_cells: int = MIN_CELLS
) -> dict[str, dict[str, list[str]]]:
    """{category: {plan: [model ids, best first]}}. A plan lists only models its tier allows.
    Ties go to the cheaper model."""
    out: dict[str, dict[str, list[str]]] = {}
    for cat, models in table.items():
        out[cat] = {}
        for plan in PLANS:
            ranked = sorted(
                (
                    m
                    for m in models.values()
                    if m.cells >= min_cells
                    and (card := registry.get(m.model_id)) is not None
                    and tier_allowed(plan, card.tier)
                ),
                key=lambda m: (-m.score, m.cost_usd),
            )
            if ranked:
                out[cat][plan.value] = [m.model_id for m in ranked]
    return {c: p for c, p in out.items() if p}


def aesthetic_report(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean critic `aesthetic` and `realism` per look, per subject type and per model:
    {"look": {name: {"aesthetic": 7.8, "realism": 8.1, "n": 12}}, "subject_type": ...}.
    Only rows the critic scored count."""
    out: dict[str, dict[str, dict[str, float]]] = {"look": {}, "subject_type": {}, "model": {}}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        crit = r.get("critic") or {}
        if r.get("status") != "ok" or crit.get("aesthetic") is None:
            continue
        for axis, key in (
            ("look", "look"),
            ("subject_type", "subject_type"),
            ("model", "model_id"),
        ):
            if r.get(key):
                buckets[(axis, r[key])].append(crit)
    for (axis, name), crits in buckets.items():
        real = [c["realism"] for c in crits if c.get("realism") is not None]
        out[axis][name] = {
            "aesthetic": round(sum(c["aesthetic"] for c in crits) / len(crits), 2),
            "realism": round(sum(real) / len(real), 2) if real else 0.0,
            "n": len(crits),
        }
    return out


def render_aesthetic(report: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = []
    for axis, title in (("look", "Look"), ("subject_type", "Subject type"), ("model", "Model")):
        lines += [f"| {title} | aesthetic | realism | images |", "|---|---|---|---|"]
        for name, v in sorted(report[axis].items(), key=lambda kv: -kv[1]["aesthetic"]):
            lines.append(f"| {name} | {v['aesthetic']:.2f} | {v['realism']:.2f} | {int(v['n'])} |")
        lines.append("")
    return "\n".join(lines)


def render_table(
    ranks: dict[str, dict[str, list[str]]], table: dict[str, dict[str, ModelScore]]
) -> str:
    """Markdown: one row per category, one column per plan, each cell the winner and its score."""
    head = "| Category | " + " | ".join(p.value.title() for p in PLANS) + " |"
    lines = [head, "|" + "---|" * (len(PLANS) + 1)]
    for cat in CATEGORIES:
        if cat not in ranks:
            continue
        cells = []
        for plan in PLANS:
            ids = ranks[cat].get(plan.value)
            cells.append(f"{ids[0]} ({table[cat][ids[0]].score:.1f})" if ids else "-")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- writing auto_rank into models.yaml ---------------------------------------------------------


def render_rank_block(ranks: dict[str, dict[str, list[str]]]) -> str:
    lines = [RANK_BEGIN, "auto_rank:"]
    for cat in CATEGORIES:
        if cat not in ranks:
            continue
        lines.append(f"  {cat}:")
        for plan in PLANS:
            if ids := ranks[cat].get(plan.value):
                lines.append(f"    {plan.value}: [{', '.join(ids)}]")
    lines.append(RANK_END)
    return "\n".join(lines) + "\n"


def write_auto_rank(path: Path, ranks: dict[str, dict[str, list[str]]]) -> None:
    """Replace the auto_rank block (between the markers) or add it at the end. Everything else in
    the file, comments included, is left as it was. The result is loaded back to check it."""
    text = path.read_text("utf-8")
    block = render_rank_block(ranks)
    pattern = re.compile(rf"{re.escape(RANK_BEGIN)}.*?{re.escape(RANK_END)}\n?", re.DOTALL)
    if pattern.search(text):
        new = pattern.sub(lambda _m: block, text)
    elif re.search(r"^auto_rank:", text, re.MULTILINE):
        raise ValueError("models.yaml has an auto_rank outside the markers. Move it by hand.")
    else:
        new = text.rstrip("\n") + "\n\n" + block
    tmp = path.with_suffix(".tmp")
    tmp.write_text(new, "utf-8")
    try:
        load_registry(tmp)  # unknown model ids or a broken file raise here
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


# ---- the LangSmith dataset ----------------------------------------------------------------------


def push_dataset(client: Any, dataset: list[dict[str, Any]], name: str = DATASET_NAME) -> str:
    """Create the LangSmith dataset once. Returns "created" or "exists". No traces are sent."""
    if client.has_dataset(dataset_name=name):
        return "exists"
    made = client.create_dataset(
        name,
        description="Golden prompts for txt2img: eight subject types, Roman Urdu and Urdu.",
    )
    client.create_examples(
        dataset_id=made.id,
        examples=[
            {
                "inputs": {"prompt": ex["prompt"]},
                "outputs": {"subject_type": ex.get("subject_type"), "human_rating": None},
                "metadata": {
                    "id": ex["id"],
                    "lang": ex["lang"],
                    "subject_type": ex.get("subject_type"),
                    "category": ex["category"],
                },
            }
            for ex in dataset
        ],
    )
    return "created"
