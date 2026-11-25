"""A/B: is a weak image the PROMPT or the MODEL? (the logic behind playground/ab.py)

For each model it makes up to three images from the same user prompt:

- `pipeline`: the full engine (Art Director, Prompt Writer, code guard). The trace is saved.
- `raw`: the user's own words sent straight to the model. No engine at all.
- `final_raw`: the pipeline's final prompt sent raw (same options, no critic or retries). If it
  looks like `pipeline`, the prompt carries the quality. If it looks like `raw`, the model does.

Then ONE critic call scores every image together (aesthetic and realism are relative), and a
side-by-side picture is saved with the traces and the final prompts. Every paid image is
estimated first and the run refuses an estimate above the cap. Each image passes the same
per-image safety check as a real run; an unsafe one is dropped.
"""

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.workers import critic as critic_worker
from app.agents.image_grand.workers.safety import Safety, check_image_detail
from app.contracts.request import ImageSettings
from app.core.plans import Plan
from app.gateway.compiler import guard_prompt
from app.gateway.options import RequestExtras, options_for
from app.gateway.registry import ModelCard
from app.tools.imaging import make_thumbnail, sniff
from evals import txt2img_eval as ev

ARMS = ("pipeline", "raw", "final_raw")


@dataclass
class ArmResult:
    model_id: str
    arm: str
    prompt_sent: str = ""
    file: str = ""
    cost_usd: float | None = None
    status: str = "ok"
    scores: dict[str, Any] | None = None
    look: str | None = None


def estimate_usd(
    cards: list[ModelCard], enabled: list[str], paid: set[str], arms: list[str]
) -> float:
    """Paid cost of the images this run would make (pipeline, raw, final_raw: one image each)."""
    total = 0.0
    for card in cards:
        total += len(arms) * ev.cell_cost_usd(card, enabled, paid)
    return round(total, 4)


async def _raw_image(
    deps: CreateDeps,
    card: ModelCard,
    prompt: str,
    *,
    plan: Plan,
    ratio: str,
    dest: Path,
) -> tuple[bytes | None, float | None, str]:
    """Send `prompt` as it is. Only the size and the model's own options are added by code."""
    guarded = guard_prompt(card, prompt, aspect_ratio=ratio)  # the same size and blocklist rules
    options = options_for(card, RequestExtras(plan=plan, manual=True))
    result = await deps.gateway.generate(
        card,
        prompt=prompt,  # the text goes unchanged: that is the point of the arm
        negative_prompt="",
        width=guarded.width,
        height=guarded.height,
        aspect_ratio=ratio,
        seed=None,
        timeout=deps.settings.image_timeout_s,
        options=options,
    )
    data = result.image.data
    info = sniff(data, result.image.media_type)
    thumb = make_thumbnail(data)
    verdict = await check_image_detail(
        deps.vision, thumb, prompt, timeout=deps.settings.llm_timeout_s
    )
    if verdict.verdict is Safety.UNSAFE:  # only a positive finding withholds an image
        return None, result.image.cost_usd, "unsafe"

    dest.parent.mkdir(parents=True, exist_ok=True)
    path = dest.with_suffix(f".{info.ext}")
    path.write_bytes(data)
    return data, result.image.cost_usd, path.name


def side_by_side(results: list[ArmResult], folder: Path, user_prompt: str) -> Path | None:
    """One picture: a column per image, labelled, with the prompt that made it underneath."""
    shown = [r for r in results if r.file and (folder / r.file).exists()]
    if not shown:
        return None
    cell, pad, head, foot = 512, 12, 44, 150
    sheet = Image.new("RGB", (len(shown) * (cell + pad) + pad, head + cell + foot), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), f"Prompt: {user_prompt}", fill="black")
    for i, r in enumerate(shown):
        x = pad + i * (cell + pad)
        with Image.open(folder / r.file) as im:
            sheet.paste(im.convert("RGB").resize((cell, cell)), (x, head))
        score = ""
        if r.scores:
            score = f"  aesthetic {r.scores.get('aesthetic')} realism {r.scores.get('realism')}"
        draw.text((x, head + cell + 4), f"{r.model_id} / {r.arm}{score}", fill="black")
        y = head + cell + 22
        for line in textwrap.wrap(r.prompt_sent, 78)[:8]:
            draw.text((x, y), line, fill=(60, 60, 60))
            y += 14
    out = folder / "side_by_side.png"
    sheet.save(out)
    return out


async def run_ab(
    deps: CreateDeps,
    prompt: str,
    cards: list[ModelCard],
    *,
    out_dir: Path,
    arms: tuple[str, ...] = ("pipeline", "raw"),
    plan: Plan = Plan.PRO,
    ratio: str = "1:1",
    say: Any = print,
) -> list[ArmResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(deps.enhancer, ev.CachingLLM):
        deps.enhancer = ev.CachingLLM(deps.enhancer)  # one ShotPlan, shared by every model
    results: list[ArmResult] = []
    for card in cards:
        pipeline_prompt = ""
        if "pipeline" in arms or "final_raw" in arms:
            cell = ev.Cell("ab", "general", "en", prompt, card.id)
            row = await ev.run_cell(deps, cell, images_dir=out_dir)
            pipeline_prompt = row.get("final_prompt") or ""
            res = ArmResult(card.id, "pipeline", pipeline_prompt, status=row["status"])
            res.cost_usd = row.get("cost_usd")
            res.look = row.get("look")
            if row.get("image"):
                src = out_dir.parent / row["image"]
                res.file = f"{card.id}__pipeline{src.suffix}"
                src.replace(out_dir / res.file)
            trace = deps.tracer.trace_for(f"eval-ab-{card.id}")
            if trace is not None:
                (out_dir / f"{card.id}__pipeline.trace.json").write_text(
                    json.dumps(trace.snapshot(row["status"]), indent=1, ensure_ascii=False), "utf-8"
                )
            say(f"{card.id} pipeline: {res.status}")
            if "pipeline" in arms:
                results.append(res)
        for arm, text in (("raw", prompt), ("final_raw", pipeline_prompt)):
            if arm not in arms or not text:
                continue
            res = ArmResult(card.id, arm, text)
            try:
                data, res.cost_usd, name = await _raw_image(
                    deps, card, text, plan=plan, ratio=ratio, dest=out_dir / f"{card.id}__{arm}"
                )
                res.file, res.status = (name, "ok") if data else ("", name)
            except Exception as exc:  # noqa: BLE001 - one arm failing never hides the others
                res.status = type(exc).__name__
            say(f"{card.id} {arm}: {res.status}")
            results.append(res)
    await _score_together(deps, prompt, results, out_dir)
    return results


async def _score_together(
    deps: CreateDeps, prompt: str, results: list[ArmResult], folder: Path
) -> None:
    """One critic call over every image, so aesthetic and realism are relative to each other."""
    shown = [r for r in results if r.file and (folder / r.file).exists()]
    if not shown:
        return
    thumbs = [
        (critic_worker.image_id(i), make_thumbnail((folder / r.file).read_bytes()))
        for i, r in enumerate(shown)
    ]
    try:
        review = await critic_worker.review(
            deps.critic,
            system=deps.prompts.get("critic").text,
            brief={"prompt": prompt, "must_have": []},
            thumbs=thumbs,
            timeout=deps.settings.llm_timeout_s,
            user_prompt=prompt,
        )
    except Exception:  # noqa: BLE001 - the images are still saved without scores
        return
    for (iid, _), r in zip(thumbs, shown, strict=True):
        v = review.verdicts.get(iid)
        if v is not None:
            r.scores = {
                "aesthetic": v.aesthetic,
                "realism": v.realism,
                "match": v.match,
                "craft": v.craft,
                "flags": list(v.flags),
            }


def write_report(results: list[ArmResult], folder: Path, prompt: str) -> Path:
    side_by_side(results, folder, prompt)
    data = [r.__dict__ for r in results]
    (folder / "report.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), "utf-8")
    lines = [
        f"# A/B: {prompt}",
        "",
        "| model | arm | status | cost | aesthetic | realism | look |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r.scores or {}
        lines.append(
            f"| {r.model_id} | {r.arm} | {r.status} | {r.cost_usd} | {s.get('aesthetic')} | "
            f"{s.get('realism')} | {r.look or ''} |"
        )
    lines += ["", "## Prompts sent", ""]
    lines += [f"- **{r.model_id} / {r.arm}**: {r.prompt_sent}" for r in results]
    (folder / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    return folder / "report.md"


__all__ = ["ARMS", "ArmResult", "ImageSettings", "estimate_usd", "run_ab", "write_report"]
