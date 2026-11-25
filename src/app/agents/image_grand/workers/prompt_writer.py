"""The Prompt Writer: renders a ShotPlan as the final prompt(s), in the model family's style and
word budget. Spec section 4 [3b].

One LLM call returns one prompt per image. With n > 1 prompt i changes ONE variation axis and
nothing else. The family guide stays (it comes last in the system prompt). If the call fails or
breaks a rule twice, the prompts are rendered from the plan by code (`render_template`), the same
renderer COMPILE_MODE=code uses for benchmarks.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from app.agents.image_grand.workers import plan_rules as rules
from app.agents.image_grand.workers.shot_plan import ShotPlan, StrList
from app.contracts.ports import LLMPort
from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.gateway.registry import ModelCard

TEMPERATURE = 0.3
REASONING_EFFORT = "low"

EDIT_MODE = """## Mode: edit
The user attached reference images. Write each prompt as a short, direct EDIT: say only what
changes and what must stay the same. Never describe what the reference images already show."""

# What a variation axis changes in code (the template renderer). Index i picks the i-th option.
_AXIS_OPTIONS: dict[str, tuple[str, ...]] = {
    "camera angle": (
        "shot from a low angle",
        "shot from a high three-quarter angle",
        "shot from the side",
    ),
    "time of day": ("at early morning light", "at golden hour", "at blue hour"),
    "prop layout": (
        "props gathered on the left",
        "props in the foreground",
        "props behind the subject",
    ),
    "light direction": ("light from the left", "light from the right", "backlit from behind"),
}


class WrittenPrompts(BaseModel):
    prompts: StrList = Field(default_factory=list)


def system_prompt(
    base: str, card: ModelCard, aspect_ratio: str, *, n: int, has_references: bool = False
) -> str:
    low, high = rules.word_range(card.prompt_max_words)
    out = base.rstrip()
    if has_references:
        out += f"\n\n{EDIT_MODE}"
    return (
        f"{out}\n\n## Target model\nModel: {card.name}\n"
        f"Each prompt: {low} to {high} words (the limit is {card.prompt_max_words}). n = {n}.\n"
        f"Aspect ratio {aspect_ratio} (never write it in a prompt).\n\n"
        f"## Prompt guide ({card.family})\n{card.prompt_guide}\n"
    )


def axis_change(axis: str, index: int) -> str:
    """The code version of one variation: a short phrase for axis `axis`, option `index`."""
    options = _AXIS_OPTIONS.get(axis.strip().lower())
    if options:
        return options[index % len(options)]
    return f"variation {index + 1}: change the {axis.strip()}" if axis.strip() else ""


def _phrases(plan: ShotPlan) -> list[str]:
    """The plan as phrases in prompt order: subject first, text last."""
    palette = f"palette of {', '.join(plan.palette)}" if plan.palette else ""
    text = ""
    if plan.wants_text:
        where = f" {plan.text.placement}" if plan.text.placement else ""
        text = f'the text "{plan.text.exact}"{where}, no other text'
    labelled = bool(
        rules._words(f"{plan.subject} {' '.join(plan.must_have)}") & rules._LABELED_OBJECTS
    )
    plate = f"a {rules.LABEL_PLATE}" if labelled and not plan.wants_text else ""
    look = ", ".join(p for p in (plan.intent.medium, plan.look.replace("_", " ")) if p)
    return [
        p
        for p in (
            plan.subject,
            *plan.subject_details,
            plate,
            plan.world,
            plan.moment,
            plan.composition,
            plan.depth_text(),
            *plan.props,
            *plan.materials,
            palette,
            plan.light_text(),
            plan.camera_text(),
            look,
            text,
        )
        if p
    ]


def render_template(card: ModelCard, plan: ShotPlan, change: str = "") -> str:
    """The prompt built from the plan by code. Same plan, same bytes."""
    parts = _phrases(plan)
    if change:  # right after the subject, so the code guard's word limit never trims it away
        parts.insert(1 if parts else 0, change)
    if not parts:
        return ""
    if card.prompt_style == "sentences":
        return " ".join(p[0].upper() + p[1:].rstrip(".") + "." for p in parts)
    return ", ".join(parts)


def render_all(card: ModelCard, plan: ShotPlan, n: int) -> list[str]:
    """n prompts by code. With n > 1 each changes one variation axis."""
    if n <= 1:
        return [render_template(card, plan)]
    axes = plan.variation_axes or ["camera angle", "time of day", "prop layout"]
    return [render_template(card, plan, axis_change(axes[i % len(axes)], i)) for i in range(n)]


def _fit(prompts: list[str], n: int, fallback: list[str]) -> list[str]:
    """Exactly n non-empty prompts: the model's, then the code's for any that are missing."""
    good = [p.strip() for p in prompts if p.strip()][:n]
    return good + fallback[len(good) : n]


async def write(
    llm: LLMPort,
    card: ModelCard,
    *,
    base_prompt: str,
    plan: ShotPlan,
    user_prompt: str,
    aspect_ratio: str,
    n: int,
    max_tokens: int,
    timeout: float,
    has_references: bool = False,
    usage: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> list[str]:
    """n final prompts. At most two calls (the second carries a `fix`). On failure the code
    renders the prompts, so the run still goes on."""
    fallback = render_all(card, plan, n)
    axes = plan.variation_axes or ["camera angle", "time of day", "prop layout"]
    payload: dict[str, Any] = {"n": n, "axes": axes[:n] if n > 1 else []}
    system = system_prompt(base_prompt, card, aspect_ratio, n=n, has_references=has_references)
    prompts: list[str] | None = None
    left: list[rules.Violation] = []
    attempts, retry_reason, source = 0, [], "llm"
    for attempt in range(2):
        attempts = attempt + 1
        user = (
            f"<shot_plan>\n{json.dumps(plan.compact(), ensure_ascii=False)}\n</shot_plan>\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            raw = await llm.complete_json(
                system=system,
                user=user,
                schema=WrittenPrompts,
                max_tokens=max_tokens * (attempt + 1),
                temperature=TEMPERATURE,
                timeout=timeout,
                reasoning_effort=REASONING_EFFORT,
                usage=usage,
            )
        except (InvalidModelOutput, ProviderCallFailed):
            if attempt == 0:
                continue  # one more try; if it also fails the code renders the prompts
            break
        got = _fit([rules.sanitize_prompt(user_prompt, plan, p) for p in raw.prompts], n, fallback)
        found = [
            v
            for p in got
            for v in rules.check_prompt(user_prompt, plan, p, max_words=card.prompt_max_words)
        ]
        prompts, left = got, found
        retry = rules.fixable_by_retry(found, length=True)
        if attempt == 0 and retry:
            payload["fix"] = rules.retry_note(retry)
            retry_reason = sorted({v.code for v in retry})
            continue
        break
    if prompts is None:
        prompts, source = fallback, "template"
    else:  # what the model left out after its retry, the code adds
        prompts = [rules.ensure_label_plate(user_prompt, plan, p) for p in prompts]
    if report is not None:
        report.update(
            system_prompt=system,
            user_input=payload,
            attempts=attempts,
            retry_reason=retry_reason,
            source=source,
            violations=[
                {"code": c, "detail": d} for c, d in sorted({(v.code, v.detail) for v in left})
            ],
        )
    return prompts
