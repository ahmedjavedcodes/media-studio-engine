"""The Art Director: one LLM call that turns ANY prompt into a ShotPlan. Spec section 4 [3].

It follows a six-step method (intent, look, world and moment, shot, realism, ship test), picks a
look from the look library, and sees up to two good plans for similar prompts as few-shot examples.
It may ask a question only when there is no clear subject. Its rules are checked in code
(`plan_rules`) and it gets ONE corrective retry. It never writes the final prompt.
"""

import asyncio
import json
from typing import Any

from app.agents.image_grand.workers import plan_rules as rules
from app.agents.image_grand.workers.shot_plan import Option, Question, ShotPlan
from app.contracts.ports import LLMPort
from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.core.safe_text import clean_ui_text
from app.gateway.looks import Look, LookBook
from app.local.plan_store import StoredPlan

TEMPERATURE = 0.4
REASONING_EFFORT = "medium"
MAX_QUESTIONS = 2
FIRST_CALL_WAIT_S = 15.0  # when a rate limit gives no Retry-After
_OTHER = Option(id="other", label="Something else")

EDIT_MODE = """## Mode: edit
The user attached reference images. Plan only what CHANGES and what must stay the same: put the
parts that stay in `must_have`. Never re-describe what the reference images already show."""


def orientation_note(aspect_ratio: str) -> str:
    """One line that tells the director how the frame is shaped, so `composition` fits it."""
    from app.tools.sizes import parse_ratio

    w, h = parse_ratio(aspect_ratio)
    if w == h:
        return "square frame: centre the subject with an even margin"
    if h > w:
        return (
            "vertical frame: stack the scene top to bottom, leave headroom, "
            "the subject sits in the lower or middle third"
        )
    return (
        "wide frame: spread the scene sideways, the subject sits on a third, "
        "leave empty space on the other side"
    )


def _shot_text(stored: StoredPlan) -> str:
    """One good plan as a few-shot example. Compact JSON, empty fields dropped."""
    plan = ShotPlan.model_validate(stored.plan).compact()
    return f"Prompt: {stored.prompt}\n{json.dumps(plan, ensure_ascii=False, separators=(',', ':'))}"


def system_prompt(
    base: str,
    looks: LookBook,
    aspect_ratio: str,
    *,
    n: int = 1,
    fixed_look: Look | None = None,
    shots: list[StoredPlan] | None = None,
    project_style: dict[str, Any] | None = None,
    has_references: bool = False,
) -> str:
    """`base` (from the Hub or the file) first so it stays cacheable. Everything that changes per
    request (the look library, a fixed look, a project style, the examples) comes after it."""
    out = base.rstrip()
    if has_references:
        out += f"\n\n{EDIT_MODE}"
    out += f"\n\n## Look library (pick one id)\n{looks.library_text()}\n"
    if fixed_look is not None:
        out += f"\nThe user chose the look `{fixed_look.id}`. Use exactly that look.\n"
    if project_style:
        out += (
            "\n## Project style (reuse it)\n"
            f"Look `{project_style.get('look', '')}`, palette {project_style.get('palette', [])}, "
            f"props {project_style.get('props', [])}. Keep the same look, palette and kind of "
            "props unless the user's prompt says otherwise.\n"
        )
    out += f"\n## Frame\nAspect ratio {aspect_ratio}: {orientation_note(aspect_ratio)}. n = {n}.\n"
    if shots:
        out += (
            "\n## Good plans for similar prompts (copy the depth of thought, never the subject)\n"
            + "\n".join(_shot_text(s) for s in shots)
            + "\n"
        )
    return out


def _normalize_question(q: Question, user_text: str = "") -> Question | None:
    text = clean_ui_text(q.text, 160, user_text)  # model text a user will read
    if text is None:
        return None
    q = q.model_copy(update={"text": text})
    cleaned = [(o, clean_ui_text(o.label, 60, user_text)) for o in q.options if o.id != "other"]
    real = [o.model_copy(update={"label": label}) for o, label in cleaned if label][:3]
    if len(real) < 2:
        return None
    if not any(o.recommended for o in real):
        real[0] = real[0].model_copy(update={"recommended": True})
    seen = False
    fixed: list[Option] = []
    for o in real:  # one recommended only
        if o.recommended and seen:
            o = o.model_copy(update={"recommended": False})
        seen = seen or o.recommended
        fixed.append(o)
    return q.model_copy(update={"options": [*fixed, _OTHER]})


def finalize(
    plan: ShotPlan,
    *,
    prompt: str,
    allow_questions: bool,
    looks: LookBook,
    fixed_look: Look | None,
) -> ShotPlan:
    """Enforce the rules in code, whatever the model sent."""
    if plan.missing and allow_questions:
        qs = [n for q in plan.missing[:MAX_QUESTIONS] if (n := _normalize_question(q, prompt))]
        if qs:
            return plan.model_copy(update={"missing": qs})
    update: dict[str, Any] = {"missing": []}
    if not plan.subject.strip():
        update["subject"] = prompt.strip()
    if fixed_look is not None:
        update["look"] = fixed_look.id  # the user's choice overrides the model's pick
    elif plan.look not in looks.valid_ids():
        update["look"] = ""
    return rules.sanitize_plan(prompt, plan.model_copy(update=update))


async def direct(
    llm: LLMPort,
    *,
    base_prompt: str,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    answers: dict[str, str],
    allow_questions: bool,
    n: int,
    looks: LookBook,
    max_tokens: int,
    timeout: float,
    fixed_look: Look | None = None,
    shots: list[StoredPlan] | None = None,
    project_style: dict[str, Any] | None = None,
    has_references: bool = False,
    usage: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> ShotPlan:
    """At most two calls. The second one runs when the first answer was cut off, asked a style
    question, or broke a rule (invented text, people or brands, a missing field, light equipment):
    it then carries a `fix` note. If the second call fails, the first answer is used (sanitized).
    `report` receives `attempts`, `retry_reason` and the rule `violations` that were left."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "n": n,
        "answers": answers,
        "allow_questions": allow_questions,
        "has_references": has_references,
    }
    system = system_prompt(
        base_prompt,
        looks,
        aspect_ratio,
        n=n,
        fixed_look=fixed_look,
        shots=shots,
        project_style=project_style,
        has_references=has_references,
    )
    plan: ShotPlan | None = None
    left: list[rules.Violation] = []
    attempts = 0
    retry_reason: list[str] = []
    may_ask = allow_questions
    first_wait_done = False
    for attempt in range(2):
        attempts = attempt + 1
        try:
            raw = await llm.complete_json(
                system=system,
                user=f"<user_prompt>\n{json.dumps(payload, ensure_ascii=False)}\n</user_prompt>",
                schema=ShotPlan,
                max_tokens=max_tokens * (attempt + 1),  # a cut-off answer gets room to finish
                temperature=TEMPERATURE,
                timeout=timeout,
                reasoning_effort=REASONING_EFFORT,
                usage=usage,
            )
        except InvalidModelOutput:
            if attempt == 1 and plan is None:
                raise
            if attempt == 1:
                break  # keep the first answer
            continue
        except ProviderCallFailed as exc:
            if attempt == 0 and plan is None and not first_wait_done and exc.retryable:
                first_wait_done = True  # a rate limit on the FIRST call: wait once, then retry
                await asyncio.sleep(min(exc.retry_after or FIRST_CALL_WAIT_S, 60.0))
                continue
            # The first call must work. The corrective retry is a bonus: if the provider refuses
            # it (a 429 on a token-per-minute limit, a 5xx, a timeout) the first answer is used.
            if attempt == 0 or plan is None:
                raise
            break
        checked = finalize(
            raw, prompt=prompt, allow_questions=may_ask, looks=looks, fixed_look=fixed_look
        )
        if attempt == 0 and may_ask and rules.needless_question(raw):
            may_ask = False  # a style question is never shown: ask once more, questions off
            payload.update(allow_questions=False, fix=rules.NO_QUESTION_NOTE)
            retry_reason = ["needless_question"]
            continue
        found = rules.check_plan(prompt, checked, valid_looks=looks.valid_ids())
        plan, left = checked, found
        retry = rules.fixable_by_retry(found)
        if attempt == 0 and retry and not checked.missing:
            payload["fix"] = rules.retry_note(retry)  # one corrective retry
            retry_reason = sorted({v.code for v in retry})
            continue
        break
    assert plan is not None
    if report is not None:
        report.update(
            system_prompt=system,
            user_input=payload,
            attempts=attempts,
            retry_reason=retry_reason,
            violations=[{"code": v.code, "detail": v.detail} for v in left],
        )
    return plan
