"""Enhancer: translate, check what is missing, write the brief and the final prompt.

One LLM call, no separate compile call. Spec section 4 [3].
"""

import json
import re

from app.agents.image_grand.workers.brief import Brief, EnhancerOutput, Option, Question
from app.contracts.ports import LLMPort
from app.core.errors import InvalidModelOutput
from app.core.safe_text import clean_ui_text
from app.gateway.registry import ModelCard

TEMPERATURE = 0.2
REASONING_EFFORT = "low"
MAX_QUESTIONS = 2
_OTHER = Option(id="other", label="Something else")

_ENGLISH_WORDS = {"a", "an", "the", "of", "with", "on", "in", "at", "and", "for", "to", "by"}
_STYLE_CUES = (
    "photo",
    "photography",
    "illustration",
    "render",
    "painting",
    "cinematic",
    "style",
    "3d",
    "watercolor",
    "sketch",
    "digital art",
    "studio",
    "lighting",
)
SKIP_MIN_WORDS = 25


EDIT_MODE = """## Mode: edit
The user attached reference images. Do NOT rewrite or re-describe the whole scene. Write
`final_prompt` as a short, direct EDIT: say only what changes and what must stay the same.
Example: "Keep the subject's face, pose and studio lighting identical; replace only the jacket
with matte black leather." Never describe what the reference images already show. Set `brief` from
the same change, and `must_have` to the things that must stay.
"""


def system_prompt(
    base: str, card: ModelCard, aspect_ratio: str, has_references: bool = False
) -> str:
    """`base` (from the Hub or the file) first so it stays cacheable. The mode, the target model
    and its family guide come at the END."""
    fixed = base.rstrip()
    mode = f"\n\n{EDIT_MODE.rstrip()}" if has_references else ""
    return (
        f"{fixed}{mode}\n\n## Target model\n"
        f"Model: {card.name}\n"
        f"Max words in final_prompt: {card.prompt_max_words}\n"
        f"Aspect ratio: {aspect_ratio}\n\n"
        f"## Prompt guide ({card.family})\n{card.prompt_guide}\n"
    )


def looks_english(prompt: str) -> bool:
    letters = [c for c in prompt if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.97:
        return False  # Urdu, Arabic and other scripts
    words = re.findall(r"[a-z']+", prompt.lower())
    return sum(w in _ENGLISH_WORDS for w in words) >= 2  # Roman Urdu has almost none


def should_skip(prompt: str) -> bool:
    """A long, English prompt that already names a style needs no enhancer. Translation still runs
    for anything else, so a non-English prompt never skips."""
    if len(prompt.split()) <= SKIP_MIN_WORDS or not looks_english(prompt):
        return False
    lower = prompt.lower()
    return any(cue in lower for cue in _STYLE_CUES)


def skipped_output(prompt: str) -> EnhancerOutput:
    """The enhancer was skipped: the user's own words are the brief and the final prompt."""
    text = prompt.strip()
    return EnhancerOutput(status="ready", brief=Brief(subject=text), final_prompt=text)


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


def check_output(out: EnhancerOutput, *, prompt: str, allow_questions: bool) -> EnhancerOutput:
    """Enforce the spec rules in code, whatever the model sent."""
    if out.status == "missing" and allow_questions:
        qs = [n for q in out.questions[:MAX_QUESTIONS] if (n := _normalize_question(q, prompt))]
        if qs:
            return out.model_copy(update={"questions": qs, "final_prompt": None})
    brief = out.brief
    if not brief.subject.strip():
        brief = brief.model_copy(update={"subject": prompt.strip()})
    final = (
        out.final_prompt or ""
    ).strip() or None  # None: the code guard builds one from the brief
    return EnhancerOutput(status="ready", brief=brief, final_prompt=final, questions=[])


async def enhance(
    llm: LLMPort,
    card: ModelCard,
    *,
    base_prompt: str,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    answers: dict[str, str],
    allow_questions: bool,
    max_tokens: int,
    timeout: float,
    has_references: bool = False,
) -> EnhancerOutput:
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "answers": answers,
        "allow_questions": allow_questions,
        "has_references": has_references,
    }
    out: EnhancerOutput | None = None
    for attempt in range(2):  # a cut-off answer is retried once with room to finish
        try:
            out = await llm.complete_json(
                system=system_prompt(base_prompt, card, aspect_ratio, has_references),
                user=f"<user_prompt>\n{json.dumps(payload, ensure_ascii=False)}\n</user_prompt>",
                schema=EnhancerOutput,
                max_tokens=max_tokens * (attempt + 1),
                temperature=TEMPERATURE,
                timeout=timeout,
                reasoning_effort=REASONING_EFFORT,
            )
            break
        except InvalidModelOutput:
            if attempt == 1:
                raise
    assert out is not None
    return check_output(out, prompt=prompt, allow_questions=allow_questions)
