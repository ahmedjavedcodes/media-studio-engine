"""Code guard after the enhancer. No LLM. Spec section 4 [4].

The enhancer's `final_prompt` is LLM output, so it is untrusted. The guard trims it to the model's
word limit, removes weak words, runs the blocklist again and builds the API parameters itself.
Only the guarded TEXT becomes the `prompt` field. The model never writes size, seed or steps.

COMPILE_MODE=code skips the LLM prompt and builds it from the brief with the family templates, so
a benchmark gives every model the exact same prompt.
"""

import re
from dataclasses import dataclass

from app.agents.image_grand.workers.brief import Brief
from app.core.errors import ContentRefused
from app.gateway.registry import ModelCard
from app.guardrails.blocklists import is_blocked
from app.tools.sizes import dimensions

# Weak words make a model produce flat, vague images (spec section 4 [4]). Longest first.
_WEAK = (
    "soft diffused",
    "hints of",
    "hint of",
    "subtle",
    "muted",
    "gentle",
    "serene",
    "evoking",
    "atmospheric",
    "atmosphere",
    "minimal",
    "balanced",
    "fading",
    "pristine",
)
# Anything that looks like an API parameter inside the text: `--ar 16:9`, `--seed 5`, `width=512`.
_FLAG_RE = re.compile(
    r"(--[a-z][\w-]*(\s+\S+)?|\b(width|height|seed|steps|size|resolution|aspect[_ ]ratio|"
    r"guidance|num_steps)\s*[=:]\s*\S+)",
    re.IGNORECASE,
)
_NEGATIVE = "blurry, low quality, distorted hands, extra fingers, watermark, cropped"


@dataclass(frozen=True)
class CompiledPrompt:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    aspect_ratio: str
    resolution: str


def _weak_re(keep: frozenset[str]) -> re.Pattern[str]:
    words = [w for w in _WEAK if w not in keep]
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def user_weak_words(*texts: str) -> frozenset[str]:
    """Weak words the USER wrote ("minimal flat logo"). They asked for them, so they stay."""
    blob = " ".join(texts).lower()
    return frozenset(w for w in _WEAK if re.search(rf"\b{re.escape(w)}\b", blob))


def strip_weak_words(text: str, keep: frozenset[str] = frozenset()) -> str:
    text = _FLAG_RE.sub("", text)
    text = _weak_re(keep).sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r"([,.])(\s*[,.])+", r"\1", text)
    return text.strip(" ,.")


def trim_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",.")


def _parts(brief: Brief, keep: frozenset[str]) -> list[str]:
    scene = " ".join(p for p in (brief.subject, brief.action) if p)
    parts = [scene, brief.setting, *brief.details, brief.lighting, brief.camera, brief.style]
    return [p for p in (strip_weak_words(x, keep) for x in parts if x) if p]


def template_prompt(card: ModelCard, brief: Brief, user_prompt: str = "") -> str:
    """COMPILE_MODE=code: the prompt built from the brief. Same brief, same bytes."""
    parts = _parts(brief, user_weak_words(user_prompt))
    if not parts:
        return ""
    if card.prompt_style == "sentences":
        return " ".join(p[0].upper() + p[1:].rstrip(".") + "." for p in parts)
    return ", ".join(parts)


def guard_prompt(
    card: ModelCard,
    text: str,
    *,
    aspect_ratio: str,
    resolution: str = "1K",
    user_prompt: str = "",
) -> CompiledPrompt:
    """Trim, clean and check `text`, then add the size. Raises ContentRefused on a blocklist hit.
    Weak words that appear in `user_prompt` are kept, because the user asked for them."""
    keep = user_weak_words(user_prompt)
    cleaned = trim_words(strip_weak_words(text, keep), card.prompt_max_words)
    if not cleaned:
        raise ContentRefused("This request can't be turned into an image prompt.")
    if is_blocked(cleaned):  # the text is LLM output, so it is untrusted
        raise ContentRefused("This request can't be generated.")
    width, height = dimensions(aspect_ratio, resolution, card.size_multiple)
    negative = _NEGATIVE if card.supports_negative_prompt or card.family == "sd" else ""
    return CompiledPrompt(
        prompt=cleaned,
        negative_prompt=negative,  # from code only, never from the model
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )


def build_prompt(
    card: ModelCard,
    brief: Brief,
    final_prompt: str | None,
    *,
    mode: str,
    aspect_ratio: str,
    resolution: str = "1K",
    user_prompt: str = "",
) -> CompiledPrompt:
    """`llm` uses the enhancer's final_prompt (the brief template if it sent none). `code` always
    uses the brief template."""
    text = (
        template_prompt(card, brief, user_prompt)
        if mode == "code" or not final_prompt
        else final_prompt
    )
    return guard_prompt(
        card, text, aspect_ratio=aspect_ratio, resolution=resolution, user_prompt=user_prompt
    )
