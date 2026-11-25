"""Code guard after the enhancer. No LLM. Spec section 4 [4].

The enhancer's `final_prompt` is LLM output, so it is untrusted. The guard trims it to the model's
word limit, removes weak words, runs the blocklist again and builds the API parameters itself.
Only the guarded TEXT becomes the `prompt` field. The model never writes size, seed or steps.

COMPILE_MODE=code skips the LLM prompt and builds it from the brief with the family templates, so
a benchmark gives every model the exact same prompt.
"""

import re
from dataclasses import dataclass

from app.agents.image_grand.workers import plan_rules
from app.agents.image_grand.workers.prompt_writer import render_template
from app.agents.image_grand.workers.shot_plan import ShotPlan
from app.core.errors import ContentRefused
from app.gateway.registry import ModelCard
from app.guardrails.blocklists import is_blocked
from app.tools.sizes import dimensions, snap_native

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
    # Generic quality words say nothing a viewer can see (item 1: every word visible and specific)
    "highly detailed",
    "high detail",
    "beautiful",
    "stunning",
    "masterpiece",
    "breathtaking",
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
    if not plan_rules._BANNED.search(user_prompt):  # the user did not ask for these ideas
        text = plan_rules._BANNED.sub("", text)
    cleaned = trim_words(strip_weak_words(text, keep), card.prompt_max_words)
    if not cleaned:
        raise ContentRefused("This request can't be turned into an image prompt.")
    if is_blocked(cleaned):  # the text is LLM output, so it is untrusted
        raise ContentRefused("This request can't be generated.")
    # The model's own size nearest to the ratio, or about 1 megapixel on a multiple of 16 or 32
    snapped = snap_native(card.native_sizes, aspect_ratio, resolution)
    width, height = snapped or dimensions(aspect_ratio, resolution, card.size_multiple)
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
    plan: ShotPlan,
    written: str | None,
    *,
    mode: str,
    aspect_ratio: str,
    resolution: str = "1K",
    user_prompt: str = "",
    change: str = "",
) -> CompiledPrompt:
    """`llm` uses the Prompt Writer's prompt (the plan rendered by code if it sent none). `code`
    always renders the plan by code, for benchmarks. `change` is one variation axis phrase for the
    code renderer. Either way the text goes through the same guard."""
    text = render_template(card, plan, change) if mode == "code" or not written else written
    return guard_prompt(
        card, text, aspect_ratio=aspect_ratio, resolution=resolution, user_prompt=user_prompt
    )
