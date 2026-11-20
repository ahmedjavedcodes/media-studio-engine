"""Compiler (IMAGE-CREATE-OPTIMIZE §5.2): the Brief plus the CHOSEN image model's style and word
budget become that model's prompt. Plain code, no model call, deterministic.

It runs inside the gateway's chain walk (router.image), once per chain entry, because the
entry that serves the request is only known there: a fallback from one model to another gets
a prompt compiled for the model that actually answers.

What it does, in order:
  1. drops every clause that holds a banned word (the vocabulary image models turn into mush:
     "pristine", "subtle", "evoking", ...). A clause, not just the word, so a trailing mood
     clause ("..., evoking a sense of calm") goes whole.
  2. lays the fields out in the model's style: `sentences`, `nouns` or `tags`.
  3. trims to the model's word budget, dropping the least important fields first (anchors, then
     camera) so the subject, setting and light always survive.
  4. drops negatives ("no X", "without X"): describe what is present. A safety net under the
     Brief's own rule.
  5. re-checks the WHOLE assembled prompt for banned vocabulary and negatives, because a phrase
     can be built across two fields that were each clean ("a soft" + "diffused lamp").
  6. reports `issues` when the Brief lacks what a good prompt needs (a concrete subject, a light
     direction, at least two detail anchors). It never invents them.

The compiled text is then re-checked by the step 0 code checks (guardrails.check_prompt), the
same as the user's own words.

A field the Brief left empty is simply omitted (e.g. `camera` for an illustration or a logo), so
the Brief and the Compiler together keep photographic terms out of non-photographic prompts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

PromptStyle = Literal["nouns", "sentences", "tags"]
# Which part opens the prompt. Most models want the subject first; GPT Image 2 weighs the start
# most and does best with background -> subject -> details.
Lead = Literal["subject", "background"]
DEFAULT_LEAD: Lead = "subject"

BANNED_WORDS = (
    "pristine",
    "subtle",
    "subtly",
    "hints of",
    "hint of",
    "fading",
    "muted",
    "soft diffused",
    "gentle",
    "minimal",
    "evoking",
    "atmosphere",
    "balanced",
    "serene",
)
_BANNED = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in BANNED_WORDS) + r")\b", re.IGNORECASE
)

# A clause that is a negative ("no dogs", "without leaves") or ends in one ("a boy with no shoes"):
# image models draw what they are told not to. The start of the clause goes whole; a tail is cut.
_NEGATIVE_CLAUSE = re.compile(r"^(?:no|without)\s", re.IGNORECASE)
_NEGATIVE_TAIL = re.compile(r"\s+(?:with no|without|no)\s.*$", re.IGNORECASE)

MIN_ANCHORS = 2
# Used when nothing else says: the registry has a style and budget for every paid model, and
# these are the free path's (Cloudflare FLUX), which have no registry entry.
DEFAULT_STYLE: PromptStyle = "sentences"
DEFAULT_MAX_WORDS = 60
_CLOUDFLARE: dict[str, tuple[PromptStyle, int]] = {
    "@cf/black-forest-labs/flux-1-schnell": ("sentences", 60),
    "@cf/black-forest-labs/flux-2-klein-4b": ("sentences", 60),
    "@cf/black-forest-labs/flux-2-klein-9b": ("sentences", 60),
}


@dataclass(frozen=True)
class Compiled:
    text: str
    issues: list[str] = field(default_factory=list)
    words: int = 0


def style_for_cloudflare(model: str) -> tuple[PromptStyle, int]:
    return _CLOUDFLARE.get(model, (DEFAULT_STYLE, DEFAULT_MAX_WORDS))


def _clean(value: object) -> str:
    """One field: banned and negative clauses removed, spacing tidied. '' when nothing is left."""
    if not isinstance(value, str):
        return ""
    clauses = [c.strip(" .;") for c in re.split(r"[,;]", value)]
    kept = []
    for clause in clauses:
        if not clause or _BANNED.search(clause) or _NEGATIVE_CLAUSE.match(clause):
            continue
        tail_cut = _NEGATIVE_TAIL.sub("", clause).strip()
        if tail_cut:
            kept.append(tail_cut)
    return ", ".join(kept)


def _anchors(brief: Mapping[str, Any]) -> list[str]:
    raw = brief.get("detail_anchors")
    items = raw if isinstance(raw, list) else []
    cleaned = (_clean(a) for a in items)
    return list(dict.fromkeys(a for a in cleaned if a))


def _words(text: str) -> int:
    return len(text.split())


def _layout(
    parts: dict[str, str], anchors: list[str], style: PromptStyle, lead: Lead = DEFAULT_LEAD
) -> str:
    subject, action = parts["subject"], parts["action"]
    setting, lighting, camera = parts["setting"], parts["lighting"], parts["camera"]
    who = " ".join(p for p in (subject, action) if p)
    if style == "sentences":
        if lead == "background":
            opening = [setting, who]
        else:
            opening = [who + (f", {setting}" if setting else "")]
        sentences = [
            *opening,
            f"Lit by {lighting}" if lighting else "",
            camera[:1].upper() + camera[1:] if camera else "",
            ", ".join(anchors),
        ]
        text = ". ".join(s for s in sentences if s).strip() + "."
        return text[:1].upper() + text[1:]
    first = [setting, who] if lead == "background" else [who, setting]
    pieces = [*first, lighting, camera, *anchors]
    joined = ", ".join(p for p in pieces if p)
    return joined.lower() if style == "tags" else joined


def _recheck(text: str, style: PromptStyle) -> tuple[str, bool]:
    """The whole prompt, once more: a clause that now holds banned vocabulary or a negative is
    dropped (it can be assembled across two clean fields). Returns (text, changed). Untouched
    text is returned as is, so the common case costs one search."""
    if not (_BANNED.search(text) or _NEGATIVE_TAIL.search(text) or _NEGATIVE_CLAUSE.search(text)):
        return text, False
    clauses = [c.strip() for c in re.split(r"[,.;]\s*", text) if c.strip()]
    kept = [
        _NEGATIVE_TAIL.sub("", c).strip()
        for c in clauses
        if not (_BANNED.search(c) or _NEGATIVE_CLAUSE.match(c))
    ]
    cleaned = ", ".join(c for c in kept if c)
    if style == "sentences" and cleaned:
        cleaned = cleaned[:1].upper() + cleaned[1:] + "."
    return cleaned, True


def compile_prompt(
    brief: Mapping[str, Any],
    style: PromptStyle,
    max_words: int,
    *,
    fallback: str = "",
    lead: Lead = DEFAULT_LEAD,
) -> Compiled:
    """Compile `brief` for a model that wants `style` and at most `max_words` words, opening
    with `lead`. `fallback` (the user's own words) stands in only when the Brief has no usable
    subject."""
    parts = {
        key: _clean(brief.get(key))
        for key in ("subject", "action", "setting", "lighting", "camera")
    }
    anchors = _anchors(brief)
    issues: list[str] = []
    if not parts["subject"]:
        issues.append("no concrete subject")
        parts["subject"] = _clean(fallback)
    if not parts["lighting"]:
        issues.append("no light direction")
    if len(anchors) < MIN_ANCHORS:
        issues.append(f"{len(anchors)} detail anchor(s), need {MIN_ANCHORS}")

    text = _layout(parts, anchors, style, lead)
    while _words(text) > max_words and anchors:  # least important first
        anchors.pop()
        text = _layout(parts, anchors, style, lead)
    if _words(text) > max_words and parts["camera"]:
        parts["camera"] = ""
        text = _layout(parts, anchors, style, lead)
    if _words(text) > max_words:
        text = " ".join(text.split()[:max_words])
        issues.append("trimmed mid-sentence to fit the word budget")
    text, changed = _recheck(text, style)
    if changed:
        issues.append("final check removed banned or negative wording")
    return Compiled(text=text, issues=issues, words=_words(text))
