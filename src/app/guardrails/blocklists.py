"""A last cheap check on the final compiled prompt. Spec section 4 [4].

The real safety work is Jev and the vision check in front of every run. This list only catches
terms that must never reach an image model, even if the enhancer or a translation brought them in.
"""

import re

from app.guardrails.clean import clean_text

# Short and exact on purpose. Context-aware checks belong to Jev.
_BLOCKED = (
    "child porn",
    "cp porn",
    "loli",
    "shota",
    "nude child",
    "naked child",
    "nude minor",
    "naked minor",
    "underage nude",
    "nudify",
    "undress her",
    "undress him",
)


def is_blocked(text: str) -> bool:
    words = re.sub(r"[^\w\s]", " ", clean_text(text).prompt_clean)
    cleaned = f" {' '.join(words.split())} "
    return any(f" {term} " in cleaned for term in _BLOCKED)
