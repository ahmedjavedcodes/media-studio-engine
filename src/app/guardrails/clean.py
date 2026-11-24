"""Text cleaner (pure, no I/O). Spec section 7.1.

`prompt_raw` is never changed. `prompt_clean` is only for the safety checks.
"""

import re
import unicodedata
from dataclasses import dataclass

from app.core.errors import BadInput
from app.guardrails.policy import MAX_PROMPT_CHARS

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_LEET_CHARS = set("013457@$")

# Tokens that look like leetspeak but are normal prompt words: 4k, 1080p, 3d, 35mm, 16:9 ...
_KEEP = re.compile(r"^(\d+(k|p|d|mm|px|x\d+)|\d+:\d+)$")
_SPACES = re.compile(r"\s+")
_SPLIT_RUN = re.compile(r"(?<!\S)(?:[a-z0-9] ){2,}[a-z0-9](?!\S)")


@dataclass(frozen=True)
class CleanText:
    prompt_raw: str
    prompt_clean: str


def _drop_hidden(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Cs", "Co", "Cn"):  # zero-width, format, surrogate, private, unassigned
            continue
        if cat == "Cc" and ch not in "\t\n\r":  # control characters, keep whitespace for now
            continue
        out.append(ch)
    return "".join(out)


def _join_split_letters(text: str) -> str:
    """`n u d e` -> `nude`. Only runs of 3 or more single characters."""
    return _SPLIT_RUN.sub(lambda m: m.group(0).replace(" ", ""), text)


def _unleet_token(token: str) -> str:
    if _KEEP.match(token):
        return token
    has_letter = any(c.isalpha() for c in token)
    if has_letter and any(c in _LEET_CHARS for c in token):
        return token.translate(_LEET)
    return token


def clean_text(raw: str | bytes) -> CleanText:
    """Return raw and cleaned text, or raise BadInput (400)."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BadInput("Prompt is not valid UTF-8.") from exc
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise BadInput("Prompt is not valid UTF-8.") from exc
    if len(raw) > MAX_PROMPT_CHARS:
        raise BadInput(f"Prompt is longer than {MAX_PROMPT_CHARS} characters.")

    text = unicodedata.normalize("NFKC", raw)
    text = _drop_hidden(text).lower()
    text = _SPACES.sub(" ", text).strip()
    text = _join_split_letters(text)
    text = " ".join(_unleet_token(tok) for tok in text.split(" "))
    return CleanText(prompt_raw=raw, prompt_clean=text)
