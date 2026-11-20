"""Step 0: code checks on a request, before ANY model call (IMAGE-CREATE-OPTIMIZE §5.1).

Plain code: no model, no network, no cost. A refused request never reaches the Gate, so it
spends nothing, not even a free Groq call. The rules live in `rules.yaml` (config, not code).

What is checked today (the owner chose a MINIMAL list, 2026-11-20):
    - prompt length and characters (control characters other than newline and tab)
    - prompt-injection phrases (a short, specific regex list)
    - banned terms and protected (celebrity / brand) names: the lists exist but are EMPTY
Not built yet: upload checks (create takes no uploads) and the request cache (it needs the
document store, §4.3).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

RULES_PATH = Path(__file__).with_name("rules.yaml")

Rule = Literal["empty", "too_long", "control_chars", "injection", "banned_term", "protected_name"]


class Rules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_prompt_chars: int = Field(gt=0)
    injection_patterns: list[str] = Field(default_factory=list)
    banned_terms: list[str] = Field(default_factory=list)
    protected_names: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CheckResult:
    """`ok` False: refuse the request. `rule` names which check, `reason` is safe to log and
    never echoes the matched text back (it could be an attack string)."""

    ok: bool
    rule: Rule | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _Compiled:
    rules: Rules
    injection: list[re.Pattern[str]]
    banned: re.Pattern[str] | None
    protected: re.Pattern[str] | None


def _words(terms: list[str]) -> re.Pattern[str] | None:
    """One case-insensitive whole-word pattern for a term list; None when the list is empty."""
    cleaned = [re.escape(t.strip()) for t in terms if t.strip()]
    if not cleaned:
        return None
    return re.compile(r"\b(?:" + "|".join(cleaned) + r")\b", re.IGNORECASE)


def compile_rules(rules: Rules) -> _Compiled:
    return _Compiled(
        rules=rules,
        injection=[re.compile(p, re.IGNORECASE) for p in rules.injection_patterns],
        banned=_words(rules.banned_terms),
        protected=_words(rules.protected_names),
    )


@lru_cache(maxsize=1)
def _default() -> _Compiled:
    raw = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    return compile_rules(Rules.model_validate(raw))


def _normalise(text: str) -> str:
    """For matching only (the prompt itself is never changed). Three tricks are undone:
    - invisible FORMAT characters (Unicode Cf: zero-width space and joiners, bidi overrides,
      soft hyphen) are removed, so "pre\u200bvious" reads as "previous";
    - NFKC folds look-alike characters (full-width letters, ligatures) onto plain ones;
    - runs of whitespace collapse to one space."""
    visible = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return " ".join(unicodedata.normalize("NFKC", visible).split())


def check_prompt(prompt: str, compiled: _Compiled | None = None) -> CheckResult:
    c = compiled or _default()
    if not prompt.strip():
        return CheckResult(False, "empty", "no prompt given")
    if len(prompt) > c.rules.max_prompt_chars:
        return CheckResult(
            False, "too_long", f"prompt is longer than {c.rules.max_prompt_chars} characters"
        )
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t\r" for ch in prompt):
        return CheckResult(False, "control_chars", "prompt contains control characters")
    text = _normalise(prompt)
    if any(p.search(text) for p in c.injection):
        return CheckResult(False, "injection", "prompt contains an instruction aimed at the system")
    if c.banned is not None and c.banned.search(text):
        return CheckResult(False, "banned_term", "prompt contains a banned term")
    if c.protected is not None and c.protected.search(text):
        return CheckResult(False, "protected_name", "prompt names a protected person or brand")
    return CheckResult(True)


__all__ = ["CheckResult", "Rules", "check_prompt", "compile_rules"]
