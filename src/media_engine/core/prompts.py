"""Prompt files. System prompts live in `media_engine/prompts/**.md` (inside the package, so they
ship in the wheel) and are loaded at runtime; no `.py` file may hardcode one.

Placeholders are `{{name}}`, so JSON braces in prompts need no escaping."""

from __future__ import annotations

import re
from importlib.resources import files
from importlib.resources.abc import Traversable

PROMPTS_DIR: Traversable = files("media_engine") / "prompts"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


_cache: dict[str, str] = {}


def load_prompt(path: Traversable) -> str:
    """Read a prompt. `path` is usually `PROMPTS_DIR / "<area>/<name>.md"`; any Traversable or
    `pathlib.Path` works (tests pass temp files). Cached, so the blocking read happens once per
    file per process, not on every agent call."""
    key = str(path)
    if key not in _cache:
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        _cache[key] = path.read_text(encoding="utf-8")
    return _cache[key]


def render(template: str, **values: str) -> str:
    """Fill `{{name}}` placeholders. Unknown placeholders are left as-is."""
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def tag_stripper(*tags: str) -> re.Pattern[str]:
    """Pattern matching open/close copies of `tags` (`<user_request>`, `</ user_request >`...).
    User-supplied values are wrapped in such tag blocks in the input prompt; stripping copies of
    the tags from the values means no value can close its block early and smuggle in text that
    reads as instructions."""
    names = "|".join(re.escape(t) for t in tags)
    return re.compile(rf"</?\s*(?:{names})\s*>", re.IGNORECASE)
