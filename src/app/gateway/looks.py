"""The look library for image create. Loaded from gateway/looks.yaml.

A look is RANGES (light options, a lens range, palette moods, a depth style), not fixed text.
The Art Director reads the whole library, picks one look and fills concrete values from inside its
ranges. The user's own `preset` choice names one look and overrides the pick.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.errors import BadInput

AUTO = "auto"


class Look(BaseModel):
    id: str
    label: str
    fits: str  # when this look is the right choice
    light: list[str] = Field(default_factory=list)  # light options
    lens: str = ""  # lens range, angle and aperture
    palette: list[str] = Field(default_factory=list)  # palette moods
    depth: str = ""  # depth style
    notes: str = ""  # extra craft advice. Kept in the file, not sent in the library line

    def as_prompt_line(self) -> str:
        """One compact entry for the Art Director's system prompt."""
        return (
            f"- {self.id}: {self.fits}. Light: {'; '.join(self.light)}. Lens: {self.lens}. "
            f"Palette: {'; '.join(self.palette)}. Depth: {self.depth}."
        )


class LookBook:
    def __init__(self, looks: list[Look]) -> None:
        self._looks = {look.id: look for look in looks}

    def all(self) -> list[Look]:
        return list(self._looks.values())

    def get(self, look_id: str) -> Look | None:
        return self._looks.get(look_id)

    def valid_ids(self) -> list[str]:
        return [AUTO, *self._looks]

    def library_text(self) -> str:
        return "\n".join(look.as_prompt_line() for look in self._looks.values())

    def pick(self, requested: str) -> Look | None:
        """The look the user chose, or None for `auto` (the Art Director decides)."""
        return None if requested == AUTO else self._looks.get(requested)


def load_looks(path: Path | None = None) -> LookBook:
    src = path or Path(__file__).with_name("looks.yaml")
    data = yaml.safe_load(src.read_text("utf-8")) or {}
    raw = data.get("looks") or {}
    if not isinstance(raw, dict) or not raw:
        raise BadInput("looks.yaml: `looks` must be a non-empty map")
    return LookBook([Look.model_validate({"id": lid, **body}) for lid, body in raw.items()])


@lru_cache
def get_looks() -> LookBook:
    return load_looks()
