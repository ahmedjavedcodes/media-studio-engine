"""Critic: one vision call sees every thumbnail. It only ranks and recommends. Spec v1.3 [6].

No tools, no buttons, no actions, and it never triggers a regeneration. Scores stay inside the
backend: they rank the images and feed tests, and are never sent to the frontend.
"""

import base64
import json
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

from app.contracts.ports import LLMPort
from app.core.safe_text import clean_ui_text

TEMPERATURE = 0.1
MAX_TOKENS = 700
GOOD_MATCH = 5  # below this the best image still does not match the prompt well

# What the critic can flag on one image. The CODE turns a flag into a score cap, so the strict
# rules hold even when the model forgets to lower a number. Anything else it writes is ignored.
FLAG_UNREQUESTED_TEXT = "unrequested_text"  # text or a logo the brief did not ask for
FLAG_MISSING_ITEM = "missing_item"  # a must_have item or the contents are not there
FLAG_BAD_ANATOMY = "bad_anatomy"  # warped hands, faces or limbs
FLAG_PLAIN_SCENE = "plain_scene"  # a plain, empty scene with flat light
FLAG_GENERIC_SCENE = "generic_scene"  # stock, sterile or empty: nothing a photographer chose
FLAG_PHYSICAL_ERROR = "physical_error"  # a double liquid line, a floating object, wrong shadows
FLAGS = (
    FLAG_UNREQUESTED_TEXT,
    FLAG_MISSING_ITEM,
    FLAG_BAD_ANATOMY,
    FLAG_PLAIN_SCENE,
    FLAG_GENERIC_SCENE,
    FLAG_PHYSICAL_ERROR,
)
HARD_FLAGS = (FLAG_UNREQUESTED_TEXT, FLAG_MISSING_ITEM, FLAG_BAD_ANATOMY)  # these break the brief
MATCH_CAP = 3  # unrequested text/logos, or a missing must_have/contents item
CRAFT_CAP = 4  # warped hands or faces
INTEREST_CAP = 6  # a plain empty scene scores 5 to 6 at most
AESTHETIC_CAP = 5  # a generic or empty scene is not beautiful, whatever else it gets right
REALISM_CAP = 4  # a physical error (floating, double liquid line, wrong shadow)
SOFT_TECHNICAL = 5  # at or below this the recommendation may mention Upscale


def _flags(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [f for f in (str(v).strip().lower() for v in value) if f in FLAGS]


class Scores(BaseModel):
    match: int = Field(ge=0, le=10)
    craft: int = Field(ge=0, le=10)
    interest: int = Field(ge=0, le=10)
    technical: int = Field(ge=0, le=10)
    # Anchored 0 to 10 (see critic.md). They feed plan retrieval and the eval.
    aesthetic: int | None = Field(default=None, ge=0, le=10)
    realism: int | None = Field(default=None, ge=0, le=10)

    @property
    def mean(self) -> float:
        parts = [self.match, self.craft, self.interest, self.technical]
        parts += [v for v in (self.aesthetic, self.realism) if v is not None]
        return sum(parts) / len(parts)


class CriticImage(BaseModel):
    id: str
    scores: Scores | None = None
    flags: Annotated[list[str], BeforeValidator(_flags)] = Field(default_factory=list)


class CriticOutput(BaseModel):
    """Only ranks and recommends. Safety is checked per image before the critic runs."""

    images: list[CriticImage]
    best: str | None = None
    highlight: str = Field(default="", max_length=140)


@dataclass(frozen=True)
class Verdict:
    """One image after review. `rank` is 1 for the best image, None if it was not scored."""

    id: str
    rank: int | None
    match: int | None
    craft: int | None = None
    technical: int | None = None
    flags: tuple[str, ...] = ()
    interest: int | None = None  # with `mean`, kept for the eval harness. Never sent to the user
    mean: float | None = None
    aesthetic: int | None = None
    realism: int | None = None


@dataclass(frozen=True)
class Review:
    verdicts: dict[str, Verdict]
    best: str | None
    highlight: str
    failed: bool = False  # the critic could not run. Images go out in generation order


def image_id(index: int) -> str:
    return f"img_{index + 1}"


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def apply_caps(item: CriticImage) -> CriticImage:
    """The strict rules, in code. match is at most 3 for unrequested text or a missing item, craft
    at most 4 for bad anatomy, interest at most 6 for a plain scene. Scores only go down."""
    if item.scores is None:
        return item
    sc = item.scores.model_copy()
    if FLAG_UNREQUESTED_TEXT in item.flags or FLAG_MISSING_ITEM in item.flags:
        sc.match = min(sc.match, MATCH_CAP)
    if FLAG_BAD_ANATOMY in item.flags:
        sc.craft = min(sc.craft, CRAFT_CAP)
    if FLAG_PLAIN_SCENE in item.flags:
        sc.interest = min(sc.interest, INTEREST_CAP)
    if sc.aesthetic is not None and (
        FLAG_GENERIC_SCENE in item.flags or FLAG_PLAIN_SCENE in item.flags
    ):
        sc.aesthetic = min(sc.aesthetic, AESTHETIC_CAP)
    if sc.realism is not None and FLAG_PHYSICAL_ERROR in item.flags:
        sc.realism = min(sc.realism, REALISM_CAP)
    return item.model_copy(update={"scores": sc})


def rank_images(out: CriticOutput, ids: list[str], user_text: str = "") -> Review:
    """Images ranked by mean score (after the caps). An image the critic did not score is last."""
    by_id = {i.id: apply_caps(i) for i in out.images}
    scored: list[tuple[str, float, CriticImage]] = []
    verdicts: dict[str, Verdict] = {}
    for iid in ids:
        item = by_id.get(iid)
        if item is None or item.scores is None:
            verdicts[iid] = Verdict(iid, rank=None, match=None)
        else:
            scored.append((iid, item.scores.mean, item))

    def sort_key(entry: tuple[str, float, CriticImage]) -> tuple[bool, float, int, int]:
        iid, mean, item = entry
        # An image that breaks the brief (unrequested text, a missing item, bad anatomy) ranks
        # below every clean image, however well the rest of it scores.
        broken = bool(set(item.flags) & set(HARD_FLAGS))
        return (broken, -mean, -(item.scores.match if item.scores else 0), ids.index(iid))

    order = sorted(scored, key=sort_key)
    for rank, (iid, _mean, item) in enumerate(order, start=1):
        sc = item.scores
        assert sc is not None
        verdicts[iid] = Verdict(
            iid,
            rank=rank,
            match=sc.match,
            craft=sc.craft,
            technical=sc.technical,
            flags=tuple(item.flags),
            interest=sc.interest,
            mean=round(sc.mean, 3),
            aesthetic=sc.aesthetic,
            realism=sc.realism,
        )
    best = order[0][0] if order else None  # the code decides. The model's `best` is a hint only
    # The highlight is model text a user reads. Links, markup and model names are dropped.
    return Review(verdicts, best, clean_ui_text(out.highlight, 140, user_text) or "")


async def review(
    llm: LLMPort,
    *,
    system: str,
    brief: dict[str, object],
    thumbs: list[tuple[str, bytes]],
    timeout: float,
    user_prompt: str = "",
    usage: dict[str, Any] | None = None,
) -> Review:
    """`thumbs` is [(image id, 256 px jpeg)] in generation order."""
    request = {"brief": brief, "images": [i for i, _ in thumbs]}
    out = await llm.complete_json(
        system=system,
        user=f"<request>\n{json.dumps(request, ensure_ascii=False)}\n</request>",
        schema=CriticOutput,
        images=[_data_url(t) for _, t in thumbs],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        timeout=timeout,
        usage=usage,
    )
    return rank_images(out, [i for i, _ in thumbs], user_prompt)


def suggestion(best: Verdict) -> str:
    """One text-only tip when the best image has a flaw a later tool could fix. No button, no
    action: the critic only advises. At most one sentence."""
    if FLAG_BAD_ANATOMY in best.flags:
        return "Edit could fix the hands or face."
    if FLAG_PHYSICAL_ERROR in best.flags:
        return "Edit could fix the physical detail."
    if FLAG_UNREQUESTED_TEXT in best.flags:
        return "Edit could remove the extra text."
    if FLAG_MISSING_ITEM in best.flags:
        return "Edit could add the missing item."
    if best.technical is not None and best.technical <= SOFT_TECHNICAL:
        return "Upscale could sharpen it."
    return ""


def recommendation(review_: Review, order: list[str]) -> str:
    """One friendly line. `order` is the image ids as the user sees them. Free and paid get it."""
    if review_.failed or review_.best is None or review_.best not in order:
        return ""  # a critic that could not run gives no recommendation
    best = review_.verdicts[review_.best]
    tip = suggestion(best)
    if best.match is not None and best.match < GOOD_MATCH:
        line = "None of these match your prompt closely. Try adding more detail."
        return f"{line} {tip}" if tip else line
    position = order.index(review_.best) + 1
    if len(order) == 1:
        line = "This image matches your prompt well."
        return f"{line} {tip}" if tip else line
    line = f"Image {position} matches your prompt best"
    highlight = review_.highlight.rstrip(" .")
    line = f"{line}: {highlight}." if highlight else f"{line}."
    return f"{line} {tip}" if tip else line
