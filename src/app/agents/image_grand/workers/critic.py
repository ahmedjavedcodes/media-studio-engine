"""Critic: one vision call sees every thumbnail. It only ranks and recommends. Spec v1.3 [6].

No tools, no buttons, no actions, and it never triggers a regeneration. Scores stay inside the
backend: they rank the images and feed tests, and are never sent to the frontend.
"""

import base64
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.contracts.ports import LLMPort
from app.core.safe_text import clean_ui_text

TEMPERATURE = 0.1
MAX_TOKENS = 700
GOOD_MATCH = 5  # below this the best image still does not match the prompt well


class Scores(BaseModel):
    match: int = Field(ge=0, le=10)
    craft: int = Field(ge=0, le=10)
    interest: int = Field(ge=0, le=10)
    technical: int = Field(ge=0, le=10)

    @property
    def mean(self) -> float:
        return (self.match + self.craft + self.interest + self.technical) / 4


class CriticImage(BaseModel):
    id: str
    scores: Scores | None = None


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


def rank_images(out: CriticOutput, ids: list[str], user_text: str = "") -> Review:
    """Images ranked by mean score. An image the critic did not score is placed last."""
    by_id = {i.id: i for i in out.images}
    scored: list[tuple[str, float, int]] = []
    verdicts: dict[str, Verdict] = {}
    for iid in ids:
        item = by_id.get(iid)
        if item is None or item.scores is None:
            verdicts[iid] = Verdict(iid, rank=None, match=None)
        else:
            scored.append((iid, item.scores.mean, item.scores.match))
    order = sorted(scored, key=lambda s: (-s[1], -s[2], ids.index(s[0])))
    for rank, (iid, _mean, match) in enumerate(order, start=1):
        verdicts[iid] = Verdict(iid, rank=rank, match=match)
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
    )
    return rank_images(out, [i for i, _ in thumbs], user_prompt)


def recommendation(review_: Review, order: list[str]) -> str:
    """One friendly line. `order` is the image ids as the user sees them. Free and paid get it."""
    if review_.failed or review_.best is None or review_.best not in order:
        return ""  # a critic that could not run gives no recommendation
    best = review_.verdicts[review_.best]
    if best.match is not None and best.match < GOOD_MATCH:
        return "None of these match your prompt closely. Try adding more detail."
    position = order.index(review_.best) + 1
    if len(order) == 1:
        return "This image matches your prompt well."
    line = f"Image {position} matches your prompt best"
    highlight = review_.highlight.rstrip(" .")
    return f"{line}: {highlight}." if highlight else f"{line}."
