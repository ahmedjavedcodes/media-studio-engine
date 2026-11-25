"""The ShotPlan: the Art Director's plan for one image. Spec section 4 [3].

One Pydantic model for ANY prompt (a perfume, a dragon, a street in Lahore). Every field is
optional, so a weaker model that leaves some out still gives a usable plan. The validators accept
the loose shapes small models write (a string where a list belongs) instead of failing the run.
The plan is traced (LangSmith, the dev panel) and never sent to the client.
"""

from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, BeforeValidator, Field, field_validator

MAX_PROPS = 3
MAX_PALETTE = 3
SUBJECT_TYPES = ("product", "person", "animal", "place", "food", "fantasy", "poster", "abstract")


def _as_list(value: Any) -> list[str]:
    """A string becomes a one-item list. None and empty values become []. Items are stripped."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        return []
    return [s for s in (str(v).strip() for v in value if v is not None) if s]


StrList = Annotated[list[str], BeforeValidator(_as_list)]


def _object(first_field: str) -> Any:
    """A model wrote one sentence where an object belongs: it fills the object's first field."""

    def coerce(value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            return {first_field: value.strip()}
        return value

    return BeforeValidator(coerce)


class Intent(BaseModel):
    subject_type: str = ""  # product, person, animal, place, food, fantasy, poster, abstract
    use: str = ""  # what the image is for: an ad, a social post, a wallpaper, a book cover
    medium: str = ""  # photo, 3D render, illustration, painting, flat vector


class Depth(BaseModel):
    fg: str = ""  # foreground: what is closest to the camera
    mid: str = ""  # the middle ground: where the subject lives, sharp
    bg: str = ""  # the background


class Light(BaseModel):
    source: str = ""  # sun, window, lamp, candle, neon, overcast sky
    direction: str = ""  # where it comes from and how high
    time: str = ""  # time of day or setting
    quality: str = ""  # hard or soft, broad or narrow, with the shadows it throws
    temp: str = ""  # warm, neutral, cool, mixed


class Camera(BaseModel):
    lens: str = ""  # focal length and aperture
    angle: str = ""  # camera height and angle
    dof: str = ""  # depth of field, and where the focus sits


class TextPlan(BaseModel):
    """Text in the image. `exact` is None unless the user asked for words."""

    exact: str | None = None
    placement: str = ""
    policy: str = ""  # what every other surface shows: "no other text; labels stay blank"


class Option(BaseModel):
    id: str
    label: str
    recommended: bool = False


class Question(BaseModel):
    id: str
    # Some models write `question` for the text. Accept both.
    text: str = Field(validation_alias=AliasChoices("text", "question"))
    options: list[Option]  # 3 choices (1 recommended) + "Something else"


class ShotPlan(BaseModel):
    intent: Annotated[Intent, _object("subject_type")] = Field(default_factory=Intent)
    look: str = ""  # one id from the look library
    subject: str = ""  # the main subject, in English
    world: str = ""  # the place and context the subject lives in
    moment: str = ""  # the instant: time, state, what has just happened
    subject_details: StrList = Field(
        default_factory=list, validation_alias=AliasChoices("subject_details", "details")
    )
    props: StrList = Field(default_factory=list)  # at most 3, all from the same world
    composition: str = ""  # framing and where the subject sits for the aspect ratio
    depth: Annotated[Depth, _object("mid")] = Field(default_factory=Depth)
    light: Annotated[Light, _object("quality")] = Field(default_factory=Light)
    palette: StrList = Field(default_factory=list)  # 2 or 3 named colours
    camera: Annotated[Camera, _object("lens")] = Field(default_factory=Camera)
    materials: StrList = Field(default_factory=list)
    realism_checks: StrList = Field(default_factory=list)  # physical facts this image must obey
    text: TextPlan = Field(default_factory=TextPlan)
    variation_axes: StrList = Field(default_factory=list)  # what can change between images
    must_have: StrList = Field(default_factory=list)  # items the user asked for (the critic checks)
    missing: list[Question] = Field(default_factory=list)  # questions. Empty = nothing missing

    @field_validator("text", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            return {"exact": value.strip() or None}
        if isinstance(value, dict) and not str(value.get("exact") or "").strip():
            return {**value, "exact": None}  # "no text" written as an empty string
        return value

    @field_validator("props")
    @classmethod
    def _max_props(cls, props: list[str]) -> list[str]:
        return props[:MAX_PROPS]  # the cap is enforced here, whatever the model wrote

    @field_validator("palette")
    @classmethod
    def _max_palette(cls, palette: list[str]) -> list[str]:
        return palette[:MAX_PALETTE]

    # ---- views for prompts and rules (no LLM, no I/O) ----------------------------------------

    @property
    def wants_text(self) -> bool:
        return self.text.exact is not None

    @property
    def subject_type(self) -> str:
        return self.intent.subject_type.strip().lower()

    def light_text(self) -> str:
        parts = [self.light.source, self.light.direction, self.light.time, self.light.quality]
        text = ", ".join(p for p in parts if p)
        return f"{text}, {self.light.temp} light" if self.light.temp and text else text

    def camera_text(self) -> str:
        return ", ".join(p for p in (self.camera.lens, self.camera.angle, self.camera.dof) if p)

    def depth_text(self) -> str:
        return "; ".join(
            f"{name}: {v}"
            for name, v in (
                ("foreground", self.depth.fg),
                ("middle", self.depth.mid),
                ("background", self.depth.bg),
            )
            if v
        )

    def compact(self) -> dict[str, Any]:
        """The plan without empty fields: for few-shot examples, prompts and traces."""

        def keep(v: Any) -> bool:
            return v not in ("", None, [], {})

        def prune(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: pv for k, v in value.items() if keep(pv := prune(v))}
            return value

        out = self.model_dump(mode="json", exclude={"missing"})
        return {k: v for k, v in prune(out).items() if keep(v)}
