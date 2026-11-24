"""What goes into the engine. Spec section 3: the prompt and the settings stay separate."""

from pydantic import BaseModel, Field

from app.core.plans import Plan


class ImageSettings(BaseModel):
    model: str = "auto"  # `auto` or a model id from the registry
    aspect_ratio: str = "1:1"
    resolution: str = "1K"
    n: int = Field(default=1, ge=1, le=8)  # the plan limit is checked in the code checks


class GenerationRequest(BaseModel):
    # Length and emptiness are checked by the guardrail, so the user gets a 400 `bad_input`.
    prompt: str = ""
    # Uploaded images as base64 (a `data:image/...;base64,` prefix is allowed).
    images: list[str] = Field(default_factory=list)
    plan: Plan = Plan.BASIC  # picked by hand in the UI during the build phase
    settings: ImageSettings = Field(default_factory=ImageSettings)


class ResumeRequest(BaseModel):
    """The user's answers to the questions the enhancer asked. {question id: option id or text}."""

    answers: dict[str, str] = Field(default_factory=dict)
