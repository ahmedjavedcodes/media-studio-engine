"""What goes into the engine."""

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    # Length and emptiness are checked by the guardrail, so the user gets a 400 `bad_input`.
    prompt: str = ""
    # Uploaded images as base64 (a `data:image/...;base64,` prefix is allowed).
    images: list[str] = Field(default_factory=list)
