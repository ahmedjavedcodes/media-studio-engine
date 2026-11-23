"""What goes into the engine."""

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
