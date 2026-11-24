"""What comes out of the engine: Document, Layer, Version. Spec section 4 [7]."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Layer(BaseModel):
    kind: str = "image"
    file: str  # path relative to the run folder
    thumb: str
    mime: str
    width: int
    height: int


class Version(BaseModel):
    number: int = 1
    layers: list[Layer]  # one image layer for a create run
    model: str
    provider: str
    seed: int | None
    prompt: str  # the final compiled prompt
    negative_prompt: str = ""


class Document(BaseModel):
    id: str
    run_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    versions: list[Version]


class RunResult(BaseModel):
    """What `GET /v1/runs/{id}` returns for a finished run."""

    run_id: str
    documents: list[Document]
    recommendation: str | None = None
    credits_charged: int = 0
