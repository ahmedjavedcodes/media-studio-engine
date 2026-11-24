"""Events streamed over SSE. Spec section 5: the pipeline is never exposed.

Only friendly status words go out. No model names, agent names, prompts or scores.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

StatusName = Literal["understanding", "thinking", "need_detail", "compiling", "creating", "final"]

# Internal step -> the words the user sees.
STATUS_LABELS: dict[str, str] = {
    "understanding": "Understanding your request",
    "thinking": "Thinking",
    "need_detail": "Need a quick detail",
    "compiling": "Compiling",
    "creating": "Creating your image",
    "final": "Final touches",
}


class ImageOut(BaseModel):
    id: str
    url: str
    thumb_url: str
    width: int
    height: int


class OptionOut(BaseModel):
    id: str
    label: str
    recommended: bool = False


class QuestionOut(BaseModel):
    id: str
    text: str
    options: list[OptionOut]


class RunEvent(BaseModel):
    """One SSE event. `type` is the SSE event name."""

    # `generated` is internal: one image finished. The driver turns it into a progress status.
    type: Literal["status", "generated", "question", "ready", "error", "expired"]
    run_id: str
    status: str | None = None
    label: str | None = None
    progress: str | None = None  # "1/2"
    images: list[ImageOut] = Field(default_factory=list)
    recommendation: str | None = None
    questions: list[QuestionOut] = Field(default_factory=list)
    code: str | None = None
    message: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def status_event(run_id: str, status: StatusName, progress: str | None = None) -> RunEvent:
    return RunEvent(
        type="status",
        run_id=run_id,
        status=status,
        label=STATUS_LABELS[status],
        progress=progress,
    )
