"""Events streamed over SSE: progress, question, image, result, error."""

from typing import Literal

from pydantic import BaseModel


class RunEvent(BaseModel):
    type: Literal["progress", "question", "image", "result", "error"]
    run_id: str
    message: str = ""
