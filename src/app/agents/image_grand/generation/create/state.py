"""LangGraph state of the txt2img sub-agent."""

import operator
from typing import Annotated, Any, TypedDict


class CreateState(TypedDict, total=False):
    # input
    run_id: str
    user_id: str
    prompt: str
    plan: str
    settings: dict[str, Any]  # model, aspect_ratio, resolution, n
    has_images: bool
    # code checks
    hold_id: str
    # router and enhancer
    category: str
    model_id: str
    brief: dict[str, Any]
    final_prompt: str | None
    questions: list[dict[str, Any]]
    answers: dict[str, str]
    # code guard
    compiled: dict[str, Any]
    # generate (one entry per variation) and critic
    results: Annotated[list[dict[str, Any]], operator.add]
    review: dict[str, Any]  # critic verdicts. Backend only, never streamed
    # end
    error: dict[str, str]
