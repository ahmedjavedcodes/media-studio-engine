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
    n_refs: int  # how many reference images the user attached
    # code checks
    hold_id: str
    # router and enhancer
    category: str  # how the model was chosen: "manual" or the router rule (for the trace)
    content_category: str  # what the prompt is about: product, photo, people ... (code classifier)
    preset: str  # the resolved style preset id, or "" for none
    model_id: str
    shot_plan: dict[str, Any]  # the ShotPlan. Traced, never streamed
    subject_type: str
    prompts: list[str]  # one written prompt per image (the Prompt Writer)
    project_id: str
    questions: list[dict[str, Any]]
    answers: dict[str, str]
    # code guard
    compiled: dict[str, Any]  # the first variation (the base prompt when n is 1)
    variants: list[dict[str, Any]]  # one guarded prompt per image
    # generate (one entry per variation) and critic
    results: Annotated[list[dict[str, Any]], operator.add]
    review: dict[str, Any]  # critic verdicts. Backend only, never streamed
    # end
    error: dict[str, str]
