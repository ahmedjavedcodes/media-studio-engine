"""The structured brief and the enhancer's answer. Spec section 4 [3]."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class Brief(BaseModel):
    subject: str = ""
    action: str = ""
    setting: str = ""
    lighting: str = ""
    camera: str = ""
    style: str = ""
    details: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)


class Option(BaseModel):
    id: str
    label: str
    recommended: bool = False


class Question(BaseModel):
    id: str
    # Some models write `question` for the text. Accept both.
    text: str = Field(validation_alias=AliasChoices("text", "question"))
    options: list[Option]  # 3 choices (1 recommended) + "Something else"


class EnhancerOutput(BaseModel):
    """What the enhancer returns. Pydantic schema only."""

    status: Literal["ready", "missing"]
    brief: Brief = Field(default_factory=Brief)
    final_prompt: str | None = None  # the model-specific prompt. None when status is "missing"
    questions: list[Question] = Field(default_factory=list)
