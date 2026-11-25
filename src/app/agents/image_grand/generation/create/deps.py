"""Everything the txt2img graph needs, in one object. Real and fake versions plug in here."""

from dataclasses import dataclass, field

from app.contracts.ports import CreditPort, LLMPort, VisionPort
from app.core.config import Settings
from app.core.prompts import PromptStore
from app.core.tokens import TokenSigner
from app.core.tracing import Tracer
from app.gateway.image_gateway import ImageGateway
from app.gateway.looks import LookBook, get_looks
from app.gateway.registry import Registry
from app.local.output_store import OutputStore
from app.local.plan_store import PlanStore


@dataclass
class CreateDeps:
    registry: Registry
    gateway: ImageGateway
    enhancer: LLMPort  # the Art Director's model
    critic: LLMPort
    vision: VisionPort  # the per-image safety check (the guardrail's vision model)
    credits: CreditPort
    store: OutputStore
    prompts: PromptStore
    tracer: Tracer
    tokens: TokenSigner
    settings: Settings
    enabled_providers: list[str]
    looks: LookBook = field(default_factory=get_looks)
    plans: PlanStore = field(default_factory=PlanStore)  # good plans + project memory
    writer: LLMPort | None = None  # the Prompt Writer's model. None = the code renders prompts
