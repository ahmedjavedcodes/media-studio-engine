"""Everything the txt2img graph needs, in one object. Real and fake versions plug in here."""

from dataclasses import dataclass

from app.contracts.ports import CreditPort, LLMPort, VisionPort
from app.core.config import Settings
from app.core.prompts import PromptStore
from app.core.tokens import TokenSigner
from app.core.tracing import Tracer
from app.gateway.image_gateway import ImageGateway
from app.gateway.registry import Registry
from app.local.output_store import OutputStore


@dataclass
class CreateDeps:
    registry: Registry
    gateway: ImageGateway
    enhancer: LLMPort
    critic: LLMPort
    vision: VisionPort  # the per-image safety check (the guardrail's vision model)
    credits: CreditPort
    store: OutputStore
    prompts: PromptStore
    tracer: Tracer
    tokens: TokenSigner
    settings: Settings
    enabled_providers: list[str]
