"""The router chain. Plain code for now. Each level has the same interface, so an LLM can
replace any of them later without touching the graph.

    Grand Agent -> Image Grand Agent -> Image Generation Agent -> Image Create Agent

Routing uses what the guardrail already knows (`intent`, when Jev gives one). No LLM call runs
before the enhancer.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RouteInput:
    has_images: bool  # the user uploaded at least one image
    intent: str | None = None  # Jev's intent, when the guardrail provides it


class Router(Protocol):
    def route(self, item: RouteInput) -> str: ...


class GrandRouter:
    """Domain. Only `image` exists. Video comes later."""

    def route(self, item: RouteInput) -> str:
        return "image"


class ImageGrandRouter:
    """`generation` or `layout`. Layout comes later."""

    def route(self, item: RouteInput) -> str:
        return "generation"


class GenerationRouter:
    """`create`, `edit`, `merge` or `restyle`. Only create is built."""

    def route(self, item: RouteInput) -> str:
        return "create"


class CreateRouter:
    """Text only goes to txt2img. Any uploaded image goes to img2img (not built yet)."""

    def route(self, item: RouteInput) -> str:
        return "img2img" if item.has_images else "txt2img"


def route_request(item: RouteInput) -> str:
    """Walk the chain and return the leaf agent: `txt2img` or `img2img`."""
    if GrandRouter().route(item) != "image":
        return "unsupported"
    if ImageGrandRouter().route(item) != "generation":
        return "unsupported"
    if GenerationRouter().route(item) != "create":
        return "unsupported"
    return CreateRouter().route(item)
