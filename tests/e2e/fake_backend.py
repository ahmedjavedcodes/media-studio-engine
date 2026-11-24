"""The real FastAPI app and run service with FAKE providers, for the browser smoke test.

    .venv/bin/python -m uvicorn tests.e2e.fake_backend:app --port 8001

Nothing here calls Cloudflare, TypeSafe or any paid service. The guardrail, enhancer, critic and
safety check are fakes, and images are plain coloured JPEGs. The run service, the graph, the
in-memory checkpointer, the signed tokens and every HTTP endpoint are the real code.

Scenarios (by prompt text):
- "make it nice"     the enhancer asks one question, then continues after the answer
- anything else      goes straight to the images
Questions expire after DRAFT_TIMEOUT_S seconds (default 3) so the expiry flow can be tested.
"""

import io
import json
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from app.agents.grand.graph import build_grand_graph
from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.workers.brief import Brief, EnhancerOutput, Option, Question
from app.agents.image_grand.workers.critic import CriticImage, CriticOutput, Scores
from app.api import run_service
from app.contracts.ports import GeneratedImage, VisionRaw
from app.core.config import Settings
from app.core.prompts import PromptStore
from app.core.tokens import TokenSigner
from app.core.tracing import Tracer
from app.gateway.image_gateway import ImageGateway
from app.gateway.registry import load_registry
from app.guardrails.deps import get_guard_deps
from app.local.fake_credits import FakeCredits
from app.local.output_store import OutputStore
from app.main import app
from tests.unit.guard_helpers import make_deps as make_guard_deps

COLORS = ["#e67e22", "#2980b9", "#27ae60", "#8e44ad", "#c0392b"]


class FakeEnhancer:
    model = "fake-enhancer"

    async def complete_json(self, **kw: Any) -> Any:
        payload = json.loads(kw["user"].split("<user_prompt>")[1].split("</user_prompt>")[0])
        prompt: str = payload["prompt"]
        if "make it nice" in prompt.lower() and not payload["answers"]:
            return EnhancerOutput(
                status="missing",
                questions=[
                    Question(
                        id="q1",
                        text="What should the image show?",
                        options=[
                            Option(id="a", label="A product on a table", recommended=True),
                            Option(id="b", label="A person in a street"),
                            Option(id="c", label="A landscape"),
                            Option(id="other", label="Something else"),
                        ],
                    )
                ],
            )
        subject = " ".join([prompt, *payload["answers"].values()]).strip()
        return EnhancerOutput(
            status="ready",
            brief=Brief(subject=subject, style="product photography", must_have=[subject]),
            final_prompt=f"{subject}, studio light, sharp focus",
        )


class FakeCritic:
    model = "fake-critic"

    async def complete_json(self, **kw: Any) -> Any:
        ids = json.loads(kw["user"].split("<request>")[1].split("</request>")[0])["images"]
        images = [
            CriticImage(id=i, scores=Scores(match=9 - n, craft=8, interest=7, technical=8))
            for n, i in enumerate(ids)
        ]
        return CriticOutput(images=images, best=ids[0], highlight="sharp and on prompt")


class FakeSafety:
    model = "fake-safety"

    async def classify(self, **_: Any) -> VisionRaw:
        return VisionRaw("safe", [("safe", -0.01), ("unsafe", -5.0)])


class FakeImages:
    name = "cloudflare"
    count = 0

    async def generate(self, **kw: Any) -> GeneratedImage:
        import asyncio

        await asyncio.sleep(0.4)  # long enough to see the progress
        FakeImages.count += 1
        buf = io.BytesIO()
        size = (kw["width"], kw["height"])
        Image.new("RGB", size, COLORS[FakeImages.count % len(COLORS)]).save(buf, format="JPEG")
        return GeneratedImage(data=buf.getvalue(), cost_usd=0.0)


def build_fake_deps() -> CreateDeps:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        draft_timeout_s=int(os.environ.get("DRAFT_TIMEOUT_S", "3")),
        local_output_dir=tempfile.mkdtemp(prefix="e2e-outputs-"),
        langsmith_tracing=False,
    )
    return CreateDeps(
        registry=load_registry(),
        gateway=ImageGateway({"cloudflare": FakeImages()}, ["cloudflare"]),
        enhancer=FakeEnhancer(),  # type: ignore[arg-type]
        critic=FakeCritic(),  # type: ignore[arg-type]
        vision=FakeSafety(),  # type: ignore[arg-type]
        credits=FakeCredits(1000),
        store=OutputStore(Path(settings.local_output_dir)),
        prompts=PromptStore(source="local"),
        tracer=Tracer(client=None, project="e2e", enabled=False),
        tokens=TokenSigner("e2e-secret"),
        settings=settings,
        enabled_providers=["cloudflare"],
    )


@asynccontextmanager
async def fake_lifespan(a: FastAPI) -> AsyncIterator[None]:
    deps = build_fake_deps()
    a.state.create_deps = deps
    a.state.run_service = run_service.RunService(build_grand_graph(deps, InMemorySaver()), deps)
    yield


app.router.lifespan_context = fake_lifespan
app.dependency_overrides[get_guard_deps] = lambda: _GUARD
_GUARD = make_guard_deps()
