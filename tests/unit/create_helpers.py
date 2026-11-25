"""Fakes and builders shared by the image create tests. No real provider is ever called."""

import io
import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client
from PIL import Image
from pydantic import BaseModel

from app.agents.grand.graph import build_grand_graph
from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.workers.brief import Brief, EnhancerOutput, Option, Question
from app.agents.image_grand.workers.critic import CriticImage, CriticOutput, Scores
from app.api.run_service import RunService
from app.contracts.ports import GeneratedImage, VisionRaw
from app.core.config import Settings
from app.core.errors import ProviderCallFailed
from app.core.prompts import PromptStore
from app.core.tokens import TokenSigner
from app.core.tracing import Tracer
from app.gateway.image_gateway import ImageGateway
from app.gateway.registry import load_registry
from app.local.fake_credits import FakeCredits
from app.local.output_store import OutputStore


def jpeg(size: tuple[int, int] = (64, 64), color: str = "orange") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def png(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "blue").save(buf, format="PNG")
    return buf.getvalue()


def good_brief(subject: str = "iced coffee cup") -> Brief:
    return Brief(
        subject=subject,
        setting="marble cafe table",
        lighting="morning window light from the left",
        camera="close-up, shallow depth of field",
        style="product photography",
        must_have=[subject],
    )


def ready(final_prompt: str | None = "iced coffee cup on a marble cafe table") -> EnhancerOutput:
    return EnhancerOutput(status="ready", brief=good_brief(), final_prompt=final_prompt)


def missing() -> EnhancerOutput:
    return EnhancerOutput(
        status="missing",
        questions=[
            Question(
                id="q1",
                text="What style do you want?",
                options=[
                    Option(id="a", label="Clean product photo", recommended=True),
                    Option(id="b", label="Lifestyle scene"),
                    Option(id="c", label="Flat illustration"),
                    Option(id="other", label="Something else"),
                ],
            )
        ],
    )


class FakeLLM:
    """Answers with queued outputs. Records every call."""

    def __init__(self, model: str, outputs: list[BaseModel | Exception]) -> None:
        self.model = model
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    async def complete_json(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if kw.get("usage") is not None:  # what a real model reports. Adds up over retries
            u = kw["usage"]
            u["prompt_tokens"] = u.get("prompt_tokens", 0) + 100
            u["completion_tokens"] = u.get("completion_tokens", 0) + 50
            u["neurons"] = u.get("neurons", 0) + 10.0
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


def critic_ok(n: int, *, best: int | None = None) -> CriticOutput:
    """Scores for n images. Safety is checked per image before the critic, not here."""
    images = [
        CriticImage(
            id=f"img_{i}",
            scores=Scores(match=9 if i == best else 7, craft=8, interest=7, technical=8),
        )
        for i in range(1, n + 1)
    ]
    return CriticOutput(images=images, highlight="sharp and on prompt")


class SeqVision:
    """The per-image safety model. Answers in call order; the last answer repeats."""

    model = "fake-safety"

    def __init__(self, answers: list[str | Exception] | None = None) -> None:
        self.answers: list[str | Exception] = answers or ["safe"]
        self.calls: list[dict[str, Any]] = []

    async def classify(self, *, text: str, image_urls: list[str], timeout: float) -> Any:
        self.calls.append({"text": text, "image_urls": image_urls})
        out = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(out, Exception):
            raise out
        return VisionRaw(
            out,
            [("unsafe", -0.01), ("safe", -5.0)]
            if out != "safe"
            else [("safe", -0.01), ("unsafe", -5.0)],
            {"neurons": 2.0},  # what a real model reports, for the dev trace
        )


class FakeProvider:
    name = "cloudflare"

    def __init__(self, *, fail_index: set[int] | None = None, data: bytes | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_seeds: set[int] = set()
        self.data = data or jpeg()

    async def generate(self, **kw: Any) -> GeneratedImage:
        self.calls.append(kw)
        if kw["seed"] in self.fail_seeds:
            raise ProviderCallFailed("boom", retryable=False)
        return GeneratedImage(data=self.data, cost_usd=0.001)


class RecordingClient(Client):  # type: ignore[misc]
    """A LangSmith client that keeps runs in memory instead of calling the network."""

    def __init__(self) -> None:
        super().__init__(api_key="test", auto_batch_tracing=False)
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    def create_run(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def update_run(self, run_id: Any, **kwargs: Any) -> None:
        self.updated.append({"id": str(run_id), **kwargs})


def make_deps(
    tmp_path: Path,
    *,
    enhancer: list[BaseModel | Exception] | None = None,
    critic: list[BaseModel | Exception] | None = None,
    provider: FakeProvider | None = None,
    credits: int = 100,
    compile_mode: str = "llm",
    trace: RecordingClient | None = None,
    vision: SeqVision | None = None,
    trace_ui: bool = False,
    app_env: str = "local",
) -> CreateDeps:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        local_output_dir=str(tmp_path),
        compile_mode=compile_mode,  # type: ignore[arg-type]
        langsmith_tracing=trace is not None,
        draft_timeout_s=300,
        trace_ui=trace_ui,
        app_env=app_env,
    )
    prov = provider or FakeProvider()
    return CreateDeps(
        registry=load_registry(),
        gateway=ImageGateway({"cloudflare": prov}, ["cloudflare"]),
        enhancer=FakeLLM("fake-enhancer", enhancer or [ready()]),
        critic=FakeLLM("fake-critic", critic or [critic_ok(4)]),
        vision=vision or SeqVision(),
        credits=FakeCredits(credits),
        store=OutputStore(tmp_path),
        prompts=PromptStore(source="local"),
        tracer=Tracer(client=trace, project="test", enabled=trace is not None),
        tokens=TokenSigner("test-secret"),
        settings=settings,
        enabled_providers=["cloudflare"],
    )


def make_service(deps: CreateDeps) -> RunService:
    return RunService(build_grand_graph(deps, InMemorySaver()), deps)


def dump(events: list[Any]) -> str:
    return json.dumps([e.model_dump(mode="json", exclude_none=True) for e in events])
