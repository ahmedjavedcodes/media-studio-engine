"""The Groq reasoning model for the enhancer. HTTP is mocked: no real Groq call."""

import httpx
import pytest
import respx
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.gateway.chains import load_chain
from app.gateway.providers.groq import GROQ_URL, GroqLLM, estimate_usd


class Answer(BaseModel):
    text: str


def reply(content: str = '{"text": "hi"}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        },
    )


async def call(llm: GroqLLM, **extra):  # type: ignore[no-untyped-def]
    return await llm.complete_json(
        system="s", user="u", schema=Answer, max_tokens=350, temperature=0.2, timeout=5, **extra
    )


@respx.mock
async def test_it_asks_for_json_and_leaves_room_for_thinking() -> None:
    route = respx.post(GROQ_URL).mock(return_value=reply())
    async with httpx.AsyncClient() as http:
        llm = GroqLLM(http, api_key="k", model="openai/gpt-oss-120b")
        usage: dict = {}
        out = await call(llm, reasoning_effort="low", usage=usage)
    assert out.text == "hi"
    sent = route.calls[0].request
    body = __import__("json").loads(sent.content)
    assert sent.headers["authorization"] == "Bearer k"
    assert body["model"] == "openai/gpt-oss-120b" and body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 350 + 1024  # reasoning tokens count against it
    assert body["include_reasoning"] is False
    assert body["response_format"]["json_schema"]["name"] == "Answer"
    assert usage["prompt_tokens"] == 1000 and usage["completion_tokens"] == 200
    assert usage["cost_usd"] == pytest.approx(estimate_usd("openai/gpt-oss-120b", 1000, 200))


@respx.mock
async def test_a_model_that_does_not_reason_gets_no_reasoning_fields() -> None:
    route = respx.post(GROQ_URL).mock(return_value=reply())
    async with httpx.AsyncClient() as http:
        await call(GroqLLM(http, api_key="k", model="some/other-model"))
    body = __import__("json").loads(route.calls[0].request.content)
    assert "reasoning_effort" not in body and body["max_completion_tokens"] == 350


@respx.mock
async def test_errors_and_bad_answers_are_mapped() -> None:
    respx.post(GROQ_URL).mock(side_effect=[httpx.Response(429), reply("not json")])
    async with httpx.AsyncClient() as http:
        llm = GroqLLM(http, api_key="k", model="openai/gpt-oss-120b")
        with pytest.raises(ProviderCallFailed) as err:
            await call(llm)
        assert err.value.retryable
        with pytest.raises(InvalidModelOutput):
            await call(llm)


def test_the_chain_moves_only_the_enhancer_and_only_when_asked() -> None:
    off = load_chain(Settings(_env_file=None, groq_llm=False))
    on = load_chain(Settings(_env_file=None, groq_llm=True))
    assert off.enhancer.provider == "cloudflare"
    assert (on.enhancer.provider, on.enhancer.model) == ("groq", "openai/gpt-oss-120b")
    assert on.critic == off.critic  # the critic stays on Cloudflare


# ---- the Cloudflare -> Groq fallback ----------------------------------------------------------

from app.contracts.ports import VisionRaw  # noqa: E402
from app.gateway.providers.fallback import FallbackLLM, FallbackVision  # noqa: E402
from app.gateway.providers.groq import GroqVision  # noqa: E402


class FailingLLM:
    provider, model = "cloudflare", "cf-model"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def complete_json(self, **kw):  # type: ignore[no-untyped-def]
        raise self.exc


class OkLLM:
    provider, model = "groq", "groq-model"

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, **kw):  # type: ignore[no-untyped-def]
        self.calls += 1
        return kw["schema"](text="from groq")


async def test_a_retryable_cloudflare_failure_asks_groq() -> None:
    backup = OkLLM()
    usage: dict = {}
    llm = FallbackLLM(FailingLLM(ProviderCallFailed("429", retryable=True)), backup)  # type: ignore[arg-type]
    out = await call(llm, usage=usage)  # type: ignore[arg-type]
    assert out.text == "from groq" and backup.calls == 1
    assert usage["fallback_to"] == "groq:groq-model"
    assert (llm.provider, llm.model) == ("groq", "groq-model")


async def test_a_bad_answer_or_a_refusal_never_falls_back() -> None:
    for exc in (InvalidModelOutput("bad"), ProviderCallFailed("400", retryable=False)):
        backup = OkLLM()
        with pytest.raises(type(exc)):
            await call(FallbackLLM(FailingLLM(exc), backup))  # type: ignore[arg-type]
        assert backup.calls == 0


class FailingVision:
    model = "cf-vision"

    async def classify(self, **kw):  # type: ignore[no-untyped-def]
        raise ProviderCallFailed("timed out", retryable=True)


@respx.mock
async def test_the_safety_check_falls_back_to_a_groq_vision_model() -> None:
    route = respx.post(GROQ_URL).mock(return_value=reply("safe"))
    async with httpx.AsyncClient() as http:
        groq = GroqVision(http, api_key="k", model="qwen/qwen3.8-27b", system_prompt="judge")
        vision = FallbackVision(FailingVision(), groq)  # type: ignore[arg-type]
        raw = await vision.classify(
            text="a fox", image_urls=["data:image/jpeg;base64,AA"], timeout=5
        )
    assert isinstance(raw, VisionRaw) and raw.text == "safe" and raw.first_token_logprobs is None
    assert vision.model == "qwen/qwen3.8-27b" and vision.provider == "groq"
    body = __import__("json").loads(route.calls[0].request.content)
    assert body["messages"][1]["content"][1]["type"] == "image_url"
    assert body["max_completion_tokens"] == 20


@respx.mock
async def test_groq_reads_images_for_the_critic() -> None:
    route = respx.post(GROQ_URL).mock(return_value=reply())
    async with httpx.AsyncClient() as http:
        await call(
            GroqLLM(http, api_key="k", model="qwen/qwen3.8-27b"),
            images=["data:image/jpeg;base64,AA"],
        )
    body = __import__("json").loads(route.calls[0].request.content)
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image")
    assert "reasoning_effort" not in body
