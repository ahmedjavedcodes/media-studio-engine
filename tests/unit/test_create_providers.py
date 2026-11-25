"""Cloudflare and OpenRouter image providers, the LLM parser, fallback and the spend guard."""

import base64
import json

import httpx
import pytest
import respx

from app.agents.image_grand.workers.shot_plan import ShotPlan as EnhancerOutput
from app.contracts.ports import GeneratedImage
from app.core.errors import InvalidModelOutput, ProviderCallFailed, SpendCapReached
from app.gateway.budget import SpendGuard
from app.gateway.image_gateway import ImageGateway
from app.gateway.providers.cloudflare import CloudflareClient
from app.gateway.providers.cloudflare_create import (
    CloudflareImages,
    CloudflareLLM,
    extract_text,
    parse_schema,
)
from app.gateway.providers.openrouter_images import OpenRouterImages
from app.gateway.registry import ModelCard, ProviderEntry, load_registry
from tests.unit.create_helpers import FakeProvider, ready

CF = "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/"
IMG = base64.b64encode(b"\xff\xd8fakejpeg").decode()


def cf_images() -> tuple[CloudflareImages, httpx.AsyncClient]:
    http = httpx.AsyncClient()
    return CloudflareImages(CloudflareClient(http, account_id="acc", api_token="tok"), http), http


async def call(provider: CloudflareImages, slug: str, **over):  # type: ignore[no-untyped-def]
    args = dict(
        slug=slug,
        family="flux",
        prompt="a cat",
        negative_prompt="",
        width=1024,
        height=768,
        aspect_ratio="4:3",
        seed=None,
        timeout=5,
    )
    args.update(over)
    return await provider.generate(**args)


@respx.mock
async def test_schnell_sends_json_with_steps_and_no_size() -> None:
    route = respx.post(CF + "@cf/black-forest-labs/flux-1-schnell").mock(
        return_value=httpx.Response(200, json={"success": True, "result": {"image": IMG}})
    )
    p, http = cf_images()
    out = await call(p, "@cf/black-forest-labs/flux-1-schnell")
    assert out.data == b"\xff\xd8fakejpeg"
    sent = json.loads(route.calls[0].request.content)
    assert sent == {"prompt": "a cat", "steps": 4}
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"
    await http.aclose()


@respx.mock
async def test_flux2_sends_multipart_with_explicit_size() -> None:
    route = respx.post(CF + "@cf/black-forest-labs/flux-2-klein-4b").mock(
        return_value=httpx.Response(200, json={"success": True, "result": {"image": IMG}})
    )
    p, http = cf_images()
    await call(p, "@cf/black-forest-labs/flux-2-klein-4b", seed=5)
    req = route.calls[0].request
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.content.decode()
    for field, value in (("prompt", "a cat"), ("width", "1024"), ("height", "768"), ("seed", "5")):
        assert f'name="{field}"' in body and value in body
    await http.aclose()


@respx.mock
async def test_leonardo_sends_json_with_size_seed_and_negative() -> None:
    route = respx.post(CF + "@cf/leonardo/phoenix-1.0").mock(
        return_value=httpx.Response(200, json={"success": True, "result": {"image": IMG}})
    )
    p, http = cf_images()
    await call(p, "@cf/leonardo/phoenix-1.0", seed=9, negative_prompt="blurry")
    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "prompt": "a cat",
        "width": 1024,
        "height": 768,
        "negative_prompt": "blurry",
        "seed": 9,
    }
    await http.aclose()


@respx.mock
async def test_raw_image_response_is_accepted() -> None:
    respx.post(CF + "@cf/leonardo/lucid-origin").mock(
        return_value=httpx.Response(200, content=b"RAWBYTES", headers={"content-type": "image/png"})
    )
    p, http = cf_images()
    assert (await call(p, "@cf/leonardo/lucid-origin")).data == b"RAWBYTES"
    await http.aclose()


@respx.mock
@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False), (401, False)])
async def test_status_errors_map_to_retryable_or_not(status: int, retryable: bool) -> None:
    respx.post(CF + "@cf/leonardo/lucid-origin").mock(return_value=httpx.Response(status))
    p, http = cf_images()
    with pytest.raises(ProviderCallFailed) as err:
        await call(p, "@cf/leonardo/lucid-origin")
    assert err.value.retryable is retryable
    await http.aclose()


def test_extract_text_accepts_every_documented_shape() -> None:
    assert extract_text({"choices": [{"message": {"content": "A"}}]}) == "A"
    assert extract_text({"response": "B"}) == "B"
    assert extract_text({"response": {"k": 1}}) == {"k": 1}
    assert (
        extract_text(
            {
                "output": [
                    {"type": "reasoning"},
                    {"type": "message", "content": [{"type": "output_text", "text": "C"}]},
                ]
            }
        )
        == "C"
    )
    with pytest.raises(InvalidModelOutput):
        extract_text({"nothing": 1})


def test_parse_schema_strips_fences_and_rejects_garbage() -> None:
    raw = '```json\n{"subject": "cat"}\n```'
    assert parse_schema(raw, EnhancerOutput).subject == "cat"
    with pytest.raises(InvalidModelOutput):
        parse_schema("not json", EnhancerOutput)
    with pytest.raises(InvalidModelOutput):
        parse_schema('{"props": {"a": 1}, "intent": 5, "palette": 3, "depth": []}', EnhancerOutput)


@respx.mock
async def test_llm_asks_for_json_schema_and_validates() -> None:
    answer = ready().model_dump_json()
    route = respx.post(CF + "@cf/openai/gpt-oss-20b").mock(
        return_value=httpx.Response(
            200, json={"success": True, "result": {"choices": [{"message": {"content": answer}}]}}
        )
    )
    http = httpx.AsyncClient()
    llm = CloudflareLLM(
        CloudflareClient(http, account_id="acc", api_token="tok"), "@cf/openai/gpt-oss-20b"
    )
    out = await llm.complete_json(
        system="S", user="U", schema=EnhancerOutput, max_tokens=300, temperature=0.2, timeout=5
    )
    assert out.subject == "iced coffee cup"
    sent = json.loads(route.calls[0].request.content)
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["max_tokens"] == 300 and sent["temperature"] == 0.2
    assert sent["messages"][0] == {"role": "system", "content": "S"}
    await http.aclose()


@respx.mock
async def test_llm_sends_images_as_image_url_parts() -> None:
    from app.agents.image_grand.workers.critic import CriticOutput

    route = respx.post(CF + "@cf/meta/llama-4-scout-17b-16e-instruct").mock(
        return_value=httpx.Response(
            200, json={"success": True, "result": {"response": {"images": [], "best": None}}}
        )
    )
    http = httpx.AsyncClient()
    llm = CloudflareLLM(
        CloudflareClient(http, account_id="acc", api_token="tok"),
        "@cf/meta/llama-4-scout-17b-16e-instruct",
    )
    await llm.complete_json(
        system="S",
        user="U",
        schema=CriticOutput,
        images=["data:image/jpeg;base64,AAA"],
        max_tokens=100,
        temperature=0.1,
        timeout=5,
    )
    content = json.loads(route.calls[0].request.content)["messages"][1]["content"]
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
    await http.aclose()


# ---- OpenRouter (paid, off by default) -----------------------------------------------------

OR = "https://openrouter.ai/api/v1/images"


def or_reply(cost: float = 0.07) -> httpx.Response:
    b64 = base64.b64encode(b"PNGDATA").decode()
    return httpx.Response(
        200,
        json={"data": [{"b64_json": b64, "media_type": "image/png"}], "usage": {"cost": cost}},
    )


@respx.mock
async def test_openrouter_reads_the_image_and_records_billed_cost() -> None:
    route = respx.post(OR).mock(return_value=or_reply(0.07))
    guard = SpendGuard(1.0)
    http = httpx.AsyncClient()
    p = OpenRouterImages(http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard)
    out = await p.generate(
        slug="google/x",
        family="gemini",
        prompt="a cat",
        negative_prompt="",
        width=1,
        height=1,
        aspect_ratio="4:5",
        seed=3,
        timeout=5,
    )
    assert out.data == b"PNGDATA" and out.cost_usd == 0.07
    assert guard.spent_usd == pytest.approx(0.07)
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "google/x" and body["response_format"] == "b64_json"
    assert body["aspect_ratio"] == "4:5" and body["seed"] == 3  # an explicit size
    assert "modalities" not in body and "input_references" not in body  # none attached
    await http.aclose()


@respx.mock
async def test_openrouter_stops_when_the_spend_cap_is_reached() -> None:
    route = respx.post(OR).mock(return_value=or_reply(0.07))
    guard = SpendGuard(0.05)
    http = httpx.AsyncClient()
    p = OpenRouterImages(http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard)
    args = dict(
        slug="s",
        family="gemini",
        prompt="x",
        negative_prompt="",
        width=1,
        height=1,
        aspect_ratio="1:1",
        seed=None,
        timeout=5,
    )
    await p.generate(**args)  # type: ignore[arg-type]
    with pytest.raises(SpendCapReached):
        await p.generate(**args)  # type: ignore[arg-type]
    assert route.call_count == 1  # the second call never left the machine
    await http.aclose()


# ---- gateway fallback ------------------------------------------------------------------------


def two_provider_card() -> ModelCard:
    return ModelCard(
        id="m",
        name="M",
        family="flux",
        status="live",
        supports_seed=True,
        providers=[
            ProviderEntry(name="openrouter", slug="a", cost=0.01),
            ProviderEntry(name="cloudflare", slug="b", cost=0.02),
        ],
    )


async def test_falls_back_to_the_other_provider_on_a_retryable_error() -> None:
    first, second = FakeProvider(), FakeProvider()
    first.name = "openrouter"

    async def down(**kw):  # type: ignore[no-untyped-def]
        raise ProviderCallFailed("503", retryable=True)

    first.generate = down  # type: ignore[method-assign]
    gw = ImageGateway({"openrouter": first, "cloudflare": second}, ["openrouter", "cloudflare"])
    res = await gw.generate(
        two_provider_card(),
        prompt="p",
        negative_prompt="",
        width=8,
        height=8,
        aspect_ratio="1:1",
        seed=1,
    )
    assert res.provider == "cloudflare" and res.tried == ["openrouter", "cloudflare"]


async def test_never_falls_back_on_a_refusal_or_other_4xx() -> None:
    first, second = FakeProvider(), FakeProvider()
    first.name = "openrouter"

    async def refused(**kw):  # type: ignore[no-untyped-def]
        raise ProviderCallFailed("400", retryable=False)

    first.generate = refused  # type: ignore[method-assign]
    gw = ImageGateway({"openrouter": first, "cloudflare": second}, ["openrouter", "cloudflare"])
    with pytest.raises(ProviderCallFailed):
        await gw.generate(
            two_provider_card(),
            prompt="p",
            negative_prompt="",
            width=8,
            height=8,
            aspect_ratio="1:1",
            seed=1,
        )
    assert second.calls == []


async def test_seed_is_only_sent_to_models_that_support_one() -> None:
    p = FakeProvider()
    gw = ImageGateway({"cloudflare": p}, ["cloudflare"])
    reg = load_registry()
    for model_id, expect_seed in (("phoenix-1", True), ("flux-1-schnell", False)):
        card = reg.get(model_id)
        assert card is not None
        res = await gw.generate(
            card, prompt="p", negative_prompt="", width=8, height=8, aspect_ratio="1:1", seed=42
        )
        assert res.seed_sent is expect_seed
        assert (p.calls[-1]["seed"] == 42) is expect_seed


def test_generated_image_holds_bytes_and_cost() -> None:
    assert GeneratedImage(b"x", 0.1).cost_usd == 0.1
