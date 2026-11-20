from __future__ import annotations

import base64
import json

import httpx
import respx

from media_engine.gateway.types import ImageRequest, LLMRequest

from .conftest import CF_CHAT, CF_FIRST_FAST_LLM, CF_RUN, GROQ_CHAT, cf_error, chat_ok

USER_MSG = [{"role": "user", "content": "x"}]


def _image_ok(data: bytes = b"img") -> httpx.Response:
    return httpx.Response(200, json={"result": {"image": base64.b64encode(data).decode()}})


@respx.mock
async def test_cloudflare_chat_sends_bearer_and_model(gateway):
    route = respx.post(CF_CHAT).mock(return_value=chat_ok())
    await gateway.chat(CF_FIRST_FAST_LLM, LLMRequest(messages=USER_MSG))
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer cf-test-token"
    assert json.loads(sent.content)["model"] == "@cf/openai/gpt-oss-20b"


@respx.mock
async def test_groq_gpt_oss_uses_include_reasoning_not_reasoning_format(gateway):
    respx.post(CF_CHAT).mock(return_value=cf_error(429, 3036))
    route = respx.post(GROQ_CHAT).mock(return_value=chat_ok())
    await gateway.chat("reasoning", LLMRequest(messages=USER_MSG, reasoning_effort="low"))
    body = json.loads(route.calls[0].request.content)
    assert body["reasoning_effort"] == "low"
    assert "include_reasoning" in body
    assert "reasoning_format" not in body


@respx.mock
async def test_groq_non_gpt_oss_hides_reasoning_and_never_sends_raw(gateway):
    route = respx.post(GROQ_CHAT).mock(return_value=chat_ok())
    request = LLMRequest(messages=USER_MSG, json_mode=True)
    await gateway.providers["groq"].chat("qwen/qwen3.8-27b", request)
    body = json.loads(route.calls[0].request.content)
    assert body["reasoning_format"] == "hidden"
    assert body["response_format"] == {"type": "json_object"}


@respx.mock
async def test_cloudflare_flux2_uses_multipart_with_reference_images(gateway):
    route = respx.post(f"{CF_RUN}/@cf/black-forest-labs/flux-2-klein-4b").mock(
        return_value=_image_ok()
    )
    await gateway.image("image_edit", ImageRequest(prompt="edit", input_images=[b"png-bytes"]))
    sent = route.calls[0].request
    assert sent.headers["content-type"].startswith("multipart/form-data")
    assert b"input_image_0" in sent.content


@respx.mock
async def test_cloudflare_flux1_schnell_is_json_and_caps_steps(gateway):
    route = respx.post(f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell").mock(
        return_value=_image_ok()
    )
    await gateway.image("image_create", ImageRequest(prompt="p", steps=50))
    assert json.loads(route.calls[0].request.content) == {"prompt": "p", "steps": 8}


def test_api_keys_never_appear_in_settings_repr(settings):
    text = repr(settings) + str(settings)
    assert "cf-test-token" not in text and "groq-test-key" not in text


@respx.mock
async def test_flux1_schnell_is_never_sent_a_seed(gateway):
    # A real call showed Cloudflare answers HTTP 400 / 5006 to a seed on flux-1-schnell.
    route = respx.post(f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell").mock(
        return_value=_image_ok()
    )
    await gateway.image("image_create", ImageRequest(prompt="a cat", seed=7))
    assert "seed" not in json.loads(route.calls[0].request.content)


@respx.mock
async def test_a_model_that_accepts_a_seed_still_gets_it(gateway):
    route = respx.post(f"{CF_RUN}/@cf/black-forest-labs/flux-2-klein-4b").mock(
        return_value=_image_ok()
    )
    await gateway.image("image_edit", ImageRequest(prompt="a cat", seed=7, input_images=[b"png"]))
    assert b'name="seed"' in route.calls[0].request.content


@respx.mock
async def test_a_text_only_flux2_request_is_still_multipart(gateway):
    # FLUX.2 requires multipart/form-data even with no reference image. httpx falls back to
    # application/x-www-form-urlencoded when it is given `data=` and no `files=`.
    route = respx.post(f"{CF_RUN}/@cf/black-forest-labs/flux-2-klein-4b").mock(
        return_value=_image_ok()
    )
    await gateway.image("image_edit", ImageRequest(prompt="a cat", width=512, height=512, seed=3))
    sent = route.calls[0].request
    assert sent.headers["content-type"].startswith("multipart/form-data")
    for field, value in (("prompt", b"a cat"), ("width", b"512"), ("seed", b"3")):
        assert f'name="{field}"'.encode() in sent.content and value in sent.content
