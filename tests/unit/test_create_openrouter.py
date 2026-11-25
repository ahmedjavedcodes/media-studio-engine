"""The five OpenRouter model families: registry, payloads, replies, routing, credits, files.

Everything here is mocked. No real call is made.
"""

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from PIL import Image

from app.contracts.ports import GeneratedImage, ImageOptions
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import BadInput, InvalidModelOutput, PlanLimit, ProviderCallFailed
from app.core.plans import Plan
from app.gateway.budget import SpendGuard
from app.gateway.image_gateway import ImageGateway
from app.gateway.options import RequestExtras, check_references, options_for
from app.gateway.pricing import credits_for_run, credits_per_image
from app.gateway.providers.openrouter_images import (
    OpenRouterImages,
    billed_cost,
    build_chat_body,
    build_images_body,
    parse_chat_reply,
    parse_images_reply,
)
from app.gateway.registry import ModelCard, ProviderEntry, Registry, load_registry
from app.gateway.router import RouteRequest, route_auto
from app.tools.imaging import is_svg, make_thumbnail, sniff
from tests.unit.create_helpers import critic_ok, jpeg, make_deps, make_service

REG = load_registry()
IMAGES = "https://openrouter.ai/api/v1/images"
CHAT = "https://openrouter.ai/api/v1/chat/completions"
SVG = (
    b'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2048 2048" '
    b'width="1024" height="1024"><rect width="2048" height="2048" fill="#dc143c"/>'
    b'<circle cx="1024" cy="1024" r="600" fill="#ffffff"/></svg>'
)


def card(model_id: str) -> ModelCard:
    c = REG.get(model_id)
    assert c is not None, model_id
    return c


def slug(model_id: str) -> str:
    return card(model_id).providers[0].slug


def live_registry() -> Registry:
    """Every model live, as it would be once OpenRouter is on and each model was verified."""
    return Registry([c.model_copy(update={"status": "live"}) for c in REG.all()])


def png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- the registry ----------------------------------------------------------------------------

FAMILIES = {
    "gpt_image": ("OpenAI GPT Image", 7),
    "gemini": ("Google Nano Banana & Gemini", 5),
    "seedream": ("ByteDance Seedream", 4),
    "recraft": ("Recraft Design & Vector", 8),
    "muse": ("Meta Muse", 1),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_is_complete(family: str) -> None:
    label, count = FAMILIES[family]
    cards = [c for c in REG.all() if c.family == family]
    assert len(cards) == count and all(c.family_label == label for c in cards)
    assert all(c.providers[0].name == "openrouter" for c in cards)


# model id -> (credits, max refs, endpoint). Credits are the brief's. Refs are the LIVE API's.
TABLE = {
    "gpt-image-2": (1, 16, "images"),
    "gpt-image-1-mini": (2, 16, "images"),
    "gpt-image-1": (4, 16, "images"),
    "gpt-image-2-5-flare": (2, 16, "images"),
    "gpt-image-2-5-sunburst": (7, 16, "images"),
    "gpt-image-2-6-flare": (4, 16, "images"),
    "gpt-image-2-6-sunburst": (7, 16, "images"),
    "nano-banana-2-lite": (2, 14, "chat"),
    "nano-banana": (2, 3, "chat"),  # the brief says 14. The live API says 3
    "nano-banana-2": (4, 14, "chat"),
    "nano-banana-2-1": (4, 14, "chat"),
    "nano-banana-pro": (12, 14, "chat"),
    "seedream-5-flash": (2, 14, "images"),  # the brief says 10. The live API says 14
    "seedream-5-5": (2, 10, "images"),
    "seedream-5-lite": (4, 14, "images"),
    "seedream-5-pro": (4, 14, "images"),
    "recraft-v4-1-flash-free": (1, 0, "images"),
    "recraft-v4": (4, 1, "images"),
    "recraft-v4-1": (4, 1, "images"),
    "recraft-v4-styles": (4, 10, "images"),  # the brief says 1. The live API says 10
    "recraft-v4-1-vector": (7, 1, "images"),  # the brief says 0. The live API says 1
    "recraft-v4-pro": (12, 1, "images"),
    "muse-image": (1, 4, "images"),
}


@pytest.mark.parametrize("model_id,expected", TABLE.items())
def test_credits_refs_and_endpoint_match_the_table(
    model_id: str, expected: tuple[int, int, str]
) -> None:
    credits, refs, endpoint = expected
    c = card(model_id)
    assert (c.credits, c.max_refs, c.endpoint) == (credits, refs, endpoint)


def test_slugs_are_the_real_openrouter_slugs() -> None:
    # the brief wrote `seedream-5.0-flash`, which answers 404. The real slug has dashes
    assert slug("seedream-5-flash") == "bytedance-seed/seedream-5-0-flash"
    assert slug("recraft-v4-1-flash-free") == "recraft/recraft-v4.1-flash-v2:free"
    assert slug("nano-banana-2-1") == "google/gemini-nano-banana-2.1"


def test_capabilities_follow_the_brief_where_the_api_agrees() -> None:
    assert "svg_vector" in card("recraft-v4-1-vector").capabilities
    assert card("recraft-v4-1-vector").output_format == "svg"
    assert {"gif", "complexity_dial"} <= set(card("gpt-image-2-6-flare").capabilities)
    assert "dense_text" in card("gpt-image-2-6-sunburst").capabilities
    assert "multi_ref_merge" in card("seedream-5-5").capabilities
    assert "seed_lock" in card("seedream-5-pro").capabilities
    assert "studio_photoreal" in card("nano-banana-pro").capabilities
    assert "svg" not in card("gpt-image-2-6-flare").capabilities  # the API has no SVG for GPT
    assert card("seedream-5-5").supports_seed and not card("gpt-image-2").supports_seed


def test_gpt_26_can_never_be_low_quality() -> None:
    for model_id in ("gpt-image-2-6-flare", "gpt-image-2-6-sunburst"):
        c = card(model_id)
        assert "low" in c.forbidden_qualities and c.quality in ("medium", "high")
    with pytest.raises(ValueError):
        ModelCard(
            id="x",
            name="x",
            family="gpt_image",
            quality="low",
            forbidden_qualities=["low"],
            providers=[
                ProviderEntry(name="openrouter", slug="openai/gpt-image-2.6-flare", cost=0.03)
            ],
        )


def test_cost_guardrail_blocks_low_quality_at_the_adapter_too() -> None:
    with pytest.raises(ProviderCallFailed):
        build_images_body(
            "openai/gpt-image-2.6-sunburst", "p", "1:1", None, ImageOptions(quality="low")
        )
    # low is fine for the cheap models
    assert (
        build_images_body("openai/gpt-image-2", "p", "1:1", None, ImageOptions(quality="low"))[
            "quality"
        ]
        == "low"
    )


# ---- payloads, one family at a time -----------------------------------------------------------


def test_gpt_image_payload() -> None:
    opts = options_for(card("gpt-image-2"))
    body = build_images_body(slug("gpt-image-2"), "a fox", "4:5", 7, opts)
    assert body == {
        "model": "openai/gpt-image-2",
        "prompt": "a fox",
        "n": 1,
        "aspect_ratio": "4:5",
        "response_format": "b64_json",
        "quality": "low",
        "seed": 7,
    }  # a seed is sent as given. The gateway only passes one to models that take it
    assert "resolution" not in body and "input_references" not in body


def test_gpt_image_references_transparent_background_and_complexity() -> None:
    refs = (png_b64(), "https://example.com/a.png")
    opts = options_for(
        card("gpt-image-2-6-flare"),
        RequestExtras(references=refs, transparent_bg=True, complexity=3),
    )
    body = build_images_body(slug("gpt-image-2-6-flare"), "edit", "1:1", None, opts)
    assert body["quality"] == "medium"  # never low on 2.6
    assert body["background"] == "transparent"
    assert body["input_references"] == [
        {"type": "image_url", "image_url": {"url": refs[0]}},
        {"type": "image_url", "image_url": {"url": refs[1]}},
    ]
    assert body["provider"] == {"options": {"openai": {"complexity": 3}}}


def test_gpt_gif_output() -> None:
    body = build_images_body(
        slug("gpt-image-2-6-flare"),
        "p",
        "1:1",
        None,
        ImageOptions(quality="medium", output_format="gif"),
    )
    assert body["output_format"] == "gif"


def test_gemini_chat_payload_single_turn_and_references() -> None:
    opts = options_for(card("nano-banana-2-1"))
    body = build_chat_body(slug("nano-banana-2-1"), "a fox", "16:9", opts)
    assert body == {
        "model": "google/gemini-nano-banana-2.1",
        "messages": [{"role": "user", "content": "a fox"}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "16:9", "image_size": "1K"},
    }
    ref = png_b64()
    with_ref = build_chat_body(
        slug("nano-banana-2-1"),
        "make it blue",
        "1:1",
        ImageOptions(references=(ref,), endpoint="chat"),
    )
    assert with_ref["messages"][0]["content"] == [
        {"type": "text", "text": "make it blue"},
        {"type": "image_url", "image_url": {"url": ref}},
    ]
    assert "image_size" not in with_ref["image_config"]  # only sent when the model takes one


def test_gemini_multi_turn_keeps_the_history() -> None:
    history = (
        {"role": "user", "content": "a red fox"},
        {
            "role": "assistant",
            "content": "ok",
            "images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}],
        },
    )
    body = build_chat_body("google/x", "now make it snowy", "1:1", ImageOptions(history=history))
    assert body["messages"][:2] == list(history)
    assert body["messages"][2] == {"role": "user", "content": "now make it snowy"}


def test_seedream_payload_has_seed_and_explicit_resolution() -> None:
    opts = options_for(card("seedream-5-flash"))
    body = build_images_body(slug("seedream-5-flash"), "a fox", "1:1", 42, opts)
    assert body["seed"] == 42 and body["resolution"] == "1K"
    assert body["model"] == "bytedance-seed/seedream-5-0-flash"


def test_recraft_brand_colors_use_provider_options_not_a_made_up_field() -> None:
    opts = options_for(card("recraft-v4-1"), RequestExtras(brand_colors=("#dc143c", "#FFFFFF")))
    body = build_images_body(slug("recraft-v4-1"), "poster", "1:1", None, opts)
    assert body["provider"] == {
        "options": {
            "recraft": {"controls": {"colors": [{"rgb": [220, 20, 60]}, {"rgb": [255, 255, 255]}]}}
        }
    }
    assert "rgb_colors" not in body and "image_config" not in body


def test_recraft_vector_asks_for_svg() -> None:
    body = build_images_body(
        slug("recraft-v4-1-vector"), "logo", "1:1", None, options_for(card("recraft-v4-1-vector"))
    )
    assert body["output_format"] == "svg"


def test_brand_colors_are_ignored_by_models_that_do_not_take_them() -> None:
    opts = options_for(card("gpt-image-2"), RequestExtras(brand_colors=("#FF0000",)))
    assert opts.brand_colors == ()


def test_muse_payload_is_minimal() -> None:
    body = build_images_body(
        slug("muse-image"), "a fox", "1:1", None, options_for(card("muse-image"))
    )
    assert body == {
        "model": "meta/muse-image",
        "prompt": "a fox",
        "n": 1,
        "aspect_ratio": "1:1",
        "response_format": "b64_json",
    }


@pytest.mark.parametrize("colors", [("red",), ("#12345",), ("#GGGGGG",), ("#FFFFFF",) * 6])
def test_bad_brand_colors_are_a_400(colors: tuple[str, ...]) -> None:
    with pytest.raises(BadInput):
        options_for(card("recraft-v4-1"), RequestExtras(brand_colors=colors))


def test_settings_validate_brand_colors() -> None:
    with pytest.raises(ValueError):
        ImageSettings(brand_colors=["blue"])
    assert ImageSettings(brand_colors=["#abcdef"]).brand_colors == ["#ABCDEF"]


def test_transparent_background_needs_the_capability() -> None:
    with pytest.raises(BadInput):
        options_for(card("nano-banana-2-1"), RequestExtras(transparent_bg=True))
    assert (
        options_for(card("gpt-image-2"), RequestExtras(transparent_bg=True)).background
        == "transparent"
    )


# ---- reference limits ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,n_refs,ok",
    [
        ("recraft-v4-1-flash-free", 0, True),
        ("recraft-v4-1-flash-free", 1, False),
        ("recraft-v4-1", 1, True),
        ("recraft-v4-1", 2, False),
        ("nano-banana", 3, True),
        ("nano-banana", 4, False),
        ("seedream-5-5", 10, True),
        ("seedream-5-5", 11, False),
        ("gpt-image-2-6-sunburst", 16, True),
        ("gpt-image-2-6-sunburst", 17, False),
        ("muse-image", 4, True),
        ("muse-image", 5, False),
    ],
)
def test_reference_limit(model_id: str, n_refs: int, ok: bool) -> None:
    if ok:
        check_references(card(model_id), n_refs)
    else:
        with pytest.raises(BadInput) as err:
            check_references(card(model_id), n_refs)
        assert card(model_id).name in str(err.value) and str(n_refs) in str(err.value)


def test_the_text_only_model_says_so() -> None:
    with pytest.raises(BadInput, match="does not take reference images"):
        check_references(card("recraft-v4-1-flash-free"), 1)


async def test_the_api_returns_400_before_anything_runs(tmp_path: Path) -> None:
    from app.api.deps import get_create_deps, get_run_service, get_token_signer
    from app.guardrails.deps import get_guard_deps
    from app.main import app
    from tests.unit.guard_helpers import make_deps as make_guard_deps

    deps = make_deps(tmp_path)
    service = make_service(deps)
    app.dependency_overrides[get_create_deps] = lambda: deps
    app.dependency_overrides[get_run_service] = lambda: service
    app.dependency_overrides[get_token_signer] = lambda: deps.tokens
    app.dependency_overrides[get_guard_deps] = lambda: make_guard_deps()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            image = png_b64().split(",", 1)[1]
            res = await c.post(
                "/v1/runs",
                json={
                    "prompt": "a fox",
                    "images": [image],
                    "settings": {"model": "recraft-v4-1-flash-free"},
                },
            )
            assert res.status_code == 400 and res.json()["code"] == "bad_input"
            assert "does not take reference images" in res.json()["message"]
            assert service.runs == {}  # no run was created, so no credits were held
            ok = await c.post("/v1/runs", json={"prompt": "a fox", "settings": {"model": "auto"}})
            assert ok.status_code == 200
    finally:
        app.dependency_overrides.clear()


# ---- replies -------------------------------------------------------------------------------------


def test_parse_images_reply_png_webp_svg_gif() -> None:
    for media, raw in (
        ("image/png", b"PNG"),
        ("image/webp", b"RIFF"),
        ("image/svg+xml", SVG),
        ("image/gif", b"GIF89a"),
    ):
        data = {"data": [{"b64_json": base64.b64encode(raw).decode(), "media_type": media}]}
        assert parse_images_reply(data) == (raw, media)
    assert parse_images_reply({"data": [{"b64_json": base64.b64encode(b"x").decode()}]})[1] is None


@pytest.mark.parametrize("bad", [{}, {"data": []}, {"data": [{}]}, {"data": [{"b64_json": "!!"}]}])
def test_an_images_reply_without_an_image_is_an_error(bad: dict[str, Any]) -> None:
    with pytest.raises(ProviderCallFailed):
        parse_images_reply(bad)


def test_parse_chat_reply() -> None:
    url = "data:image/jpeg;base64," + base64.b64encode(b"JPEG").decode()
    reply = {"choices": [{"message": {"images": [{"image_url": {"url": url}}]}}]}
    assert parse_chat_reply(reply) == (b"JPEG", "image/jpeg")
    for bad in (
        {},
        {"choices": []},
        {"choices": [{"message": {"images": []}}]},
        {"choices": [{"message": {"content": "no image"}}]},
    ):
        with pytest.raises(ProviderCallFailed):
            parse_chat_reply(bad)


def test_billed_cost() -> None:
    assert billed_cost({"usage": {"cost": 0.0336}}) == 0.0336
    assert billed_cost({"usage": {"cost": 0}}) == 0.0  # the free model
    assert billed_cost({}) is None


# ---- the adapter over HTTP -----------------------------------------------------------------------


def call_args(model_id: str, **over: Any) -> dict[str, Any]:
    c = card(model_id)
    args: dict[str, Any] = dict(
        slug=c.providers[0].slug,
        family=c.family,
        prompt="a fox",
        negative_prompt="",
        width=1024,
        height=1024,
        aspect_ratio="1:1",
        seed=None,
        timeout=5,
        estimated_cost_usd=c.cost_per_image,
        options=options_for(c),
    )
    args.update(over)
    return args


def images_reply(
    raw: bytes = b"PNG", media: str = "image/png", cost: float | None = 0.006
) -> httpx.Response:
    body: dict[str, Any] = {
        "data": [{"b64_json": base64.b64encode(raw).decode(), "media_type": media}]
    }
    if cost is not None:
        body["usage"] = {"cost": cost}
    return httpx.Response(200, json=body)


def adapter(http: httpx.AsyncClient, cap: float = 5.0) -> tuple[OpenRouterImages, SpendGuard]:
    guard = SpendGuard(cap)
    return OpenRouterImages(
        http, api_key="k", base_url="https://openrouter.ai/api/v1", guard=guard
    ), guard


@respx.mock
async def test_gpt_goes_to_the_images_route_and_gemini_to_chat() -> None:
    images = respx.post(IMAGES).mock(return_value=images_reply())
    chat_url = "data:image/jpeg;base64," + base64.b64encode(b"JPEG").decode()
    chat = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"images": [{"image_url": {"url": chat_url}}]}}],
                "usage": {"cost": 0.0336},
            },
        )
    )
    async with httpx.AsyncClient() as http:
        p, guard = adapter(http)
        gpt = await p.generate(**call_args("gpt-image-2"))
        gem = await p.generate(**call_args("nano-banana-2-lite"))
    assert images.call_count == 1 and chat.call_count == 1
    assert (gpt.data, gpt.media_type, gpt.cost_usd) == (b"PNG", "image/png", 0.006)
    assert (gem.data, gem.media_type, gem.cost_usd) == (b"JPEG", "image/jpeg", 0.0336)
    assert json.loads(images.calls[0].request.content)["quality"] == "low"
    assert json.loads(chat.calls[0].request.content)["modalities"] == ["image", "text"]
    assert guard.spent_usd == pytest.approx(0.006 + 0.0336)


@respx.mock
async def test_svg_comes_through_as_bytes_with_its_media_type() -> None:
    respx.post(IMAGES).mock(return_value=images_reply(SVG, "image/svg+xml", 0.08))
    async with httpx.AsyncClient() as http:
        p, _ = adapter(http)
        out = await p.generate(**call_args("recraft-v4-1-vector"))
    assert out.media_type == "image/svg+xml" and out.data == SVG and is_svg(out.data)


@respx.mock
async def test_a_503_is_retried_once_and_then_succeeds() -> None:
    route = respx.post(IMAGES).mock(side_effect=[httpx.Response(503), images_reply()])
    async with httpx.AsyncClient() as http:
        p, guard = adapter(http)
        out = await p.generate(**call_args("gpt-image-2"))
    assert route.call_count == 2 and out.cost_usd == 0.006
    assert guard.spent_usd == pytest.approx(0.006)  # the billed cost of the one that worked


@respx.mock
@pytest.mark.parametrize("status", [429, 502, 500])
async def test_it_gives_up_after_one_retry(status: int) -> None:
    route = respx.post(IMAGES).mock(return_value=httpx.Response(status))
    async with httpx.AsyncClient() as http:
        p, _ = adapter(http)
        with pytest.raises(ProviderCallFailed) as err:
            await p.generate(**call_args("gpt-image-2"))
    assert route.call_count == 2 and err.value.retryable


@respx.mock
@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413])
async def test_other_errors_are_not_retried(status: int) -> None:
    route = respx.post(IMAGES).mock(return_value=httpx.Response(status))
    async with httpx.AsyncClient() as http:
        p, guard = adapter(http)
        with pytest.raises(ProviderCallFailed) as err:
            await p.generate(**call_args("gpt-image-2"))
    assert route.call_count == 1 and not err.value.retryable
    assert guard.spent_usd == 0  # the provider refused it, nothing was billed


@respx.mock
async def test_a_timeout_is_never_retried() -> None:
    route = respx.post(IMAGES).mock(side_effect=httpx.ReadTimeout("slow"))
    async with httpx.AsyncClient() as http:
        p, guard = adapter(http)
        with pytest.raises(ProviderCallFailed):
            await p.generate(**call_args("gpt-image-2"))
    assert route.call_count == 1  # the first request may still be running and be billed
    assert guard.spent_usd == pytest.approx(0.006)  # so the estimate counts against the cap


@respx.mock
async def test_the_gateway_sends_the_card_options_and_a_seed_only_when_supported() -> None:
    route = respx.post(IMAGES).mock(return_value=images_reply())
    async with httpx.AsyncClient() as http:
        p, _ = adapter(http)
        gw = ImageGateway({"openrouter": p}, ["openrouter"], paid={"openrouter"})
        for model_id in ("seedream-5-flash", "gpt-image-2"):
            c = card(model_id)
            await gw.generate(
                c,
                prompt="a fox",
                negative_prompt="",
                width=1024,
                height=1024,
                aspect_ratio="1:1",
                seed=5,
                options=options_for(c),
            )
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert bodies[0]["seed"] == 5 and "seed" not in bodies[1]


# ---- images: SVG, GIF, WEBP ----------------------------------------------------------------------


def test_sniff_reads_svg_size_from_width_and_height_or_the_viewbox() -> None:
    info = sniff(SVG, "image/svg+xml")
    assert (info.mime, info.ext, info.width, info.height) == ("image/svg+xml", "svg", 1024, 1024)
    only_box = (
        b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480"></svg>'
    )
    assert (sniff(only_box).width, sniff(only_box).height) == (640, 480)


def test_sniff_reads_gif_and_webp() -> None:
    gif = io.BytesIO()
    Image.new("RGB", (32, 16), "blue").save(gif, format="GIF")
    assert sniff(gif.getvalue(), "image/gif").ext == "gif"
    webp = io.BytesIO()
    Image.new("RGB", (32, 16), "blue").save(webp, format="WEBP")
    assert (sniff(webp.getvalue()).mime, sniff(webp.getvalue()).width) == ("image/webp", 32)


@pytest.mark.parametrize(
    "evil",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><foreignObject><div/></foreignObject></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><a href="javascript:alert(1)"/></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://evil.example/x.png"/></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><use xlink:href="https://evil.example/x.svg#a"/></svg>',
    ],
)
def test_an_svg_with_active_content_is_refused(evil: bytes) -> None:
    with pytest.raises(InvalidModelOutput):
        sniff(evil, "image/svg+xml")


def test_a_non_svg_labelled_svg_is_refused() -> None:
    with pytest.raises(InvalidModelOutput):
        sniff(b"just text", "image/svg+xml")


def test_svg_and_gif_thumbnails_are_256px_jpeg() -> None:
    thumb = make_thumbnail(SVG)
    info = sniff(thumb)
    assert info.mime == "image/jpeg" and max(info.width, info.height) == 256
    with Image.open(io.BytesIO(thumb)) as im:
        r, g, b = im.convert("RGB").getpixel((5, 5))  # the crimson corner, so it really was drawn
        assert r > 180 and g < 80
    frames = io.BytesIO()
    first = Image.new("RGB", (64, 64), "red")
    first.save(
        frames, format="GIF", save_all=True, append_images=[Image.new("RGB", (64, 64), "blue")]
    )
    assert sniff(make_thumbnail(frames.getvalue())).mime == "image/jpeg"


# ---- routing (the Image Grand Agent) -------------------------------------------------------------

LIVE = live_registry()
BOTH = ["cloudflare", "openrouter"]


def route(prompt: str = "a fox in the snow", plan: Plan = Plan.MAX, **kw: Any) -> Any:
    reg = kw.pop("registry", LIVE)
    providers = kw.pop("providers", BOTH)
    return route_auto(
        reg,
        RouteRequest(
            prompt=prompt, plan=plan, aspect_ratio="1:1", enabled_providers=providers, **kw
        ),
    )


@pytest.mark.parametrize(
    "prompt,kw,model_id,rule",
    [
        ("a logo for my coffee shop", {}, "recraft-v4-1-vector", "vector"),
        ("minimal vector icon set", {}, "recraft-v4-1-vector", "vector"),
        ("an emblem for a football club", {}, "recraft-v4-1-vector", "vector"),
        ("a fox", {"brand_colors": ("#112233",)}, "recraft-v4-1", "brand"),
        ("a summer sale poster", {}, "recraft-v4-1", "brand"),
        ("a flyer for a concert", {}, "recraft-v4-1", "brand"),
        ("packaging for a tea box", {}, "recraft-v4-1", "brand"),
        (
            "a poster in this style",
            {"has_style_ref": True},
            "recraft-v4-1",
            "brand",
        ),  # Styles is not listed
        ("put these together", {"n_refs": 2}, "seedream-5-flash", "merge"),
        ("put these together", {"n_refs": 14}, "seedream-5-flash", "merge"),  # it takes 14
        # nothing listed takes 15 images, so the plan's own tier decides (Max: Sunburst)
        ("put these together", {"n_refs": 15}, "gpt-image-2-5-sunburst", "merge->category"),
        ("change the jacket to leather", {"n_refs": 1}, "gpt-image-2-5-sunburst", "precision"),
        (
            'a sign that says "Open every day from nine to five"',
            {},
            "gpt-image-2-5-sunburst",
            "precision",
        ),
        ("a slide with a headline", {}, "gpt-image-2-5-sunburst", "precision"),
        ("a fox", {"draft": True}, "flux-2-klein-4b", "draft"),
        ("a fox", {"draft": True, "n_refs": 1}, "gpt-image-2-5-flare", "draft"),
        (
            "a logo",
            {"draft": True},
            "flux-2-klein-4b",
            "draft",
        ),  # a draft is the cheap choice
        ("a fox in the snow", {}, "nano-banana-2-1", "default"),
    ],
)
def test_auto_routing_rules(prompt: str, kw: dict[str, Any], model_id: str, rule: str) -> None:
    routed = route(prompt, **kw)
    assert (routed.card.id, routed.rule) == (model_id, rule)


def test_a_candidate_that_cannot_run_is_skipped_for_the_next() -> None:
    no_seedream = Registry([c for c in LIVE.all() if c.id != "seedream-5-flash"])
    assert route("merge", registry=no_seedream, n_refs=3).card.id == "nano-banana-2-1"
    no_klein = Registry([c for c in LIVE.all() if c.id != "flux-2-klein-4b"])
    assert route("a fox", registry=no_klein, draft=True).card.id == "recraft-v4-1-flash"


def test_auto_never_picks_a_model_outside_the_curated_list() -> None:
    prompts = [
        "a fox",
        "a logo",
        "a poster",
        'a sign that says "Open every day from nine to five"',
        "merge these",
        "edit this",
    ]
    for prompt in prompts:
        for refs in (0, 1, 3, 12, 15):
            for draft in (False, True):
                try:
                    card = route(prompt, n_refs=refs, draft=draft).card
                except PlanLimit:
                    continue
                assert card.auto, (prompt, refs, draft, card.id)
    # even with every model live, the hidden ones (GPT Image 2, Nano Banana 2 ...) are not chosen
    hidden = {c.id for c in LIVE.all() if not c.auto}
    assert {"gpt-image-2", "nano-banana-2", "seedream-5-5", "muse-image"} <= hidden


def test_the_plan_limits_the_rules() -> None:
    # the vector model is Ultra tier. A Basic user falls through to the category picker
    routed = route("a logo", plan=Plan.BASIC)
    assert routed.rule == "vector->category" and routed.card.tier.value == "basic"
    assert route("a logo", plan=Plan.ULTRA).card.id == "recraft-v4-1-vector"


def test_with_openrouter_off_the_cloudflare_models_are_used() -> None:
    routed = route("a logo for my shop", providers=["cloudflare"])
    assert routed.rule.endswith("->category") and routed.card.providers[0].name == "cloudflare"
    off = route("a fox", registry=REG, providers=["cloudflare"], plan=Plan.PRO)
    assert off.card.id in {"flux-1-schnell", "flux-2-klein-4b", "flux-2-klein-9b", "phoenix-1"}


def test_only_models_that_take_the_references_can_be_picked() -> None:
    only = Registry([c for c in LIVE.all() if c.id == "nano-banana-2-1"])
    assert route("x", n_refs=14, registry=only).card.id == "nano-banana-2-1"
    with pytest.raises(PlanLimit):
        route("x", n_refs=30)  # no model takes 30 images


def test_transparent_background_only_picks_capable_models() -> None:
    routed = route("a fox", transparent=True)
    assert "transparent_bg" in routed.card.capabilities


def test_aspect_ratio_is_respected() -> None:
    req = RouteRequest(prompt="a fox", plan=Plan.MAX, aspect_ratio="21:9", enabled_providers=BOTH)
    assert "21:9" in route_auto(LIVE, req).card.supported_aspect_ratios


# ---- credits -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,n,kw,expected",
    [
        ("gpt-image-2", 1, {}, 1),
        ("gpt-image-2", 4, {}, 4),
        ("gpt-image-2-6-sunburst", 2, {}, 14),
        ("nano-banana-2-lite", 1, {}, 2),
        ("nano-banana-pro", 1, {}, 12),
        ("nano-banana-pro", 1, {"resolution": "4K"}, 24),  # 4K multiplier (assumption: 2x)
        ("nano-banana-pro", 2, {"resolution": "4K"}, 48),
        ("seedream-5-5", 1, {}, 2),
        ("seedream-5-5", 1, {"n_refs": 1}, 2),  # the first reference is free
        ("seedream-5-5", 1, {"n_refs": 4}, 5),  # +1 for each reference above 1
        ("seedream-5-5", 2, {"n_refs": 4}, 10),
        ("seedream-5-flash", 1, {"n_refs": 9}, 2),  # no surcharge on other models
        ("recraft-v4-1-vector", 1, {}, 7),
        ("recraft-v4-pro", 1, {}, 12),
        ("recraft-v4-1-flash-free", 1, {}, 1),
        ("muse-image", 1, {}, 1),
        ("flux-1-schnell", 1, {}, 1),  # no own credits: the tier's (basic = 1)
        ("phoenix-1", 1, {}, 3),  # pro tier = 3
    ],
)
def test_credits(model_id: str, n: int, kw: dict[str, Any], expected: int) -> None:
    assert credits_for_run(card(model_id), n, **kw) == expected


def test_credits_round_up_so_a_run_is_never_under_charged() -> None:
    c = card("gpt-image-2").model_copy(update={"resolution_multipliers": {"2K": 1.5}})
    assert credits_per_image(c, resolution="2K") == 2


async def test_a_failed_openrouter_run_refunds_the_hold(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)
    deps.registry = LIVE
    deps.enabled_providers = ["openrouter"]

    class Down:
        name = "openrouter"

        async def generate(self, **_: Any) -> GeneratedImage:
            raise ProviderCallFailed("503", retryable=True)

    deps.gateway = ImageGateway({"openrouter": Down()}, ["openrouter"], paid={"openrouter"})  # type: ignore[dict-item]
    service = make_service(deps)
    service.start(
        "rf",
        "u1",
        GenerationRequest(
            prompt="a fox", plan=Plan.MAX, settings=ImageSettings(model="nano-banana-2-1", n=2)
        ),
    )
    events = [e async for e in service.stream("rf")]
    assert events[-1].type == "error"
    assert await deps.credits.balance("u1") == 100  # held 8, refunded 8


# ---- the whole pipeline with an OpenRouter model -------------------------------------------------


class RecordingProvider:
    def __init__(self, name: str, data: bytes, media: str | None) -> None:
        self.name, self.data, self.media = name, data, media
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kw: Any) -> GeneratedImage:
        self.calls.append(kw)
        return GeneratedImage(data=self.data, cost_usd=0.08, media_type=self.media)


async def test_a_logo_is_routed_to_the_vector_model_and_delivered_as_an_svg(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    provider = RecordingProvider("openrouter", SVG, "image/svg+xml")
    deps.registry = LIVE
    deps.enabled_providers = ["openrouter", "cloudflare"]
    deps.gateway = ImageGateway(
        {"openrouter": provider, "cloudflare": provider},
        deps.enabled_providers,  # type: ignore[dict-item]
        paid={"openrouter"},
    )
    service = make_service(deps)
    service.start(
        "logo",
        "u1",
        GenerationRequest(
            prompt="a logo for my coffee shop", plan=Plan.MAX, settings=ImageSettings(n=1)
        ),
    )
    events = [e async for e in service.stream("logo")]
    ready = events[-1]
    assert ready.type == "ready" and ready.images[0].url.split("?")[0].endswith(".svg")
    sent = provider.calls[0]
    assert sent["slug"] == "recraft/recraft-v4.1-vector"
    assert sent["options"].output_format == "svg" and sent["options"].resolution is None
    # the safety check saw pixels (a JPEG thumbnail of the drawn SVG), not SVG text
    vision = deps.vision.calls[0]["image_urls"][0]  # type: ignore[attr-defined]
    assert vision.startswith("data:image/jpeg;base64,")
    assert (tmp_path / "logo" / "img_1.svg").read_bytes() == SVG
    assert (tmp_path / "logo" / "thumb_1.jpg").exists()
    assert await deps.credits.balance("u1") == 100 - 7  # the brief's 7 credits for vector


async def test_brand_colors_reach_the_recraft_call(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    provider = RecordingProvider("openrouter", jpeg(), "image/jpeg")
    deps.registry = LIVE
    deps.enabled_providers = ["openrouter"]
    deps.gateway = ImageGateway({"openrouter": provider}, ["openrouter"], paid={"openrouter"})  # type: ignore[dict-item]
    service = make_service(deps)
    service.start(
        "bc",
        "u1",
        GenerationRequest(
            prompt="a summer sale",
            plan=Plan.MAX,
            settings=ImageSettings(n=1, brand_colors=["#dc143c"]),
        ),
    )
    events = [e async for e in service.stream("bc")]
    assert events[-1].type == "ready"
    assert provider.calls[0]["slug"] == "recraft/recraft-v4.1"
    assert provider.calls[0]["options"].brand_colors == ("#DC143C",)


async def test_an_svg_with_a_script_is_dropped_and_not_charged(tmp_path: Path) -> None:
    evil = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><script>x()</script></svg>'
    )
    deps = make_deps(tmp_path)
    provider = RecordingProvider("openrouter", evil, "image/svg+xml")
    deps.registry = LIVE
    deps.enabled_providers = ["openrouter"]
    deps.gateway = ImageGateway({"openrouter": provider}, ["openrouter"], paid={"openrouter"})  # type: ignore[dict-item]
    service = make_service(deps)
    service.start(
        "ev", "u1", GenerationRequest(prompt="a logo", plan=Plan.MAX, settings=ImageSettings(n=1))
    )
    events = [e async for e in service.stream("ev")]
    assert events[-1].type == "error" and await deps.credits.balance("u1") == 100


async def test_svg_files_are_served_locked_down(tmp_path: Path) -> None:
    from app.api.deps import get_create_deps, get_run_service, get_token_signer
    from app.guardrails.deps import get_guard_deps
    from app.main import app
    from tests.unit.guard_helpers import make_deps as make_guard_deps

    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    provider = RecordingProvider("openrouter", SVG, "image/svg+xml")
    deps.registry = LIVE
    deps.enabled_providers = ["openrouter"]
    deps.gateway = ImageGateway({"openrouter": provider}, ["openrouter"], paid={"openrouter"})  # type: ignore[dict-item]
    service = make_service(deps)
    app.dependency_overrides[get_create_deps] = lambda: deps
    app.dependency_overrides[get_run_service] = lambda: service
    app.dependency_overrides[get_token_signer] = lambda: deps.tokens
    app.dependency_overrides[get_guard_deps] = lambda: make_guard_deps()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            created = (
                await c.post(
                    "/v1/runs", json={"prompt": "a logo", "plan": "max", "settings": {"n": 1}}
                )
            ).json()
            text = (
                await c.get(f"/v1/runs/{created['run_id']}/events", params={"t": created["token"]})
            ).text
            url = next(
                ln for ln in text.splitlines() if ln.startswith("data:") and '"images"' in ln
            )
            file_url = json.loads(url[6:])["images"][0]["url"]
            res = await c.get(file_url)
            assert res.status_code == 200 and res.headers["content-type"] == "image/svg+xml"
            assert res.headers["x-content-type-options"] == "nosniff"
            assert "sandbox" in res.headers["content-security-policy"]
            assert res.content == SVG
    finally:
        app.dependency_overrides.clear()


# ---- edit mode -----------------------------------------------------------------------------------


def test_edit_mode_plans_the_change_and_comes_before_the_model_section() -> None:
    from app.agents.image_grand.workers import art_director as ad
    from app.agents.image_grand.workers import prompt_writer as pw
    from app.gateway.looks import get_looks

    plan_text = ad.system_prompt("BASE", get_looks(), "1:1", has_references=True)
    assert "## Mode: edit" in plan_text and "Plan only what CHANGES" in plan_text
    assert plan_text.index("## Mode: edit") < plan_text.index("## Look library")
    text = pw.system_prompt("BASE", card("gpt-image-2"), "1:1", n=1, has_references=True)
    assert "## Mode: edit" in text and "short, direct EDIT" in text
    assert text.index("## Mode: edit") < text.index("## Target model")
    assert text.rstrip().endswith(card("gpt-image-2").prompt_guide.strip())  # the guide stays last
    assert "Mode: edit" not in pw.system_prompt("BASE", card("gpt-image-2"), "1:1", n=1)


def test_recraft_guide_keeps_copy_in_quotes_and_drops_lens_jargon() -> None:
    guide = card("recraft-v4-1").prompt_guide
    assert '"double quotes"' in guide and "headline" in guide and "hierarchy" in guide
    assert "Do NOT use lens jargon" in guide


@pytest.mark.parametrize(
    "model_id", ["gpt-image-2", "nano-banana-2-1", "seedream-5-pro", "muse-image"]
)
def test_photoreal_guides_expand_into_visual_detail(model_id: str) -> None:
    guide = card(model_id).prompt_guide
    assert "camera" in guide and "texture" in guide and "light" in guide
    assert "atmosphere" not in guide.lower()  # a weak word the code guard strips


# ---- the catalog (for the picker) ----------------------------------------------------------------


async def test_the_catalog_sends_credits_refs_capabilities_and_surcharges(tmp_path: Path) -> None:
    from app.api.deps import get_create_deps
    from app.main import app

    deps = make_deps(tmp_path)
    app.dependency_overrides[get_create_deps] = lambda: deps
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            res = (await c.get("/v1/models", params={"plan": "max"})).json()
            models = {m["id"]: m for m in res["models"]}
    finally:
        app.dependency_overrides.clear()
    assert models["gpt-image-2-5-flare"]["credits_per_image"] == 2
    assert models["nano-banana-pro"]["credits_per_image"] == 12
    assert models["nano-banana-pro"]["resolution_multipliers"] == {"4K": 2.0}
    assert models["seedream-5-flash"]["max_refs"] == 14
    assert "multi_ref_merge" in models["seedream-5-flash"]["capabilities"]
    assert "multi_ref_merge" in models["nano-banana-2-1"]["capabilities"]
    assert "svg_vector" in models["recraft-v4-1-vector"]["capabilities"]
    assert models["recraft-v4-1-vector"]["credits_per_image"] == 7
    assert models["flux-2-klein-4b"]["credits_per_image"] == 1
    assert models["flux-2-klein-4b"]["max_refs"] == 0
    assert "flux-1-schnell" not in models and "seedream-5-5" not in models  # not in the picker
    assert "openrouter" not in json.dumps(res).lower()  # no provider names go out
