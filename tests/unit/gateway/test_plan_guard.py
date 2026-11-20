"""The gateway's pre-call guard: plan tier, price-multiplying settings and `n`, enforced in the
GATEWAY (not the agents), before any provider is called. Plus: one registry, one view.

Every test here asserts on what reached the (respx-mocked) provider: a refused request must
never be sent, because a sent request is a billed request."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from media_engine.core.config import Settings
from media_engine.gateway.errors import AllProvidersFailed, RequestNotAllowed
from media_engine.gateway.fallback import AvailabilityState
from media_engine.gateway.image_registry import (
    ImageRegistry,
    image_registry,
    reload_image_registry,
    use_image_registry,
)
from media_engine.gateway.model_gateway import ModelGateway
from media_engine.gateway.registry import card_or_default
from media_engine.gateway.router import Gateway, parse_ref
from media_engine.gateway.types import ImageRequest
from tests.unit.harness.fakes import ScriptedImages, image_bytes

from .conftest import ACCOUNT, CF_RUN, NOW

OR_IMAGES = "https://openrouter.ai/api/v1/images"
SCHNELL_URL = f"{CF_RUN}/@cf/black-forest-labs/flux-1-schnell"
GPT_IMAGE = "openai/gpt-image-2"
FLUX_PRO = "black-forest-labs/flux.2-pro"  # pro tier, 1 image per call
QWEN = "qwen/qwen-image-3"  # pro tier, 6 per call
RIVERFLOW = "sourceful/riverflow-v2.5-pro"  # max tier, offers 4K


def or_ok(n: int = 1) -> httpx.Response:
    png = base64.b64encode(image_bytes(8, 8)).decode()
    return httpx.Response(200, json={"data": [{"b64_json": png}] * n, "usage": {"cost": 0.006}})


def cf_ok() -> httpx.Response:
    png = base64.b64encode(image_bytes(8, 8)).decode()
    return httpx.Response(200, json={"result": {"image": png}})


@pytest.fixture
def registry() -> ImageRegistry:
    """The real models.yaml with no catalogue cache: deterministic on every machine."""
    reg = ImageRegistry.load(None)
    use_image_registry(reg)
    return reg


def gateway(*entries: str) -> Gateway:
    settings = Settings(
        _env_file=None,
        cloudflare_account_id=ACCOUNT,
        cloudflare_api_token="cf-test",
        openrouter_api_key="or-test-key",
        allow_paid=True,  # the guard must hold even with paid calls switched ON
    )

    async def no_sleep(_: float) -> None:
        return None

    return Gateway(
        settings,
        chains={"image_create": [parse_ref(e) for e in entries]},
        state=AvailabilityState(now=lambda: NOW),
        sleep=no_sleep,
    )


def sent(route: respx.Route, i: int = 0) -> dict:
    return json.loads(route.calls[i].request.content)


# -- item 1: quality is enforced by plan, in the gateway ----------------------------------------


@respx.mock
@pytest.mark.parametrize("plan", ["free", "pro", "max"])
async def test_high_quality_below_ultra_never_reaches_the_provider(registry, plan):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{GPT_IMAGE}")
    try:
        with pytest.raises(RequestNotAllowed) as exc:
            await gw.image("image_create", ImageRequest(prompt="a", quality="high"), plan=plan)
    finally:
        await gw.aclose()
    assert not route.called  # nothing sent, nothing billed
    assert exc.value.limits == [f"quality=high on {GPT_IMAGE} needs the ultra plan; plan is {plan}"]


@respx.mock
async def test_high_quality_is_sent_on_the_ultra_plan(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{GPT_IMAGE}")
    try:
        await gw.image("image_create", ImageRequest(prompt="a", quality="high"), plan="ultra")
    finally:
        await gw.aclose()
    assert sent(route)["quality"] == "high"


@respx.mock
async def test_unset_quality_is_sent_as_low_and_allowed_on_free(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{GPT_IMAGE}")
    try:
        await gw.image("image_create", ImageRequest(prompt="a"), plan="free")
    finally:
        await gw.aclose()
    assert sent(route)["quality"] == "low"  # billed like unset ($0.006), never the model's pick


@respx.mock
async def test_no_plan_given_is_treated_as_free(registry):
    # A caller (or a bug upstream) that drops the plan can only make a request cheaper.
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{GPT_IMAGE}")
    try:
        with pytest.raises(RequestNotAllowed, match="plan is free"):
            await gw.image("image_create", ImageRequest(prompt="a", quality="high"))
    finally:
        await gw.aclose()
    assert not route.called


@respx.mock
async def test_4k_resolution_below_ultra_is_refused(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{RIVERFLOW}")
    try:
        with pytest.raises(RequestNotAllowed, match="resolution=4K"):
            await gw.image("image_create", ImageRequest(prompt="a", resolution="4K"), plan="max")
    finally:
        await gw.aclose()
    assert not route.called


@respx.mock
async def test_a_model_above_the_plans_tier_is_skipped(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{QWEN}")
    try:
        with pytest.raises(RequestNotAllowed, match="pro-tier model; plan free"):
            await gw.image("image_create", ImageRequest(prompt="a"), plan="free")
    finally:
        await gw.aclose()
    assert not route.called


@respx.mock
async def test_an_unregistered_paid_model_is_never_called(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway("openrouter/google/gemini-2.5-flash-image")  # not in models.yaml
    try:
        with pytest.raises(RequestNotAllowed, match="not in the image registry"):
            await gw.image("image_create", ImageRequest(prompt="a"), plan="ultra")
    finally:
        await gw.aclose()
    assert not route.called


@respx.mock
async def test_a_refused_entry_is_skipped_and_the_next_one_serves(registry):
    paid = respx.post(OR_IMAGES).mock(return_value=or_ok())
    free = respx.post(SCHNELL_URL).mock(return_value=cf_ok())
    gw = gateway(f"openrouter/{GPT_IMAGE}", "cloudflare/@cf/black-forest-labs/flux-1-schnell")
    try:
        result = await gw.image("image_create", ImageRequest(prompt="a", quality="high"))
    finally:
        await gw.aclose()
    assert not paid.called and free.called
    assert result.provider == "cloudflare"
    reason = result.attempts[0].reason or ""
    assert result.attempts[0].outcome == "skipped" and reason.startswith("not_allowed:")


async def test_the_image_creator_path_passes_its_plan_to_the_gateway():
    images = ScriptedImages([image_bytes(8, 8)])
    gw = ModelGateway(images, lambda agent, plan: "image_create")
    await gw.generate("a", width=64, height=64, tier="pro", seed=1, deadline_ms=1000)
    assert images.plans == ["pro"]


# -- item 7: n is routed, never silently reduced ------------------------------------------------


@respx.mock
async def test_n_the_model_supports_is_sent_as_is(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok(4))
    gw = gateway(f"openrouter/{QWEN}")
    try:
        result = await gw.image("image_create", ImageRequest(prompt="a", n=4), plan="pro")
    finally:
        await gw.aclose()
    assert sent(route)["model"] == QWEN and sent(route)["n"] == 4
    assert len(result.output.images) == 4


@respx.mock
async def test_n_above_the_models_limit_is_routed_to_a_same_tier_model_that_can(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok(4))
    gw = gateway(f"openrouter/{FLUX_PRO}")  # FLUX.2 Pro: 1 per call
    try:
        result = await gw.image("image_create", ImageRequest(prompt="a", n=4), plan="pro")
    finally:
        await gw.aclose()
    assert sent(route)["model"] == QWEN and sent(route)["n"] == 4  # same pro tier, max 6
    assert result.model == QWEN
    routed = result.attempts[0]
    assert routed.model == FLUX_PRO and routed.outcome == "skipped"
    assert "routed to qwen/qwen-image-3" in (routed.reason or "")


@respx.mock
async def test_n_no_same_tier_model_can_serve_is_a_typed_error_naming_the_limit(registry):
    route = respx.post(OR_IMAGES).mock(return_value=or_ok())
    gw = gateway(f"openrouter/{FLUX_PRO}")
    try:
        with pytest.raises(RequestNotAllowed) as exc:
            await gw.image("image_create", ImageRequest(prompt="a", n=8), plan="pro")
    finally:
        await gw.aclose()
    assert not route.called
    (limit,) = exc.value.limits
    assert "n=8 exceeds the per-call max of black-forest-labs/flux.2-pro (1)" in limit


@respx.mock
async def test_cloudflare_never_silently_returns_fewer_images_than_asked(registry):
    route = respx.post(SCHNELL_URL).mock(return_value=cf_ok())
    gw = gateway("cloudflare/@cf/black-forest-labs/flux-1-schnell")
    try:
        with pytest.raises(RequestNotAllowed, match="returns 1 image per call"):
            await gw.image("image_create", ImageRequest(prompt="a", n=2), plan="ultra")
    finally:
        await gw.aclose()
    assert not route.called


async def test_a_provider_failure_is_still_all_providers_failed_not_a_plan_refusal(registry):
    with respx.mock:
        respx.post(OR_IMAGES).mock(return_value=httpx.Response(503))
        gw = gateway(f"openrouter/{GPT_IMAGE}")
        try:
            with pytest.raises(AllProvidersFailed) as exc:
                await gw.image("image_create", ImageRequest(prompt="a"), plan="free")
        finally:
            await gw.aclose()
    assert not isinstance(exc.value, RequestNotAllowed)


# -- item 2: one registry, one view -------------------------------------------------------------


async def test_the_provider_cards_and_gateway_read_the_same_registry(registry):
    gw = gateway(f"openrouter/{GPT_IMAGE}")
    try:
        provider = gw.providers["openrouter"]
        assert provider.registry is image_registry() is registry  # type: ignore[attr-defined]
        for spec in registry.all():
            mine = provider.spec(spec.slug)  # type: ignore[attr-defined]
            assert mine == spec  # same price, tier and capabilities
            card = card_or_default("openrouter", spec.slug, "image")
            assert card.accepts_seed == spec.supports_seed
            assert card.supports_edit == spec.supports_edit
    finally:
        await gw.aclose()


async def test_swapping_the_registry_changes_every_view_together(tmp_path):
    from .conftest import entry_model, install_registry

    install_registry(tmp_path, [], [entry_model(GPT_IMAGE, supports_seed=True, n_per_call=3)])
    gw = gateway(f"openrouter/{GPT_IMAGE}")
    try:
        provider = gw.providers["openrouter"]
        assert provider.spec(GPT_IMAGE).n_per_call == 3  # type: ignore[attr-defined]
        assert card_or_default("openrouter", GPT_IMAGE, "image").accepts_seed
        reloaded = reload_image_registry()  # the refresh path: back to models.yaml + cache file
        assert provider.registry is reloaded  # type: ignore[attr-defined]
        assert not card_or_default("openrouter", GPT_IMAGE, "image").accepts_seed
    finally:
        await gw.aclose()
