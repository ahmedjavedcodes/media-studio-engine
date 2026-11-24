"""B4: the fal adapter and the fal registry sync. Mock tests only. fal is OFF."""

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx

from app.core.errors import ProviderCallFailed, SpendCapReached
from app.gateway.budget import SpendGuard
from app.gateway.chains import load_chain
from app.gateway.providers.fal_images import FalImages
from app.gateway.registry import ModelCard, ProviderEntry, Registry, load_registry
from app.gateway.sync import sync_fal

RUN = "https://fal.run/fal-ai/flux/schnell"
IMG = "https://v3.fal.media/files/abc/out.jpg"


def args(estimate: float = 1.0, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        slug="fal-ai/flux/schnell",
        family="flux",
        prompt="a fox",
        negative_prompt="",
        width=1024,
        height=768,
        aspect_ratio="4:3",
        seed=7,
        timeout=5,
        estimated_cost_usd=estimate,
    )
    base.update(over)
    return base


def reply(url: str = IMG) -> httpx.Response:
    return httpx.Response(
        200, json={"images": [{"url": url, "width": 1024, "height": 768}], "seed": 7}
    )


@respx.mock
async def test_it_sends_an_explicit_size_and_the_key_and_downloads_the_image() -> None:
    route = respx.post(RUN).mock(return_value=reply())
    dl = respx.get(IMG).mock(return_value=httpx.Response(200, content=b"JPEGDATA"))
    guard = SpendGuard(5)
    async with httpx.AsyncClient() as http:
        out = await FalImages(http, api_key="k", guard=guard).generate(**args())
    assert out.data == b"JPEGDATA" and dl.called
    req = route.calls[0].request
    assert req.headers["authorization"] == "Key k"
    sent = json.loads(req.content)
    assert sent["image_size"] == {"width": 1024, "height": 768}  # always explicit
    assert sent["seed"] == 7 and sent["num_images"] == 1 and sent["enable_safety_checker"] is True
    assert guard.spent_usd == pytest.approx(1.0)  # fal reports no cost: the estimate is charged


@respx.mock
async def test_the_cap_holds_with_parallel_calls() -> None:
    route = respx.post(RUN).mock(return_value=reply())
    respx.get(IMG).mock(return_value=httpx.Response(200, content=b"x"))
    guard = SpendGuard(1.5)
    async with httpx.AsyncClient() as http:
        p = FalImages(http, api_key="k", guard=guard)
        results = await asyncio.gather(
            *(p.generate(**args()) for _ in range(4)), return_exceptions=True
        )
    assert route.call_count == 1 and sum(isinstance(r, SpendCapReached) for r in results) == 3


@respx.mock
@pytest.mark.parametrize(
    "status,retryable", [(422, False), (401, False), (403, False), (429, True), (503, True)]
)
async def test_errors_map_to_retryable_or_not(status: int, retryable: bool) -> None:
    respx.post(RUN).mock(return_value=httpx.Response(status))
    guard = SpendGuard(5)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ProviderCallFailed) as err:
            await FalImages(http, api_key="k", guard=guard).generate(**args())
    assert err.value.retryable is retryable
    assert guard.spent_usd == (1.0 if retryable else 0.0)  # a refused request is not counted


@respx.mock
@pytest.mark.parametrize(
    "url",
    [
        "http://v3.fal.media/x.jpg",
        "https://evil.example/x.jpg",
        "https://fal.media.evil.com/x.jpg",
        "file:///etc/passwd",
        "https://169.254.169.254/latest",
    ],
)
async def test_it_only_downloads_from_fal_hosts_over_https(url: str) -> None:
    respx.post(RUN).mock(return_value=reply(url))
    async with httpx.AsyncClient() as http:
        with pytest.raises(ProviderCallFailed) as err:
            await FalImages(http, api_key="k", guard=SpendGuard(5)).generate(**args())
    assert "unexpected image address" in str(err.value)


async def test_no_key_means_no_call() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ProviderCallFailed):
            await FalImages(http, api_key="", guard=SpendGuard(5)).generate(**args())


@respx.mock
async def test_a_reply_with_no_image_is_an_error() -> None:
    respx.post(RUN).mock(return_value=httpx.Response(200, json={"images": []}))
    async with httpx.AsyncClient() as http:
        with pytest.raises(ProviderCallFailed):
            await FalImages(http, api_key="k", guard=SpendGuard(5)).generate(**args())


# ---- chains: fal is added only by its flag ----------------------------------------------------


def settings(**kw: Any) -> Any:
    from app.core.config import Settings

    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def test_fal_and_openrouter_join_the_chain_only_when_their_flags_are_on() -> None:
    assert load_chain(settings()).image_providers == ["cloudflare"]
    assert load_chain(settings(fal_images=True)).image_providers == ["fal", "cloudflare"]
    both = load_chain(settings(fal_images=True, openrouter_images=True))
    assert both.image_providers == ["openrouter", "fal", "cloudflare"]
    assert both.paid == {"openrouter", "fal"}
    assert load_chain(settings(provider_mode="test")).image_providers == ["cloudflare"]


def test_the_factory_builds_fal_only_when_asked() -> None:
    from app.agents.image_grand.generation.create.factory import build_create_deps

    deps, _ = build_create_deps(settings())
    assert deps.enabled_providers == ["cloudflare"]
    deps2, _ = build_create_deps(settings(fal_images=True, fal_key="k"))
    assert deps2.enabled_providers == ["fal", "cloudflare"]


# ---- sync -------------------------------------------------------------------------------------

LIST = "https://api.fal.ai/v1/models"
PRICES = "https://api.fal.ai/v1/models/pricing"


def fal_registry() -> Registry:
    card = ModelCard(
        id="flux-schnell-fal",
        name="x",
        family="flux",
        status="hidden",
        providers=[ProviderEntry(name="fal", slug="fal-ai/flux/schnell", cost=0.003)],
    )
    return Registry([*load_registry().all(), card])


@respx.mock
async def test_sync_reads_the_public_list_with_paging_and_reports_prices() -> None:
    page1 = httpx.Response(
        200,
        json={
            "models": [
                {
                    "endpoint_id": "fal-ai/flux/schnell",
                    "metadata": {"display_name": "FLUX schnell"},
                },
                {"endpoint_id": "fal-ai/new-one", "metadata": {"display_name": "New One"}},
            ],
            "next_cursor": "Mg==",
            "has_more": True,
        },
    )
    page2 = httpx.Response(
        200,
        json={
            "models": [{"endpoint_id": "fal-ai/another", "metadata": {"display_name": "Another"}}],
            "next_cursor": None,
            "has_more": False,
        },
    )
    route = respx.get(LIST).mock(side_effect=[page1, page2])
    prices = respx.get(PRICES).mock(
        return_value=httpx.Response(
            200,
            json={
                "prices": [
                    {
                        "endpoint_id": "fal-ai/flux/schnell",
                        "unit_price": 0.004,
                        "unit": "image",
                        "currency": "USD",
                    },
                    {
                        "endpoint_id": "fal-ai/new-one",
                        "unit_price": 0.05,
                        "unit": "image",
                        "currency": "USD",
                    },
                    {
                        "endpoint_id": "fal-ai/another",
                        "unit_price": 2.0,
                        "unit": "gpu-second",
                        "currency": "USD",
                    },
                ],
                "next_cursor": None,
                "has_more": False,
            },
        )
    )
    async with httpx.AsyncClient() as http:
        report = await sync_fal(http, registry=fal_registry(), api_key="k")
    assert route.call_count == 2
    assert route.calls[0].request.url.params["category"] == "text-to-image"
    assert route.calls[1].request.url.params["cursor"] == "Mg=="
    assert prices.calls[0].request.headers["authorization"] == "Key k"
    assert report.price_changes == [{"model": "flux-schnell-fal", "old": 0.003, "new": 0.004}]
    assert {m["slug"]: m["cost"] for m in report.new_models} == {
        "fal-ai/new-one": 0.05,
        "fal-ai/another": None,
    }  # a non-image unit is not a per-image price
    assert report.missing == []


def test_megapixel_prices_become_a_price_per_1k_image() -> None:
    from app.gateway.sync import fal_cost_per_image

    assert fal_cost_per_image("image", 0.04) == 0.04
    assert fal_cost_per_image("megapixels", 0.025) == pytest.approx(0.02625)  # x 1.05
    assert fal_cost_per_image("gpu-second", 2.0) is None  # cannot be made a per-image price


@respx.mock
async def test_sync_prices_a_megapixel_model() -> None:
    respx.get(LIST).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [{"endpoint_id": "fal-ai/flux/dev", "metadata": {"display_name": "dev"}}],
                "has_more": False,
            },
        )
    )
    respx.get(PRICES).mock(
        return_value=httpx.Response(
            200,
            json={
                "prices": [
                    {
                        "endpoint_id": "fal-ai/flux/dev",
                        "unit_price": 0.025,
                        "unit": "megapixels",
                        "currency": "USD",
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient() as http:
        report = await sync_fal(http, registry=load_registry(), api_key="k")
    assert report.new_models[0]["cost"] == pytest.approx(0.02625)


@respx.mock
async def test_sync_without_a_key_still_lists_models_and_asks_for_no_prices() -> None:
    respx.get(LIST).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [{"endpoint_id": "fal-ai/x", "metadata": {"display_name": "X"}}],
                "has_more": False,
            },
        )
    )
    prices = respx.get(PRICES).mock(return_value=httpx.Response(401))
    async with httpx.AsyncClient() as http:
        report = await sync_fal(http, registry=load_registry())
    assert not prices.called and "authorization" not in respx.calls[0].request.headers
    assert report.new_models == [{"slug": "fal-ai/x", "name": "X", "cost": None}]


@respx.mock
async def test_sync_reports_registry_models_that_left_fal_and_never_edits_the_registry() -> None:
    respx.get(LIST).mock(return_value=httpx.Response(200, json={"models": [], "has_more": False}))
    reg = fal_registry()
    async with httpx.AsyncClient() as http:
        report = await sync_fal(http, registry=reg)
    assert report.missing == ["flux-schnell-fal"]
    card = reg.get("flux-schnell-fal")
    assert card is not None and card.status == "hidden"


@respx.mock
async def test_a_fal_catalogue_error_is_a_provider_error() -> None:
    respx.get(LIST).mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as http:
        with pytest.raises(ProviderCallFailed):
            await sync_fal(http, registry=load_registry())
