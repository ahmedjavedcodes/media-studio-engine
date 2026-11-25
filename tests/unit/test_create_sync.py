import httpx
import respx

from app.gateway.registry import load_registry
from app.gateway.sync import sync_openrouter

BASE = "https://openrouter.ai/api/v1"


@respx.mock
async def test_sync_reports_new_models_and_never_trusts_the_input_image_price() -> None:
    route = respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "google/gemini-3.1-flash-image",
                        "name": "Nano Banana 2",
                        "pricing": {"image": "0.08"},
                        "architecture": {"output_modalities": ["image", "text"]},
                    },
                    {
                        "id": "acme/new-image",
                        "name": "New",
                        "pricing": {"image": "0.01"},
                        "architecture": {"output_modalities": ["image"]},
                    },
                    {"id": "acme/text-only", "architecture": {"output_modalities": ["text"]}},
                ]
            },
        )
    )
    reg = load_registry()
    async with httpx.AsyncClient() as client:
        report = await sync_openrouter(client, base_url=BASE, registry=reg)
    assert route.calls[0].request.url.params["output_modalities"] == "image"
    # `pricing.image` is the price of an INPUT image. It is not an output price, so no change is
    # reported and no cost is shown for the new model.
    assert report.price_changes == []
    assert report.new_models == [{"slug": "acme/new-image", "name": "New", "cost": None}]
    # every other OpenRouter model in the registry is not in this tiny fake catalogue
    assert "nano-banana-2" not in report.missing  # it IS in the fake catalogue
    assert "gpt-image-2-5-flare" in report.missing
    card = reg.get("nano-banana-2")
    assert card is not None and card.status == "testing"  # a sync never makes a model live


@respx.mock
async def test_models_gone_from_the_provider_are_reported() -> None:
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    async with httpx.AsyncClient() as client:
        report = await sync_openrouter(client, base_url=BASE, registry=load_registry())
    assert "nano-banana-2" in report.missing and "flux-2-klein-4b" in report.missing
