"""Registry sync (spec section 6). Reads a provider's model list and reports what changed.

It never edits models.yaml and never makes a model live. It only reports new image models (to be
added as `hidden`) and, for fal, price changes (to correct the estimates). OpenRouter has no
per-output-image price in its catalogue, so its prices come from `usage.cost` after a call. A
person then edits the YAML, and a model goes live only after the real test.

UNVERIFIED for OpenRouter generation until the owner turns it on. The catalogue shape was checked
against the live `GET /models` on 2026-11-25 (data[].id, name, architecture.output_modalities).
fal's list and prices were checked live the same day (`GET /v1/models`, `/v1/models/pricing`).
"""

from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.errors import ProviderCallFailed
from app.gateway.registry import Registry


@dataclass
class SyncReport:
    new_models: list[dict[str, Any]] = field(default_factory=list)
    price_changes: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # in the registry but gone from the provider


def _image_models(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("data", []):
        if not isinstance(item, dict) or "id" not in item:
            continue
        modalities = (item.get("architecture") or {}).get("output_modalities") or []
        if "image" in modalities:
            out[str(item["id"])] = item
    return out


async def sync_openrouter(
    client: httpx.AsyncClient, *, base_url: str, registry: Registry, api_key: str = ""
) -> SyncReport:
    """A free metadata read. No image is generated."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = await client.get(
            f"{base_url.rstrip('/')}/models",
            params={"output_modalities": "image"},
            headers=headers,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise ProviderCallFailed("openrouter catalogue is not reachable", retryable=True) from exc
    if resp.status_code >= 400:
        raise ProviderCallFailed(f"openrouter returned {resp.status_code}", retryable=False)
    remote = _image_models(resp.json())

    report = SyncReport()
    known: dict[str, tuple[str, float]] = {}
    for card in registry.all():
        for p in card.providers:
            if p.name == "openrouter":
                known[p.slug] = (card.id, p.cost)
    for slug, (model_id, _cost) in known.items():
        item = remote.get(slug)
        if item is None:
            report.missing.append(model_id)
            continue
        # No price comparison: OpenRouter's `pricing.image` is the price of an INPUT image (most
        # image models list $0), not of an output image. The real cost is `usage.cost` on each
        # call, which the spend guard records. Checked against the live catalogue on 2026-11-25.
    for slug, item in remote.items():
        if slug not in known:
            report.new_models.append({"slug": slug, "name": item.get("name", slug), "cost": None})
    return report


FAL_API = "https://api.fal.ai/v1"
FAL_MAX_PAGES = 10  # 50 models a page. Enough for the text-to-image category


async def _fal_models(client: httpx.AsyncClient, headers: dict[str, str]) -> dict[str, str]:
    """{endpoint_id: display name} of active text-to-image models. Public, no key needed."""
    out: dict[str, str] = {}
    cursor: str | None = None
    for _ in range(FAL_MAX_PAGES):
        params: dict[str, str | int] = {
            "category": "text-to-image",
            "status": "active",
            "limit": 50,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            resp = await client.get(f"{FAL_API}/models", params=params, headers=headers, timeout=30)
        except httpx.HTTPError as exc:
            raise ProviderCallFailed("fal catalogue is not reachable", retryable=True) from exc
        if resp.status_code >= 400:
            raise ProviderCallFailed(f"fal returned {resp.status_code}", retryable=False)
        body = resp.json()
        for item in body.get("models", []):
            meta = item.get("metadata") or {}
            out[str(item["endpoint_id"])] = str(meta.get("display_name", item["endpoint_id"]))
        cursor = body.get("next_cursor")
        if not body.get("has_more") or not cursor:
            break
    return out


MEGAPIXELS_PER_1K_IMAGE = 1.05  # 1024 x 1024 = 1.048576 MP. fal may round a call up: check it


def fal_cost_per_image(unit: str, unit_price: float) -> float | None:
    """USD for one 1K image, or None when the unit is not per image or per megapixel (GPU
    seconds and so on can't be turned into a per-image price here)."""
    if unit == "image":
        return unit_price
    if unit == "megapixels":
        return unit_price * MEGAPIXELS_PER_1K_IMAGE
    return None


async def _fal_prices(
    client: httpx.AsyncClient, endpoint_ids: list[str], headers: dict[str, str]
) -> dict[str, float]:
    """USD per 1K image. Needs a key. Up to 50 endpoints a call. A failed batch is skipped."""
    prices: dict[str, float] = {}
    for start in range(0, len(endpoint_ids), 50):
        batch = endpoint_ids[start : start + 50]
        try:
            resp = await client.get(
                f"{FAL_API}/models/pricing",
                params={"endpoint_id": batch},
                headers=headers,
                timeout=30,
            )
        except httpx.HTTPError:
            continue
        if resp.status_code >= 400:
            continue
        for row in resp.json().get("prices", []):
            if row.get("currency") != "USD":
                continue
            cost = fal_cost_per_image(str(row.get("unit")), float(row["unit_price"]))
            if cost is not None:
                prices[str(row["endpoint_id"])] = cost
    return prices


async def sync_fal(
    client: httpx.AsyncClient, *, registry: Registry, api_key: str = ""
) -> SyncReport:
    """A free metadata read of fal's text-to-image models. The list is public. Prices need a key,
    so without one the report has names only. Nothing is generated and nothing is edited."""
    headers = {"Authorization": f"Key {api_key}"} if api_key else {}
    remote = await _fal_models(client, headers)
    known: dict[str, tuple[str, float]] = {}
    for card in registry.all():
        for p in card.providers:
            if p.name == "fal":
                known[p.slug] = (card.id, p.cost)
    prices = await _fal_prices(client, sorted(remote), headers) if api_key else {}

    report = SyncReport()
    for slug, (model_id, cost) in known.items():
        if slug not in remote:
            report.missing.append(model_id)
        elif slug in prices and abs(prices[slug] - cost) > 1e-9:
            report.price_changes.append({"model": model_id, "old": cost, "new": prices[slug]})
    for slug, name in sorted(remote.items()):
        if slug not in known:
            report.new_models.append({"slug": slug, "name": name, "cost": prices.get(slug)})
    return report
