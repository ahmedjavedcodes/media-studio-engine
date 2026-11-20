"""The USD estimate is enforced by the gateway before every paid call (M4). This pins WHERE it
is called from, so it cannot quietly move or be bypassed, and checks it against a real bill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_engine.gateway import budget
from media_engine.gateway.image_registry import image_registry
from media_engine.gateway.router import Gateway
from media_engine.gateway.types import ImageRequest, ModelRef, Route

SRC = Path(budget.__file__).resolve().parents[1]  # src/media_engine
FIXTURE = Path(__file__).parents[2] / "fixtures" / "openrouter" / "gpt-image-2.images.json"


def test_the_estimate_is_declared_enforced():
    assert budget.USD_ESTIMATE_ENFORCED is True
    assert "ENFORCED" in (budget.__doc__ or "")


def test_only_the_gateway_router_calls_the_usd_estimate():
    users = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name != "budget.py" and "estimate_image_usd" in path.read_text(encoding="utf-8")
    )
    # the router checks every paid call; the supervisor gets its step-guard figure injected
    assert users == ["gateway/router.py"]
    supervisor = (SRC / "agents/image_generation/create/agent.py").read_text(encoding="utf-8")
    assert "image_registry" not in supervisor  # the price lookup is injected, not imported


def test_the_estimate_matches_a_real_bill():
    billed = json.loads(FIXTURE.read_text())["body"]["usage"]["cost"]
    spec = image_registry().get("openai/gpt-image-2")
    assert spec is not None
    estimate = budget.estimate_image_usd(spec, ImageRequest(prompt="x", quality="low"))
    assert estimate == pytest.approx(billed, rel=0.05)  # $0.006 against $0.00596


def _check(slug: str, cap: float | None, **request: object) -> Route:
    ref = ModelRef("openrouter", slug)
    decided = Route(slug)
    return Gateway._affordable(decided, ref, ImageRequest(prompt="x", **request), cap)  # type: ignore[arg-type]


def test_a_call_over_the_cap_is_refused_before_it_is_made():
    route = _check("openai/gpt-image-2", 0.05, quality="high")  # estimate $0.211
    assert route.model is None and "over the $0.0500 left" in (route.reason or "")


def test_a_call_within_the_cap_goes_ahead():
    assert _check("openai/gpt-image-2", 0.05, quality="low").model == "openai/gpt-image-2"


def test_an_unknown_price_cannot_be_afforded():
    route = _check("some/unregistered-model", 100.0)
    assert route.model is None and "price unknown" in (route.reason or "")


def test_no_cap_given_means_the_plan_guard_alone_decides():
    assert _check("openai/gpt-image-2", None, quality="high").model == "openai/gpt-image-2"


def test_cloudflare_is_never_priced_in_dollars():
    decided = Route("@cf/black-forest-labs/flux-1-schnell")
    ref = ModelRef("cloudflare", decided.model or "")
    assert Gateway._affordable(decided, ref, ImageRequest(prompt="x"), 0.0) is decided
