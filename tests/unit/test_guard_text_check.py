import json

import httpx
import pytest
import respx

from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.gateway.providers.cloudflare import CloudflareClient, CloudflareJev
from app.gateway.providers.typesafe import TypeSafeJev
from app.guardrails.clean import clean_text
from app.guardrails.schemas import Category, Decision
from app.guardrails.text_check import QUESTION_CATEGORY, check_text, load_questions

URL = "https://api.typesafe.ai/v1/systemone"


def answers(**over: float) -> dict[str, dict[str, object]]:
    base = {k: 0.01 for k in QUESTION_CATEGORY}
    base.update(over)
    return {k: {"type": "noul", "noul": v} for k, v in base.items()}


async def run_check(body: object, status: int = 200) -> object:
    async with httpx.AsyncClient() as client:
        jev = TypeSafeJev(client, api_key="k", base_url="https://api.typesafe.ai")
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(status, json=body))
            return await check_text(jev, clean_text("a red fox"), has_image=False, timeout=0.5)


def test_questions_file_covers_all_categories() -> None:
    qs = load_questions()
    assert set(qs) == set(QUESTION_CATEGORY)
    for q in qs.values():
        assert q["type"] == "noul"
        assert q["criteria"]["true"] and q["criteria"]["false"]  # allow examples present


async def test_safe_prompt_approves() -> None:
    v = await run_check({"answers": answers()})
    assert v.decision is Decision.APPROVE  # type: ignore[attr-defined]
    assert v.category is Category.SAFE  # type: ignore[attr-defined]


async def test_request_body_shape() -> None:
    async with httpx.AsyncClient() as client:
        jev = TypeSafeJev(client, api_key="secret-key", base_url="https://api.typesafe.ai/")
        with respx.mock:
            route = respx.post(URL).mock(
                return_value=httpx.Response(200, json={"answers": answers()})
            )
            await check_text(jev, clean_text("N u d e"), has_image=True, timeout=0.5)
    req = route.calls[0].request
    sent = json.loads(req.content)
    assert req.headers["authorization"] == "Bearer secret-key"
    assert sent["model"] == "jev-1.13.0"
    assert sent["state"] == {"prompt_raw": "N u d e", "prompt_clean": "nude", "has_image": True}
    assert set(sent["questions"]) == set(QUESTION_CATEGORY)


@pytest.mark.parametrize(
    ("over", "decision", "category"),
    [
        ({"csam": 0.97}, Decision.INSTANT_REJECT, Category.CSAM),
        ({"ncii_undressing": 0.7}, Decision.STANDARD_REJECT, Category.NCII_UNDRESSING),
        ({"nsfw_explicit": 0.99}, Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT),
        ({"nsfw_explicit": 0.5}, Decision.APPROVE, Category.SAFE),
        # strictest wins: instant beats standard even if standard score is higher
        (
            {"nsfw_explicit": 0.99, "terrorism_weapons": 0.9},
            Decision.INSTANT_REJECT,
            Category.TERRORISM_WEAPONS,
        ),
        # same decision: higher tier wins over higher score
        (
            {"nsfw_explicit": 0.99, "gore_self_harm": 0.7},
            Decision.STANDARD_REJECT,
            Category.GORE_SELF_HARM,
        ),
    ],
)
async def test_decisions(over: dict[str, float], decision: Decision, category: Category) -> None:
    v = await run_check({"answers": answers(**over)})
    assert (v.decision, v.category) == (decision, category)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"answers": {}},
        {"answers": {**answers(), "csam": {"type": "noul"}}},
        {"answers": {**answers(), "csam": {"type": "noul", "noul": "0.9"}}},
        {"answers": {**answers(), "csam": {"type": "noul", "noul": 1.5}}},
        {"answers": {**answers(), "csam": {"type": "noul", "noul": True}}},
    ],
)
async def test_missing_or_invalid_fails_closed(bad: object) -> None:
    with pytest.raises((InvalidModelOutput, ProviderCallFailed)):
        await run_check(bad)


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (503, True), (400, False), (401, False)]
)
async def test_http_errors(status: int, retryable: bool) -> None:
    with pytest.raises(ProviderCallFailed) as ei:
        await run_check({"error": "x"}, status)
    assert ei.value.retryable is retryable


async def test_timeout_is_retryable() -> None:
    async with httpx.AsyncClient() as client:
        jev = TypeSafeJev(client, api_key="k", base_url="https://api.typesafe.ai")
        with respx.mock:
            respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
            with pytest.raises(ProviderCallFailed) as ei:
                await check_text(jev, clean_text("x"), has_image=False, timeout=0.5)
    assert ei.value.retryable


async def test_cloudflare_jev() -> None:
    cf_url = "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/typesafe/jev"
    async with httpx.AsyncClient() as client:
        jev = CloudflareJev(CloudflareClient(client, account_id="acc", api_token="tok"))
        with respx.mock:
            route = respx.post(cf_url).mock(
                return_value=httpx.Response(
                    200, json={"success": True, "result": {"answers": answers()}}
                )
            )
            v = await check_text(jev, clean_text("a fox"), has_image=False, timeout=0.5)
    assert v.model == "typesafe/jev"
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"
