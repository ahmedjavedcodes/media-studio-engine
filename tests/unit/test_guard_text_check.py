import json

import httpx
import pytest
import respx

from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.gateway.providers.cloudflare import CloudflareClient, CloudflareJev
from app.gateway.providers.typesafe import TypeSafeJev
from app.guardrails.clean import clean_text
from app.guardrails.schemas import Category, Decision
from app.guardrails.text_check import check_text, load_questions
from tests.unit.guard_helpers import ALIAS

URL = "https://api.typesafe.ai/v1/systemone"


def answers(**over: float) -> dict[str, dict[str, object]]:
    base = {k: 0.01 for k in load_questions().questions}
    base.update({ALIAS.get(k, k): v for k, v in over.items()})
    return {k: {"type": "noul", "noul": v} for k, v in base.items()}


async def run_check(body: object, status: int = 200) -> object:
    async with httpx.AsyncClient() as client:
        jev = TypeSafeJev(client, api_key="k", base_url="https://api.typesafe.ai")
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(status, json=body))
            return await check_text(jev, clean_text("a red fox"), has_image=False, timeout=0.5)


def test_questions_file_covers_all_categories() -> None:
    qs = load_questions()
    assert {c for c in qs.category_of.values() if c} == set(Category) - {
        Category.SAFE,
        Category.OTHER_UNSAFE,  # vision only
    }
    assert qs.category_of["involves_minor"] is None  # helper, never rejects alone
    assert qs.context
    for q in qs.questions.values():
        assert q["type"] == "noul" and q["instructions"]
    assert qs.questions["sexual_content"]["criteria"] == {
        "true": qs.questions["sexual_content"]["criteria"]["true"],
        "false": qs.questions["sexual_content"]["criteria"]["false"],
    }
    assert "criteria" not in qs.questions["involves_minor"]  # both sides missing -> omitted


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
    state = sent["state"]
    assert state["context"] == load_questions().context
    assert {k: state[k] for k in ("prompt_raw", "prompt_clean", "has_image")} == {
        "prompt_raw": "N u d e",
        "prompt_clean": "nude",
        "has_image": True,
    }
    assert set(sent["questions"]) == set(load_questions().questions)


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


async def test_highest_question_score_per_category_wins() -> None:
    # two questions feed GORE_SELF_HARM: the higher one decides
    v = await run_check({"answers": answers(self_harm=0.9, graphic_gore=0.1)})
    assert (v.decision, v.category) == (Decision.INSTANT_REJECT, Category.GORE_SELF_HARM)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("over", "decision"),
    [
        ({"involves_minor": 0.99}, Decision.APPROVE),  # helper alone never rejects
        ({"involves_minor": 0.95, "sexual_content": 0.95}, Decision.INSTANT_REJECT),  # combo
        ({"involves_minor": 0.95, "undress_real_person": 0.9}, Decision.INSTANT_REJECT),
        ({"involves_minor": 0.1, "sexual_content": 0.95}, Decision.STANDARD_REJECT),  # adult nsfw
        (
            {"involves_minor": 0.95, "sexual_content": 0.1},
            Decision.APPROVE,
        ),  # child, nothing sexual
    ],
)
async def test_csam_combo_rule(over: dict[str, float], decision: Decision) -> None:
    v = await run_check({"answers": answers(**over)})
    assert v.decision is decision  # type: ignore[attr-defined]
    if decision is Decision.INSTANT_REJECT:
        assert v.category is Category.CSAM  # type: ignore[attr-defined]
