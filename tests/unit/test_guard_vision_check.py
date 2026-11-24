import json
import math

import httpx
import pytest
import respx

from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.gateway.providers.cloudflare import CloudflareClient, CloudflareVisionStandIn
from app.gateway.providers.openrouter import OpenRouterLlamaGuard
from app.guardrails.image_prep import PreparedImage
from app.guardrails.schemas import Category, Decision, GuardVerdict
from app.guardrails.vision_check import (
    check_vision_batch,
    combine_batches,
    p_unsafe,
    parse_guard_text,
    verdict_from_output,
)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
IMG = PreparedImage("data:image/jpeg;base64,AAAA", "0" * 64, 10, 10)


def lp(p_unsafe_value: float) -> list[tuple[str, float]]:
    return [("unsafe", math.log(p_unsafe_value)), ("safe", math.log(1 - p_unsafe_value))]


def chat_body(text: str, p: float | None) -> dict[str, object]:
    choice: dict[str, object] = {"message": {"role": "assistant", "content": text}}
    if p is not None:
        first = "unsafe" if text.startswith("unsafe") else "safe"
        choice["logprobs"] = {
            "content": [
                {
                    "token": first,
                    "logprob": math.log(p if first == "unsafe" else 1 - p),
                    "top_logprobs": [
                        {"token": "unsafe", "logprob": math.log(p)},
                        {"token": "safe", "logprob": math.log(1 - p)},
                    ],
                }
            ]
        }
    return {"choices": [choice]}


def openrouter(client: httpx.AsyncClient) -> OpenRouterLlamaGuard:
    return OpenRouterLlamaGuard(client, api_key="sk", base_url="https://openrouter.ai/api/v1")


def test_parse_safe_and_unsafe() -> None:
    assert parse_guard_text("safe") == (False, [])
    assert parse_guard_text(" Safe\n") == (False, [])
    assert parse_guard_text("unsafe\nS1,S12") == (True, ["S1", "S12"])
    assert parse_guard_text("unsafe\ns4") == (True, ["S4"])
    assert parse_guard_text("unsafe") == (True, [])


@pytest.mark.parametrize(
    "bad", ["", "  ", "maybe", "I cannot help", "unsafe\nnot codes", "safe-ish"]
)
def test_parse_bad_fails_closed(bad: str) -> None:
    with pytest.raises(InvalidModelOutput):
        parse_guard_text(bad)


def test_p_unsafe() -> None:
    assert p_unsafe(lp(0.9)) == pytest.approx(0.9)
    assert p_unsafe(None) is None
    assert p_unsafe([("other", -1.0)]) is None
    assert p_unsafe([("unsafe", -0.1)]) == pytest.approx(1.0)  # only unsafe candidate


@pytest.mark.parametrize(
    ("text", "p", "decision", "category"),
    [
        ("safe", 0.02, Decision.APPROVE, Category.SAFE),
        ("unsafe\nS4", 0.97, Decision.INSTANT_REJECT, Category.CSAM),
        ("unsafe\nS9", 0.7, Decision.STANDARD_REJECT, Category.TERRORISM_WEAPONS),
        ("unsafe\nS3", 0.9, Decision.INSTANT_REJECT, Category.NCII_UNDRESSING),
        ("unsafe\nS12", 0.99, Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT),
        ("unsafe\nS12", 0.55, Decision.APPROVE, Category.SAFE),
        ("unsafe\nS1,S12", 0.9, Decision.STANDARD_REJECT, Category.OTHER_UNSAFE),  # tie: first
        ("unsafe\nS12,S11", 0.9, Decision.INSTANT_REJECT, Category.GORE_SELF_HARM),  # tier wins
        ("unsafe\nS7", 0.8, Decision.STANDARD_REJECT, Category.PII_WATERMARK),
        ("unsafe\nS99", 0.8, Decision.STANDARD_REJECT, Category.OTHER_UNSAFE),
        ("unsafe", 0.8, Decision.STANDARD_REJECT, Category.OTHER_UNSAFE),
    ],
)
def test_verdicts(text: str, p: float, decision: Decision, category: Category) -> None:
    v = verdict_from_output(text, lp(p), model="m", latency_ms=5)
    assert (v.decision, v.category) == (decision, category)
    assert v.needs_review is False


def test_no_logprobs_unsafe_is_standard_reject_with_review() -> None:
    v = verdict_from_output("unsafe\nS4", None, model="m", latency_ms=5)
    assert v.decision is Decision.STANDARD_REJECT  # never instant without a score
    assert v.category is Category.CSAM
    assert v.confidence is None
    assert v.needs_review is True


def test_no_logprobs_safe_approves() -> None:
    v = verdict_from_output("safe", None, model="m", latency_ms=5)
    assert v.decision is Decision.APPROVE


async def test_openrouter_request_and_result() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock:
            route = respx.post(OR_URL).mock(
                return_value=httpx.Response(200, json=chat_body("unsafe\nS4", 0.93))
            )
            v = await check_vision_batch(openrouter(client), "a photo", [IMG, IMG], timeout=1.5)
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "meta-llama/llama-guard-4-12b"
    assert sent["temperature"] == 0
    assert sent["max_tokens"] == 10
    assert sent["logprobs"] is True
    parts = sent["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "a photo"}
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
    assert route.calls[0].request.headers["authorization"] == "Bearer sk"
    assert (v.decision, v.category) == (Decision.INSTANT_REJECT, Category.CSAM)
    assert v.confidence == pytest.approx(0.93)


async def test_openrouter_empty_text_and_no_logprobs() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock:
            route = respx.post(OR_URL).mock(
                return_value=httpx.Response(200, json=chat_body("unsafe\nS12", None))
            )
            v = await check_vision_batch(openrouter(client), "", [IMG], timeout=1.5)
    first_part = json.loads(route.calls[0].request.content)["messages"][0]["content"][0]
    assert first_part["text"] == "(no text)"
    assert v.needs_review and v.decision is Decision.STANDARD_REJECT


@pytest.mark.parametrize("status", [429, 500, 400])
async def test_openrouter_errors(status: int) -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock:
            respx.post(OR_URL).mock(return_value=httpx.Response(status, json={}))
            with pytest.raises(ProviderCallFailed) as ei:
                await check_vision_batch(openrouter(client), "x", [IMG], timeout=1.5)
    assert ei.value.retryable is (status != 400)


async def test_garbage_answer_fails_closed() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock:
            respx.post(OR_URL).mock(return_value=httpx.Response(200, json={"choices": []}))
            with pytest.raises(InvalidModelOutput):
                await check_vision_batch(openrouter(client), "x", [IMG], timeout=1.5)


async def test_cloudflare_standin_plain_response_has_no_confidence() -> None:
    url = "https://api.cloudflare.com/client/v4/accounts/a/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"
    async with httpx.AsyncClient() as client:
        cf = CloudflareClient(client, account_id="a", api_token="t")
        vision = CloudflareVisionStandIn(cf, system_prompt="SYS")
        with respx.mock:
            route = respx.post(url).mock(
                return_value=httpx.Response(
                    200, json={"success": True, "result": {"response": "unsafe\nS12"}}
                )
            )
            v = await check_vision_batch(vision, "text", [IMG], timeout=1.5)
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert "<user_text>" in sent["messages"][1]["content"][0]["text"]
    assert v.decision is Decision.STANDARD_REJECT and v.needs_review


def gv(decision: Decision, cat: Category, conf: float) -> GuardVerdict:
    return GuardVerdict(
        source="vision",
        decision=decision,
        category=cat,
        confidence=conf,
        reason="",
        model="m",
        latency_ms=3,
    )


def test_combine_batches() -> None:
    ok = gv(Decision.APPROVE, Category.SAFE, 0.1)
    bad = gv(Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT, 0.9)
    boom = ProviderCallFailed("x", retryable=True)
    assert combine_batches([ok, ok]).decision is Decision.APPROVE
    assert combine_batches([ok, bad]).category is Category.NSFW_EXPLICIT
    assert combine_batches([boom, bad]).decision is Decision.STANDARD_REJECT  # reject beats failure
    with pytest.raises(ProviderCallFailed):
        combine_batches([ok, boom])
