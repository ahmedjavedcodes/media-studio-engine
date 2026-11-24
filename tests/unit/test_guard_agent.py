import time
from datetime import timedelta

import pytest

from app.core.errors import (
    AccountRestricted,
    BadInput,
    ContentBlocked,
    Cooldown,
    GuardrailUnavailable,
)
from app.guardrails.agent import Timing, check, guard
from app.guardrails.schemas import Category, Decision, GuardInput
from app.local.fake_strikes import FakeStrikeLedger
from tests.unit.guard_helpers import T0, FakeJev, FakeVision, boom, jev_answers, make_deps, png


def inp(
    prompt: str = "a red fox in the snow",
    images: list[bytes] | None = None,
    rid: str = "req-1",
    user: str = "u1",
) -> GuardInput:
    return GuardInput(request_id=rid, user_id=user, prompt=prompt, images=images or [])


# ---- approve ----------------------------------------------------------------------------------


async def test_text_only_approves_and_skips_vision() -> None:
    jev, vision = FakeJev(), FakeVision()
    d = await check(inp(), make_deps(jev, vision))
    assert d.decision is Decision.APPROVE and d.category is Category.SAFE
    assert jev.calls == 1 and vision.calls == []
    assert [v.source for v in d.verdicts] == ["jev"]
    assert d.account_action == "none"


async def test_image_request_runs_both_with_right_inputs() -> None:
    jev, vision = FakeJev(), FakeVision()
    d = await check(inp("N u d e fox", [png()]), make_deps(jev, vision))
    assert d.decision is Decision.APPROVE
    assert {v.source for v in d.verdicts} == {"jev", "vision"}
    assert jev.states[0] == {
        "prompt_raw": "N u d e fox",
        "prompt_clean": "nude fox",
        "has_image": True,
    }
    assert vision.calls[0]["text"] == "nude fox"  # vision gets the cleaned text
    assert vision.calls[0]["image_urls"][0].startswith("data:image/jpeg;base64,")


async def test_checks_run_in_parallel() -> None:
    deps = make_deps(FakeJev(delay=0.3), FakeVision(delay=0.3))
    start = time.monotonic()
    await check(inp(images=[png()]), deps)
    assert time.monotonic() - start < 0.5


async def test_images_over_five_go_in_parallel_batches() -> None:
    vision = FakeVision(delay=0.2)
    start = time.monotonic()
    await check(inp(images=[png()] * 7), make_deps(vision=vision))
    assert len(vision.calls) == 2
    assert [
        len(c["image_urls"]) for c in sorted(vision.calls, key=lambda c: -len(c["image_urls"]))
    ] == [5, 2]
    assert time.monotonic() - start < 0.35


# ---- rejects and merge ------------------------------------------------------------------------


async def test_jev_reject_blocked_by_prompt() -> None:
    jev = FakeJev(jev_answers(nsfw_explicit=0.9))
    d = await check(inp(), make_deps(jev))
    assert (d.decision, d.category) == (Decision.STANDARD_REJECT, Category.NSFW_EXPLICIT)
    assert d.user_message == "Your request asks for nudity or sexual content."


async def test_strictest_wins_across_checks() -> None:
    jev = FakeJev(jev_answers(nsfw_explicit=0.99))
    vision = FakeVision("unsafe\nS4", 0.95)
    d = await check(inp(images=[png()]), make_deps(jev, vision))
    assert (d.decision, d.category) == (Decision.INSTANT_REJECT, Category.CSAM)
    assert len(d.verdicts) == 2


async def test_instant_cancels_other_check() -> None:
    jev = FakeJev(jev_answers(csam=0.97))
    vision = FakeVision(delay=5)
    start = time.monotonic()
    d = await check(inp(images=[png()]), make_deps(jev, vision))
    assert d.decision is Decision.INSTANT_REJECT
    assert time.monotonic() - start < 0.5
    assert vision.cancelled


async def test_reject_beats_failure_in_other_check() -> None:
    jev = FakeJev(errors=[boom(False)])
    vision = FakeVision("unsafe\nS12", 0.9)
    d = await check(inp(images=[png()]), make_deps(jev, vision))
    assert d.decision is Decision.STANDARD_REJECT and d.category is Category.NSFW_EXPLICIT


async def test_vision_no_logprobs_rejects_with_review() -> None:
    d = await check(inp(images=[png()]), make_deps(vision=FakeVision("unsafe\nS12", None)))
    assert d.decision is Decision.STANDARD_REJECT
    assert d.verdicts[-1].needs_review or d.verdicts[0].needs_review


# ---- failures: always fail closed -------------------------------------------------------------


@pytest.mark.parametrize("which", ["jev", "vision"])
async def test_failure_without_reject_is_503_and_no_strike(which: str) -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    jev = FakeJev(errors=[boom(False)] if which == "jev" else None)
    vision = FakeVision(errors=[boom(False)] if which == "vision" else None)
    with pytest.raises(GuardrailUnavailable) as ei:
        await check(inp(images=[png()]), make_deps(jev, vision, ledger))
    assert ei.value.http_status == 503
    assert ei.value.message == "Safety verification timed out. Please try again."
    assert (
        await ledger.add_reject("other", "x", "standard", 10)
    ).total == 1  # nothing recorded for u1
    assert (await ledger.get_account("u1")).cooldown_until is None


async def test_invalid_jev_output_is_503() -> None:
    with pytest.raises(GuardrailUnavailable):
        await check(inp(), make_deps(FakeJev({"csam": {"type": "noul"}})))


async def test_retry_then_success() -> None:
    jev = FakeJev(errors=[boom(True)])
    timing = Timing(jev_retry_delay_s=0.01)
    d = await check(inp(), make_deps(jev, timing=timing))
    assert d.decision is Decision.APPROVE and jev.calls == 2


async def test_no_retry_on_non_retryable() -> None:
    jev = FakeJev(errors=[boom(False)])
    with pytest.raises(GuardrailUnavailable):
        await check(inp(), make_deps(jev))
    assert jev.calls == 1


async def test_deadline_never_exceeded() -> None:
    deps = make_deps(FakeJev(delay=10), FakeVision(delay=10))
    start = time.monotonic()
    with pytest.raises(GuardrailUnavailable):
        await check(inp(images=[png()]), deps)
    assert time.monotonic() - start < 2.05


async def test_hung_check_with_reject_from_other_returns_reject() -> None:
    deps = make_deps(
        FakeJev(jev_answers(nsfw_explicit=0.9)), FakeVision(delay=10), timing=Timing(deadline_s=0.3)
    )
    d = await check(inp(images=[png()]), deps)
    assert d.decision is Decision.STANDARD_REJECT


async def test_ledger_failure_on_account_check_fails_closed() -> None:
    class BrokenLedger(FakeStrikeLedger):
        async def get_account(self, user_id: str):  # type: ignore[no-untyped-def,override]
            raise RuntimeError("db down")

    with pytest.raises(GuardrailUnavailable):
        await check(inp(), make_deps(ledger=BrokenLedger()))


# ---- step 0 and bad input ---------------------------------------------------------------------


async def test_banned_account_is_403_and_models_not_called() -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    await ledger.ban("u1", "test")
    jev = FakeJev()
    with pytest.raises(AccountRestricted) as ei:
        await check(inp(), make_deps(jev, ledger=ledger))
    assert ei.value.http_status == 403 and jev.calls == 0


async def test_cooldown_is_429_with_retry_after_and_not_counted() -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    await ledger.set_cooldown("u1", T0 + timedelta(seconds=45))
    jev = FakeJev()
    with pytest.raises(Cooldown) as ei:
        await check(inp(), make_deps(jev, ledger=ledger))
    assert ei.value.http_status == 429 and ei.value.retry_after == 45 and jev.calls == 0
    assert ei.value.body()["retry_after"] == 45
    assert (await ledger.add_reject("u1", "probe", "standard", 10)).total == 1  # nothing counted


async def test_expired_cooldown_passes() -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    await ledger.set_cooldown("u1", T0 - timedelta(seconds=1))
    d = await check(inp(), make_deps(ledger=ledger))
    assert d.decision is Decision.APPROVE


@pytest.mark.parametrize(
    "bad", [inp("a" * 4001), inp(""), inp("   "), inp(images=[b"junk"]), inp(images=[png()] * 11)]
)
async def test_bad_input_is_400(bad: GuardInput) -> None:
    jev = FakeJev()
    with pytest.raises(BadInput):
        await check(bad, make_deps(jev))
    assert jev.calls == 0


async def test_image_only_request_is_allowed() -> None:
    d = await check(inp("", [png()]), make_deps())
    assert d.decision is Decision.APPROVE


# ---- strikes through the agent ----------------------------------------------------------------


async def test_instant_reject_records_a_strike_and_warns() -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    d = await check(inp(), make_deps(FakeJev(jev_answers(csam=0.97)), ledger=ledger))
    assert (d.account_action, d.strike_count, d.retry_after) == ("warn", 1, None)


async def test_standard_reject_starts_cooldown_and_no_strike() -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    d = await check(inp(), make_deps(FakeJev(jev_answers(nsfw_explicit=0.9)), ledger=ledger))
    assert (d.account_action, d.strike_count, d.retry_after) == ("cooldown_1m", None, 60)
    with pytest.raises(Cooldown):
        await check(inp(rid="req-2"), make_deps(ledger=ledger))


async def test_fifth_instant_strike_bans() -> None:
    ledger = FakeStrikeLedger(now=lambda: T0)
    deps = make_deps(FakeJev(jev_answers(csam=0.97)), ledger=ledger)
    # same clock for every call, so use distinct users' windows: 5 strikes, 5 rejects in 10 min
    for n in range(1, 6):
        d = await check(inp(rid=f"r{n}"), deps)
    assert d.account_action == "ban" and d.strike_count == 5
    with pytest.raises(AccountRestricted):
        await check(inp(rid="r6"), deps)


async def test_replaying_a_request_id_is_one_strike() -> None:
    deps = make_deps(FakeJev(jev_answers(csam=0.97)))
    a = await check(inp(rid="same"), deps)
    b = await check(inp(rid="same"), deps)
    assert a.strike_count == b.strike_count == 1


async def test_ledger_failure_while_striking_still_rejects() -> None:
    class BrokenLedger(FakeStrikeLedger):
        async def add_reject(self, *a, **k):  # type: ignore[no-untyped-def,override]
            raise RuntimeError("db down")

    d = await check(
        inp(), make_deps(FakeJev(jev_answers(csam=0.97)), ledger=BrokenLedger(now=lambda: T0))
    )
    assert d.decision is Decision.INSTANT_REJECT and d.account_action == "none"


# ---- guard(): the door in front of the generator ----------------------------------------------


async def test_guard_returns_same_input_untouched() -> None:
    raw = "  N u d e?  No: A RED fox​"
    image = png()
    original = inp(raw, [image])
    out = await guard(original, make_deps())
    assert out is original
    assert out.prompt == raw
    assert out.images[0] is image and out.images[0] == png()


@pytest.mark.parametrize(
    ("answers", "vision"),
    [
        (jev_answers(nsfw_explicit=0.9), FakeVision()),
        (jev_answers(), FakeVision("unsafe\nS12", 0.9)),
    ],
)
async def test_guard_raises_422_body_on_reject(answers: dict, vision: FakeVision) -> None:  # type: ignore[type-arg]
    with pytest.raises(ContentBlocked) as ei:
        await guard(inp(images=[png()]), make_deps(FakeJev(answers), vision))
    body = ei.value.body()
    assert ei.value.http_status == 422
    assert body["code"] == "content_blocked" and body["category"] == "NSFW_EXPLICIT"
    assert body["blocked_by"] in {"prompt", "image"}
    assert body["message"] and body["tip"]
    assert "reason" not in body and "scores" not in body and "verdicts" not in body


async def test_guard_never_approves_on_failure() -> None:
    with pytest.raises(GuardrailUnavailable):
        await guard(inp(), make_deps(FakeJev(errors=[boom(False)])))
