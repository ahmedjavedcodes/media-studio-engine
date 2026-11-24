import httpx
import pytest
import respx

from app.core.config import Settings
from app.guardrails.deps import build_deps, close_guard_clients, timing_for
from app.guardrails.warmup import warm_up
from tests.unit.guard_helpers import FakeJev, FakeVision, jev_answers, make_deps


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_build_mode_timings() -> None:
    t = timing_for(settings(provider_mode="build"))
    assert (t.jev_timeout_s, t.jev_retry_delay_s) == (0.8, 0.4)
    assert (t.vision_timeout_s, t.vision_retry_delay_s) == (2.5, None)  # no vision retry
    assert (t.deadline_for(False), t.deadline_for(True)) == (1.5, 3.0)


def test_test_mode_jev_matches_build_and_vision_is_configurable() -> None:
    t = timing_for(settings(provider_mode="test"))
    assert (t.jev_timeout_s, t.jev_retry_delay_s) == (0.8, 0.4)
    t = timing_for(
        settings(
            provider_mode="test",
            vision_timeout_s=1.2,
            vision_retry_delay_s=0.5,
            deadline_text_s=1.0,
            deadline_images_s=1.8,
        )
    )
    assert (t.vision_timeout_s, t.vision_retry_delay_s) == (1.2, 0.5)
    assert (t.deadline_text_s, t.deadline_images_s) == (1.0, 1.8)


def test_env_overrides_and_zero_means_no_retry() -> None:
    t = timing_for(settings(jev_timeout_s=0.6, jev_retry_delay_s=0, vision_retry_delay_s=0.3))
    assert t.jev_timeout_s == 0.6 and t.jev_retry_delay_s is None
    assert t.vision_retry_delay_s == 0.3


async def test_build_mode_uses_typesafe_jev_and_one_client_per_provider() -> None:
    deps = build_deps(
        settings(typesafe_api_key="k", cloudflare_account_id="a", cloudflare_api_token="t")
    )
    try:
        assert deps.jev.model == "jev-1.13.0"
        assert deps.vision.model == "@cf/meta/llama-4-scout-17b-16e-instruct"
        jev_client = deps.jev._client  # type: ignore[attr-defined]
        vision_client = deps.vision._cf._client  # type: ignore[attr-defined]
        assert jev_client is not vision_client  # one client per provider
        pool = jev_client._transport._pool  # keep-alive settings
        assert pool._keepalive_expiry == 60
    finally:
        await close_guard_clients()


async def test_warm_up_calls_both_and_never_raises() -> None:
    jev, vision = FakeJev(jev_answers()), FakeVision()
    assert await warm_up(make_deps(jev, vision)) == {"jev": "ok", "vision": "ok"}
    assert jev.calls == 1 and len(vision.calls) == 1
    assert vision.calls[0]["image_urls"][0].startswith("data:image/jpeg;base64,")

    broken = FakeJev(errors=[RuntimeError("down")])
    out = await warm_up(make_deps(broken, FakeVision()))
    assert out == {"jev": "RuntimeError", "vision": "ok"}


async def test_warm_up_sends_one_cheap_jev_question() -> None:
    async with httpx.AsyncClient() as c:
        from app.gateway.providers.typesafe import TypeSafeJev

        deps = make_deps(
            TypeSafeJev(c, api_key="k", base_url="https://api.typesafe.ai"), FakeVision()
        )
        with respx.mock:
            route = respx.post("https://api.typesafe.ai/v1/systemone").mock(
                return_value=httpx.Response(200, json={"answers": {}})
            )
            await warm_up(deps)
    import json

    assert list(json.loads(route.calls[0].request.content)["questions"]) == ["csam"]


@pytest.mark.parametrize("flag", [True, False])
def test_warmup_flag_exists(flag: bool) -> None:
    assert settings(guard_warmup=flag).guard_warmup is flag
