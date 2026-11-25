"""Per-model parameters, routing, best-of and the critic."""

from pathlib import Path
from typing import Any

import pytest

from app.agents.image_grand.workers import critic as critic_worker
from app.agents.image_grand.workers.critic import CriticImage, CriticOutput, Scores
from app.agents.image_grand.workers.shot_plan import ShotPlan as EnhancerOutput
from app.contracts.ports import ImageOptions
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import BadInput
from app.core.plans import Plan
from app.gateway.compiler import guard_prompt
from app.gateway.options import RequestExtras, options_for, quality_for
from app.gateway.providers.openrouter_images import build_images_body
from app.gateway.registry import load_registry
from app.gateway.router import RouteRequest, route_auto, wants_text
from app.tools.sizes import snap_native
from tests.unit.create_helpers import (
    critic_ok,
    make_deps,
    make_service,
)

REG = load_registry()


def card(model_id: str):  # type: ignore[no-untyped-def]
    c = REG.get(model_id)
    assert c is not None
    return c


# ---- item 4: sizes, quality, parameters, routing ------------------------------------------------


def test_snap_native_picks_the_nearest_ratio_then_the_nearest_area() -> None:
    sizes = card("recraft-v4-1").native_sizes
    assert snap_native(sizes, "1:1") == (1024, 1024)
    assert snap_native(sizes, "16:9") == (1344, 768)
    assert snap_native(sizes, "4:5") == (896, 1152)
    assert snap_native(sizes, "21:9") == (1536, 768)  # nearest of Recraft's own list
    assert snap_native([], "1:1") is None


def test_the_guard_uses_a_native_size_when_the_model_has_one() -> None:
    out = guard_prompt(card("nano-banana-pro"), "a red fox, sharp", aspect_ratio="9:16")
    assert (out.width, out.height) == (768, 1376)
    plain = guard_prompt(card("flux-2-klein-4b"), "a red fox, sharp", aspect_ratio="9:16")
    assert plain.width % 16 == 0 and plain.height % 16 == 0
    gpt = guard_prompt(card("gpt-image-2-5-flare"), "a red fox, sharp", aspect_ratio="16:9")
    assert gpt.width % 16 == 0 and gpt.height % 16 == 0 and gpt.width > gpt.height


@pytest.mark.parametrize(
    ("model", "plan", "manual", "expected"),
    [
        ("gpt-image-2-5-flare", Plan.BASIC, False, "low"),
        ("gpt-image-2-5-flare", Plan.PRO, False, "medium"),
        ("gpt-image-2-5-flare", Plan.MAX, True, "medium"),
        ("gpt-image-2-5-sunburst", Plan.PRO, True, "medium"),
        ("gpt-image-2-5-sunburst", Plan.ULTRA, True, "medium"),
        ("gpt-image-2-5-sunburst", Plan.MAX, True, "high"),  # a Max user picks it by hand
        ("gpt-image-2-5-sunburst", Plan.MAX, False, "medium"),  # Auto never goes to high
        ("seedream-5-flash", Plan.PRO, True, None),  # no quality knob
    ],
)
def test_gpt_image_quality_by_plan(
    model: str, plan: Plan, manual: bool, expected: str | None
) -> None:
    assert quality_for(card(model), plan, manual=manual) == expected
    opts = options_for(card(model), RequestExtras(plan=plan, manual=manual))
    assert opts.quality == expected


def test_quality_without_a_plan_is_the_cards_own() -> None:
    assert options_for(card("gpt-image-2-5-flare")).quality == "low"


def test_medium_quality_has_its_own_cost_estimate() -> None:
    c = card("gpt-image-2-5-flare")
    entry = c.providers[0]
    assert c.cost_for(entry, "low") == entry.cost
    assert c.cost_for(entry, "medium") == 0.03


def test_guidance_steps_style_are_sent_only_when_the_card_lists_them() -> None:
    c = card("flux-2-pro").model_copy(
        update={"supported_params": ["guidance"], "params": {"guidance": 3.5, "steps": 40}}
    )
    opts = options_for(c)
    assert opts.guidance == 3.5 and opts.steps is None  # steps has a value but is not listed
    plain = options_for(card("flux-2-pro"))
    assert plain.guidance is None and plain.steps is None and plain.style is None


def test_openrouter_sends_extras_under_the_vendor_not_at_top_level() -> None:
    body = build_images_body(
        "black-forest-labs/flux.2-flex", "p", "1:1", None, ImageOptions(guidance=3.0, steps=30)
    )
    assert body["provider"]["options"]["black-forest-labs"] == {"guidance": 3.0, "steps": 30}
    assert "guidance" not in body and "steps" not in body and "negative_prompt" not in body
    assert "provider" not in build_images_body("x/y", "p", "1:1", None, ImageOptions())


def test_none_of_the_curated_models_sends_unsupported_parameters() -> None:
    for c in REG.all():
        if c.picker:
            assert c.supported_params == [] and not c.supports_negative_prompt, c.id


def route(prompt: str, plan: Plan = Plan.MAX, **kw: Any) -> Any:
    return route_auto(
        routed_registry({}),  # testing models are runnable, so the whole curated list is in play
        RouteRequest(
            prompt=prompt,
            plan=plan,
            aspect_ratio="1:1",
            enabled_providers=["openrouter", "cloudflare"],
            **kw,
        ),
    )


def routed_registry(rank: dict[str, dict[str, list[str]]]):  # type: ignore[no-untyped-def]
    from app.gateway.registry import Registry

    return Registry(REG.all(), run_testing=True, auto_rank=rank)


def test_text_requests_only_go_to_typography_or_dense_text_models() -> None:
    for prompt in (
        'a poster that says "OPEN LATE"',
        "a banner with the headline sale",
        "a sign with text",
    ):
        r = route(prompt)
        assert wants_text(prompt) and r.card.handles_text, (prompt, r.card.id)
    assert not wants_text("a red fox in the snow")


def test_a_vector_logo_still_uses_the_svg_model() -> None:
    assert route("a minimal logo for a bakery").card.id == "recraft-v4-1-vector"


def test_text_with_no_text_model_available_falls_back_and_says_so() -> None:
    r = route_auto(
        REG,
        RouteRequest(
            prompt='a poster that says "OPEN"',
            plan=Plan.BASIC,
            aspect_ratio="1:1",
            enabled_providers=["cloudflare"],
        ),
    )
    assert r.rule.endswith("no_text_model") and not r.card.handles_text


def test_auto_rank_is_tried_first_for_the_default_rule() -> None:
    reg = routed_registry({"product": {"max": ["flux-3", "seedream-5-pro"]}})
    req = RouteRequest(
        prompt="a perfume bottle on marble",
        plan=Plan.MAX,
        aspect_ratio="1:1",
        enabled_providers=["openrouter", "cloudflare"],
    )
    r = route_auto(reg, req)
    assert r.card.id == "flux-3" and r.rule == "default+auto_rank"
    other_plan = RouteRequest(**{**req.__dict__, "plan": Plan.PRO})
    assert route_auto(reg, other_plan).rule == "default"  # no rank for Pro: the static rule runs


def test_auto_rank_never_overrides_a_hard_rule_or_text_capability() -> None:
    reg = routed_registry({"text_in_image": {"max": ["flux-3"]}, "design": {"max": ["flux-3"]}})
    text = route_auto(
        reg,
        RouteRequest(
            prompt='a poster that says "HI"',
            plan=Plan.MAX,
            aspect_ratio="1:1",
            enabled_providers=["openrouter", "cloudflare"],
        ),
    )
    assert text.card.handles_text  # flux-3 is skipped: it is not a text model
    refs = route_auto(
        reg,
        RouteRequest(
            prompt="a perfume bottle",
            plan=Plan.MAX,
            aspect_ratio="1:1",
            enabled_providers=["openrouter", "cloudflare"],
            n_refs=3,
        ),
    )
    assert refs.rule == "merge"


def test_a_stale_auto_rank_id_fails_loudly(tmp_path: Path) -> None:
    src = (Path(__file__).parents[2] / "src/app/gateway/models.yaml").read_text()
    f = tmp_path / "m.yaml"
    f.write_text(src + "\nauto_rank:\n  product:\n    max: [no-such-model]\n")
    with pytest.raises(BadInput):
        load_registry(f)


async def run(service: Any, run_id: str, req: GenerationRequest) -> list[Any]:
    service.start(run_id, "u1", req)
    return [e async for e in service.stream(run_id)]


def req(plan: Plan = Plan.PRO, **settings: Any) -> GenerationRequest:
    return GenerationRequest(
        prompt="iced coffee ad for summer", plan=plan, settings=ImageSettings(**settings)
    )


# ---- every generated image is delivered ----------------------------------------------------------


def critic_low(index: int) -> CriticOutput:
    """The critic rates one image terribly and flags it. It must still be delivered."""
    images = []
    for i in (1, 2, 3):
        bad = i == index
        sc = Scores(
            match=1 if bad else 9,
            craft=1 if bad else 9,
            interest=1 if bad else 9,
            technical=1 if bad else 9,
            aesthetic=1 if bad else 9,
            realism=1 if bad else 9,
        )
        images.append(
            CriticImage(
                id=f"img_{i}", scores=sc, flags=["bad_anatomy", "generic_scene"] if bad else []
            )
        )
    return CriticOutput(images=images, highlight="x")


async def test_the_critic_never_drops_an_image(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_low(2)])
    events = await run(make_service(deps), "d1", req(plan=Plan.MAX, n=3))
    assert events[-1].type == "ready" and len(events[-1].images) == 3
    assert (tmp_path / "d1" / "img_2.jpg").exists()  # the worst-rated image is still there


async def test_an_image_whose_safety_check_could_not_run_is_still_delivered(tmp_path: Path) -> None:
    from app.core.errors import ProviderCallFailed
    from tests.unit.create_helpers import SeqVision

    deps = make_deps(
        tmp_path,
        vision=SeqVision([ProviderCallFailed("429", retryable=True)]),
        critic=[critic_ok(2)],
    )
    events = await run(make_service(deps), "d2", req(n=2))
    assert events[-1].type == "ready" and len(events[-1].images) == 2


async def test_only_an_image_found_unsafe_is_withheld(tmp_path: Path) -> None:
    from tests.unit.create_helpers import SeqVision

    deps = make_deps(tmp_path, vision=SeqVision(["unsafe\nS4", "safe"]), critic=[critic_ok(2)])
    events = await run(make_service(deps), "d3", req(n=2))
    assert events[-1].type == "ready" and len(events[-1].images) == 1


# ---- item 7: the stricter critic ----------------------------------------------------------------


def one(scores: Scores, *flags: str) -> CriticImage:
    return CriticImage(id="img_1", scores=scores, flags=list(flags))  # type: ignore[arg-type]


def test_flags_cap_the_scores_in_code_even_if_the_model_forgot() -> None:
    s = Scores(match=9, craft=9, interest=9, technical=9)
    assert critic_worker.apply_caps(one(s, "unrequested_text")).scores.match == 3  # type: ignore[union-attr]
    assert critic_worker.apply_caps(one(s, "missing_item")).scores.match == 3  # type: ignore[union-attr]
    assert critic_worker.apply_caps(one(s, "bad_anatomy")).scores.craft == 4  # type: ignore[union-attr]
    assert critic_worker.apply_caps(one(s, "plain_scene")).scores.interest == 6  # type: ignore[union-attr]
    low = Scores(match=1, craft=2, interest=3, technical=4)
    assert (
        critic_worker.apply_caps(one(low, "unrequested_text", "bad_anatomy")).scores == low
    )  # never raised


def test_unknown_flags_are_ignored_and_a_string_is_accepted() -> None:
    img = CriticImage.model_validate({"id": "img_1", "flags": ["bad_anatomy", "made_up"]})
    assert img.flags == ["bad_anatomy"]
    assert CriticImage.model_validate({"id": "img_1", "flags": "plain_scene"}).flags == [
        "plain_scene"
    ]


def test_a_flagged_image_loses_the_ranking_to_a_clean_one() -> None:
    out = CriticOutput(
        images=[
            one(Scores(match=10, craft=10, interest=10, technical=10), "unrequested_text"),
            CriticImage(id="img_2", scores=Scores(match=7, craft=7, interest=7, technical=7)),
        ]
    )
    review = critic_worker.rank_images(out, ["img_1", "img_2"])
    assert review.best == "img_2" and review.verdicts["img_1"].flags == ("unrequested_text",)


def test_the_recommendation_may_suggest_edit_or_upscale_as_text_only() -> None:
    def rec(**kw: Any) -> str:
        v = critic_worker.Verdict("img_1", rank=1, **{"match": 8, **kw})
        review = critic_worker.Review({"img_1": v}, "img_1", "sharp light")
        return critic_worker.recommendation(review, ["img_1"])

    assert rec() == "This image matches your prompt well."
    assert rec(flags=("bad_anatomy",)).endswith("Edit could fix the hands or face.")
    assert rec(flags=("unrequested_text",)).endswith("Edit could remove the extra text.")
    assert rec(flags=("missing_item",)).endswith("Edit could add the missing item.")
    assert rec(technical=4).endswith("Upscale could sharpen it.")
    assert "Edit" not in rec(technical=8) and "Upscale" not in rec(technical=8)


def test_the_critic_prompt_states_the_strict_rules() -> None:
    text = (Path(__file__).parents[2] / "prompts/create/txt2img/critic.md").read_text()
    for needle in (
        "3 or lower",
        "4 or lower",
        "5 to 6",
        "unrequested_text",
        "bad_anatomy",
        "plain_scene",
    ):
        assert needle in text


# ---- a 429 from Groq (the enhancer's token-per-minute limit) -------------------------------------


async def test_groq_waits_for_retry_after_once_then_succeeds() -> None:
    import httpx
    import respx

    from app.gateway.providers.groq import GROQ_URL, GroqLLM

    ok = {
        "choices": [{"message": {"content": '{"subject":"x"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    with respx.mock:
        route = respx.post(GROQ_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "0"}),
                httpx.Response(200, json=ok),
            ]
        )
        async with httpx.AsyncClient() as client:
            llm = GroqLLM(client, api_key="k", model="openai/gpt-oss-120b")
            out = await llm.complete_json(
                system="s",
                user="u",
                schema=EnhancerOutput,
                max_tokens=50,
                temperature=0.3,
                timeout=5,
            )
    assert out.subject == "x" and route.call_count == 2


async def test_groq_does_not_wait_for_a_long_retry_after_or_retry_twice() -> None:
    import httpx
    import respx

    from app.core.errors import ProviderCallFailed
    from app.gateway.providers.groq import GROQ_URL, GroqLLM

    with respx.mock:
        slow = respx.post(GROQ_URL).mock(
            return_value=httpx.Response(429, headers={"retry-after": "300"})
        )
        async with httpx.AsyncClient() as client:
            llm = GroqLLM(client, api_key="k", model="openai/gpt-oss-120b")
            with pytest.raises(ProviderCallFailed) as err:
                await llm.complete_json(
                    system="s",
                    user="u",
                    schema=EnhancerOutput,
                    max_tokens=5,
                    temperature=0,
                    timeout=5,
                )
    assert slow.call_count == 1 and err.value.retry_after == 300.0
    with respx.mock:
        twice = respx.post(GROQ_URL).mock(
            return_value=httpx.Response(429, headers={"retry-after": "0"})
        )
        async with httpx.AsyncClient() as client:
            llm = GroqLLM(client, api_key="k", model="openai/gpt-oss-120b")
            with pytest.raises(ProviderCallFailed):
                await llm.complete_json(
                    system="s",
                    user="u",
                    schema=EnhancerOutput,
                    max_tokens=5,
                    temperature=0,
                    timeout=5,
                )
    assert twice.call_count == 2  # one retry, never a loop
