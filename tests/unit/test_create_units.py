"""Small pure pieces: plans, sizes, imaging, registry, critic ranking, credits, checks."""

from pathlib import Path
from typing import Any

import pytest

from app.agents.image_grand.workers.critic import (
    CriticImage,
    CriticOutput,
    Review,
    Scores,
    Verdict,
    rank_images,
    recommendation,
)
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import InvalidModelOutput, OutOfCredits
from app.core.plans import PLAN_LIMITS, Plan, Tier, credits_for, tier_allowed, tier_for_cost
from app.gateway.registry import ModelCard, ProviderEntry, Registry, load_registry
from app.local.fake_credits import FakeCredits
from app.local.output_store import OutputStore
from app.tools.imaging import make_thumbnail, sniff
from app.tools.sizes import dimensions, parse_ratio
from tests.unit.create_helpers import jpeg, make_deps, make_service, png


def test_plan_limits_match_the_spec() -> None:
    assert [PLAN_LIMITS[p].max_n for p in Plan] == [2, 4, 4, 4]
    assert [PLAN_LIMITS[p].max_resolution for p in Plan] == ["1K", "2K", "2K", "2K"]


@pytest.mark.parametrize(
    "cost,tier",
    [
        (0.001, Tier.BASIC),
        (0.015, Tier.PRO),
        (0.039, Tier.PRO),
        (0.04, Tier.ULTRA),
        (0.089, Tier.ULTRA),
        (0.09, Tier.MAX),
        (0.5, Tier.MAX),
    ],
)
def test_tier_comes_from_the_cost(cost: float, tier: Tier) -> None:
    assert tier_for_cost(cost) == tier


def test_a_plan_uses_its_own_tier_and_everything_below() -> None:
    # Free (Basic) users get models up to the Pro tier. Pro and above can use any model
    assert tier_allowed(Plan.BASIC, Tier.BASIC) and tier_allowed(Plan.BASIC, Tier.PRO)
    assert not tier_allowed(Plan.BASIC, Tier.ULTRA) and not tier_allowed(Plan.BASIC, Tier.MAX)
    assert all(tier_allowed(p, t) for p in (Plan.PRO, Plan.ULTRA, Plan.MAX) for t in Tier)
    assert credits_for(Tier.PRO, 4) == 12


def test_model_tier_is_computed_never_written() -> None:
    card = ModelCard(
        id="x",
        name="X",
        family="flux",
        providers=[
            ProviderEntry(name="a", slug="s", cost=0.05),
            ProviderEntry(name="b", slug="s", cost=0.2),
        ],
    )
    assert card.tier == Tier.ULTRA and card.cost_per_image == 0.05
    assert [p.name for p in card.providers_in(["b", "a"])] == ["a", "b"]  # cheapest first


def test_registry_loads_with_live_cloudflare_models_and_testing_openrouter_ones() -> None:
    reg = load_registry()
    live = {c.id for c in reg.live(["cloudflare"])}
    assert {"flux-1-schnell", "flux-2-klein-4b", "flux-2-klein-9b", "phoenix-1"} <= live
    assert "lucid-origin" not in live  # not verified with a real call yet
    assert "nano-banana-2" not in live
    assert reg.live([]) == []  # no provider, no model
    assert {c.id for c in reg.allowed(Plan.BASIC, ["cloudflare"])} <= live
    assert all(c.tier in (Tier.BASIC, Tier.PRO) for c in reg.allowed(Plan.BASIC, ["cloudflare"]))


@pytest.mark.parametrize("ratio", ["1:1", "4:5", "5:4", "9:16", "16:9", "21:9"])
def test_sizes_are_about_one_megapixel_and_a_multiple(ratio: str) -> None:
    w, h = dimensions(ratio, "1K", 16)
    assert w % 16 == 0 and h % 16 == 0
    assert 0.9 < (w * h) / 1024**2 < 1.1
    a, b = parse_ratio(ratio)
    assert abs(w / h - a / b) < 0.05


def test_bad_ratio_raises() -> None:
    with pytest.raises(ValueError):
        parse_ratio("wide")


def test_file_type_is_read_from_the_bytes() -> None:
    assert sniff(jpeg()).mime == "image/jpeg" and sniff(jpeg()).ext == "jpg"
    assert sniff(png()).mime == "image/png" and sniff(png()).ext == "png"
    with pytest.raises(InvalidModelOutput):
        sniff(b"not an image")


def test_thumbnail_is_256px_jpeg() -> None:
    thumb = make_thumbnail(png((1024, 512)))
    info = sniff(thumb)
    assert info.mime == "image/jpeg" and max(info.width, info.height) == 256


def test_store_blocks_path_tricks(tmp_path: Path) -> None:
    store = OutputStore(tmp_path)
    store.write_bytes("run1", "a.jpg", b"x")
    assert store.read_bytes("run1", "a.jpg") == b"x"
    assert store.read_bytes("..", "a.jpg") is None
    assert store.read_bytes("run1", "../../etc/passwd") is None
    with pytest.raises(ValueError):
        store.write_bytes("run1", "../evil.jpg", b"x")


def s(m: int, c: int = 8, i: int = 7, t: int = 8) -> Scores:
    return Scores(match=m, craft=c, interest=i, technical=t)


def test_critic_ranks_images_and_unscored_ones_go_last() -> None:
    out = CriticOutput(
        images=[
            CriticImage(id="img_1", scores=s(6)),
            CriticImage(id="img_2", scores=s(9)),
            CriticImage(id="img_3"),  # no scores
        ]
    )
    r = rank_images(out, ["img_1", "img_2", "img_3", "img_4"])  # img_4 was never reviewed
    assert r.best == "img_2"
    assert (r.verdicts["img_2"].rank, r.verdicts["img_1"].rank) == (1, 2)
    assert r.verdicts["img_3"].rank is None and r.verdicts["img_4"].rank is None


def test_the_critic_schema_has_no_safety_field() -> None:
    assert "safe" not in CriticImage.model_fields  # safety is checked per image, before the critic


def test_scores_are_validated_0_to_10() -> None:
    with pytest.raises(ValueError):
        Scores(match=11, craft=1, interest=1, technical=1)


def test_recommendation_line() -> None:
    v = {"img_1": Verdict("img_1", 2, 7), "img_2": Verdict("img_2", 1, 9)}
    r = Review(v, best="img_2", highlight="sharp. ")
    assert recommendation(r, ["img_2", "img_1"]) == "Image 1 matches your prompt best: sharp."
    assert recommendation(Review(v, "img_2", ""), ["img_1", "img_2"]) == (
        "Image 2 matches your prompt best."
    )
    assert recommendation(r, ["img_2"]) == "This image matches your prompt well."
    weak = {"img_1": Verdict("img_1", 1, 3)}
    assert "None of these match" in recommendation(Review(weak, "img_1", ""), ["img_1", "img_2"])
    assert recommendation(Review({}, None, ""), []) == ""
    assert recommendation(Review(v, "img_2", "x", failed=True), ["img_2"]) == ""  # critic failed


async def test_credits_hold_settle_refund_and_release() -> None:
    c = FakeCredits(10)
    hold = await c.hold("u", "r", 6)
    assert await c.balance("u") == 4
    assert await c.settle(hold, 2) == 2 and await c.balance("u") == 8  # refund the rest
    assert await c.settle(hold, 2) == 0  # idempotent
    h2 = await c.hold("u", "r2", 5)
    await c.release(h2)
    await c.release(h2)
    assert await c.balance("u") == 8
    with pytest.raises(OutOfCredits):
        await c.hold("u", "r3", 99)


async def test_a_manual_pick_is_strict(tmp_path: Path) -> None:
    for model, plan, n, ratio, code in [
        ("recraft-v4-1-vector", Plan.BASIC, 1, "1:1", "plan_limit"),  # Ultra model, Basic plan
        ("flux-1-schnell", Plan.BASIC, 1, "16:9", "bad_input"),  # ratio not supported
        ("does-not-exist", Plan.MAX, 1, "1:1", "plan_limit"),
        ("nano-banana-2", Plan.MAX, 1, "1:1", "plan_limit"),  # not live, provider off
    ]:
        svc = make_service(make_deps(tmp_path))
        svc.start(
            "x",
            "u",
            GenerationRequest(
                prompt="a fox",
                plan=plan,
                settings=ImageSettings(model=model, n=n, aspect_ratio=ratio),
            ),
        )
        events = [e async for e in svc.stream("x")]
        assert events[-1].code == code, (model, events[-1])


async def test_2k_is_rejected_never_lowered_silently(tmp_path: Path) -> None:
    for plan, text in ((Plan.BASIC, "not part of your plan"), (Plan.PRO, "coming soon")):
        deps = make_deps(tmp_path)
        svc = make_service(deps)
        svc.start(
            "a",
            "u",
            GenerationRequest(
                prompt="a fox", plan=plan, settings=ImageSettings(resolution="2K", n=1)
            ),
        )
        last = [e async for e in svc.stream("a")][-1]
        assert last.code == "plan_limit" and text in (last.message or "")
        assert deps.gateway._providers["cloudflare"].calls == []  # type: ignore[attr-defined]
        assert await deps.credits.balance("u") == 100


def test_the_registry_has_every_model_the_spec_decided_in_its_tier() -> None:
    """Spec section 6, the tier table. Names and tiers must match."""
    spec = {
        "basic": [
            "Recraft V4.1 Flash",
            "GPT Image 2.5 Flare (low)",
            "FLUX 3 Klein",
            "FLUX.2 klein 4B",
            "Muse Image",
        ],
        "pro": [
            "Seedream 5.0 Flash",
            "FLUX.2 Pro",
            "Qwen Image 3",
            "Nano Banana 2.1",
            "Nano Banana 2 Lite",
            "Recraft V4.1",
            "Seedream 5.0 Lite",
        ],
        # FLUX.3 is in the spec's Pro list, but its real price ($0.048 at 1K) makes it Ultra
        "ultra": [
            "FLUX.2 Flex",
            "Krea 2 Large",
            "Nano Banana 2",
            "FLUX.2 Max",
            "Seedream 5.0 Pro",
            "FLUX.3",
        ],
        "max": [
            "Nano Banana Pro",
            "GPT Image 2.5 Sunburst (high)",
            "Recraft V4.1 Pro",
            "MAI-Image 2.5 Pro",
            "Riverflow 2.5 Pro",
        ],
    }
    reg = load_registry()
    by_name = {c.name: c for c in reg.all()}
    for tier, names in spec.items():
        for name in names:
            assert name in by_name, name
            assert by_name[name].tier.value == tier, (name, by_name[name].tier)


def test_openrouter_models_cannot_run_while_openrouter_is_off() -> None:
    reg = load_registry()
    assert reg.live(["cloudflare"]) == [c for c in reg.live(["cloudflare"]) if c.status == "live"]
    for c in reg.listed():
        if c.status == "testing":
            assert c not in reg.live(["cloudflare", "openrouter"])  # not live until a real run
    # klein 4B is on both providers: Cloudflare first (cheaper), OpenRouter only as a fallback
    klein = reg.get("flux-2-klein-4b")
    assert klein is not None
    assert [p.name for p in klein.providers_in(["cloudflare", "openrouter"])] == [
        "cloudflare",
        "openrouter",
    ]
    assert [p.name for p in klein.providers_in(["cloudflare"])] == ["cloudflare"]


def test_every_family_has_a_label_and_a_guide() -> None:
    for c in load_registry().all():
        assert c.family_label and c.prompt_guide, c.id


SHOW = {
    "basic": ["flux-2-klein-4b", "gpt-image-2-5-flare", "recraft-v4-1-flash", "flux-3-klein"],
    "pro": ["seedream-5-flash", "flux-2-pro", "nano-banana-2-1", "recraft-v4-1"],
    "ultra": ["seedream-5-pro", "flux-3", "recraft-v4-1-vector"],
    "max": ["gpt-image-2-5-sunburst", "nano-banana-pro", "recraft-v4-1-pro"],
}
FREE_ENGINES = ["flux-1-schnell", "flux-2-klein-9b", "phoenix-1", "lucid-origin"]


def test_the_picker_list_is_the_curated_14_in_their_tiers() -> None:
    reg = load_registry()
    assert sorted(c.id for c in reg.all() if c.picker) == sorted(
        m for ids in SHOW.values() for m in ids
    )
    for tier, ids in SHOW.items():
        for model_id in ids:  # the tier comes from the real price, and matches the plan list
            assert reg.get(model_id).tier.value == tier, model_id  # type: ignore[union-attr]
    assert [c.id for c in reg.listed()].__len__() == 14


def test_a_plan_gets_its_own_models_and_every_model_below_it() -> None:
    reg = load_registry()
    order = list(SHOW)
    both = ["cloudflare", "openrouter"]
    live = Registry([c.model_copy(update={"status": "live"}) for c in reg.all()])
    for plan in Plan:  # Basic gets up to the Pro tier. Pro and above can use any model
        allowed = {c.id for c in live.allowed(plan, both) if c.picker}
        tiers = order[:2] if plan == Plan.BASIC else order
        assert allowed == {m for t in tiers for m in SHOW[t]}, plan


def test_auto_picks_only_the_curated_models_and_the_free_engines() -> None:
    reg = load_registry()
    assert sorted(c.id for c in reg.all() if c.auto) == sorted(
        [m for ids in SHOW.values() for m in ids] + FREE_ENGINES
    )
    # the older and overlapping models are not Auto's to pick
    for model_id in (
        "nano-banana-2",
        "gpt-image-2",
        "gpt-image-2-6-flare",
        "seedream-5-5",
        "recraft-v4",
        "flux-2-flex",
        "muse-image",
        "recraft-v4-1-flash-free",
    ):
        assert not reg.get(model_id).auto, model_id  # type: ignore[union-attr]


def test_only_the_cloudflare_model_is_in_the_picker_of_the_free_engines() -> None:
    reg = load_registry()
    in_picker = {
        c.id for c in reg.all() if c.picker and "cloudflare" in {p.name for p in c.providers}
    }
    assert in_picker == {"flux-2-klein-4b"}


def test_hidden_models_can_still_be_picked_by_id_when_live() -> None:
    """Hidden = not in the picker. It does not mean unusable: internal tests pick by id."""
    reg = load_registry()
    live = Registry([c.model_copy(update={"status": "live"}) for c in reg.all()])
    hidden = live.get("gpt-image-2")
    assert hidden is not None and not hidden.picker
    assert hidden in live.live(["cloudflare", "openrouter"])
    assert hidden not in live.listed()


async def test_a_manual_pick_of_a_hidden_model_passes_the_checks(tmp_path: Path) -> None:
    from app.agents.image_grand.generation.create.checks import run_code_checks

    deps = make_deps(tmp_path)
    deps.registry = Registry(
        [c.model_copy(update={"status": "live"}) for c in load_registry().all()]
    )
    deps.enabled_providers = ["cloudflare", "openrouter"]
    state: dict[str, Any] = {
        "plan": "max",
        "n_refs": 0,
        "settings": {"model": "gpt-image-2", "aspect_ratio": "1:1", "resolution": "1K", "n": 1},
    }
    await run_code_checks(deps, state)  # type: ignore[arg-type]  # no error: a hidden model is allowed
