"""The Prompt Writer, the code guard and the whole pipeline with fake providers."""

from pathlib import Path
from typing import Any

import pytest

from app.agents.image_grand.workers import prompt_writer as pw
from app.agents.image_grand.workers.shot_plan import ShotPlan
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.errors import ContentRefused, ProviderCallFailed
from app.core.plans import Plan
from app.gateway.compiler import build_prompt, guard_prompt
from app.gateway.registry import load_registry
from tests.unit.create_helpers import (
    FakeLLM,
    FakeProvider,
    critic_ok,
    good_plan,
    make_deps,
    make_service,
    missing,
    ready,
    written,
)

REG = load_registry()


def card(model_id: str = "seedream-5-flash"):  # type: ignore[no-untyped-def]
    c = REG.get(model_id)
    assert c is not None
    return c


def plan_with(**over: Any) -> ShotPlan:
    return good_plan("perfume bottle").model_copy(update=over)


GOOD = (
    "A faceted glass perfume bottle with a brushed brass cap and a blank label plate on a veined "
    "marble vanity by a window, low golden sun from the left throwing long soft shadows, "
    "a sprig of dried lavender, blurred cream wall, 70mm at f/2.8 at eye level"
)


# ---- the system prompt and the code renderer ----------------------------------------------------


def test_the_writer_prompt_states_range_n_and_ends_with_the_family_guide() -> None:
    c = card()
    text = pw.system_prompt("BASE", c, "4:5", n=3)
    assert text.startswith("BASE") and "42 to 60 words" in text and "n = 3" in text
    assert text.rstrip().endswith(c.prompt_guide.strip())
    assert "Mode: edit" not in text and "Mode: edit" in pw.system_prompt(
        "B", c, "1:1", n=1, has_references=True
    )
    base = (Path(pw.__file__).parents[5] / "prompts/create/txt2img/prompt_writer.md").read_text()
    for needle in (
        "90 percent",
        "blank label plate",
        "ONE change each",
        "never by equipment",
        "glowing edge",
    ):
        assert needle in base, needle


def test_the_template_renders_the_plan_in_the_family_style_and_is_deterministic() -> None:
    plan = plan_with()
    keywords = pw.render_template(card("seedream-5-flash"), plan)
    sentences = pw.render_template(card("flux-2-klein-4b"), plan)
    assert ", " in keywords and ". " not in keywords and "blank label plate" in keywords
    assert sentences.endswith(".") and sentences.count(". ") >= 5
    assert keywords == pw.render_template(card("seedream-5-flash"), plan)
    assert pw.render_template(card(), ShotPlan()) == ""


def test_the_template_adds_text_only_when_it_was_asked_for() -> None:
    texty = good_plan("poster").model_copy(
        update={
            "text": good_plan().text.model_copy(update={"exact": "FRESH ROAST", "placement": "top"})
        }
    )
    out = pw.render_template(card(), texty)
    assert '"FRESH ROAST"' in out and "blank label plate" not in out
    assert '"' not in pw.render_template(card(), good_plan())


def test_each_variation_changes_one_axis() -> None:
    plan = plan_with(variation_axes=["camera angle", "time of day", "prop layout"])
    prompts = pw.render_all(card(), plan, 3)
    assert len(set(prompts)) == 3
    assert (
        "low angle" in prompts[0] and "golden hour" in prompts[1] and "props behind" in prompts[2]
    )
    shared = pw.render_template(card(), plan).split(", ")[0]
    assert all(p.startswith(shared) for p in prompts)  # the subject, then the one change
    assert len(pw.render_all(card(), plan_with(), 1)) == 1
    assert pw.axis_change("lens flare", 1) == "variation 2: change the lens flare"


# ---- the writer's call --------------------------------------------------------------------------


async def write(
    llm: FakeLLM, plan: ShotPlan | None = None, n: int = 1, user: str = "perfume bottle on marble"
):  # type: ignore[no-untyped-def]
    report: dict[str, Any] = {}
    out = await pw.write(
        llm, card(), base_prompt="BASE", plan=plan or plan_with(), user_prompt=user,
        aspect_ratio="1:1", n=n, max_tokens=1200, timeout=5, report=report,
    )  # fmt: skip
    return out, report


async def test_the_writer_returns_one_prompt_per_image() -> None:
    llm = FakeLLM("w", [written(GOOD)])
    out, report = await write(llm)
    assert out == [GOOD] and report["source"] == "llm" and report["attempts"] == 1
    call = llm.calls[0]
    assert call["temperature"] == 0.3 and "<shot_plan>" in call["user"]
    three = FakeLLM("w", [written(GOOD, GOOD + " low angle", GOOD + " dusk")])
    out, _ = await write(three, n=3)
    assert len(out) == 3 and '"axes"' in three.calls[0]["user"]


async def test_missing_prompts_are_filled_by_the_code_renderer() -> None:
    out, _ = await write(FakeLLM("w", [written(GOOD)]), n=3)
    assert out[0] == GOOD and len(out) == 3 and out[1] and out[2]


async def test_a_negated_label_gets_one_corrective_retry() -> None:
    bad = GOOD.replace("a blank label plate", "no label")
    llm = FakeLLM("w", [written(bad), written(GOOD)])
    out, report = await write(llm)
    assert out == [GOOD] and report["retry_reason"] and "blank label plate" in llm.calls[1]["user"]


async def test_unrequested_quotes_are_removed_by_code() -> None:
    quoted = GOOD + ' the word "LUXE"'
    out, _ = await write(FakeLLM("w", [written(quoted), written(quoted)]))
    assert '"' not in out[0]


async def test_if_the_llm_fails_the_code_renders_the_prompts() -> None:
    llm = FakeLLM("w", [ProviderCallFailed("429", retryable=True)])
    out, report = await write(llm)
    assert report["source"] == "template" and out == pw.render_all(card(), plan_with(), 1)


# ---- the code guard -----------------------------------------------------------------------------


def test_banned_phrases_are_stripped_unless_the_user_wrote_them() -> None:
    c = card("flux-2-klein-4b")
    text = "a bottle on a seamless grey background, glowing edge, floating cap, soft light"
    out = guard_prompt(c, text, aspect_ratio="1:1").prompt
    assert "seamless" not in out and "glowing" not in out and "floating" not in out
    kept = guard_prompt(
        c, "a floating island at dusk", aspect_ratio="1:1", user_prompt="a floating island"
    ).prompt
    assert "floating" in kept


def test_code_mode_renders_the_plan_and_ignores_the_written_prompt() -> None:
    c = card("flux-2-klein-4b")
    a = build_prompt(c, good_plan(), "LLM A", mode="code", aspect_ratio="1:1")
    b = build_prompt(c, good_plan(), None, mode="code", aspect_ratio="1:1")
    assert a.prompt == b.prompt and "LLM" not in a.prompt
    llm = build_prompt(c, good_plan(), "my written prompt", mode="llm", aspect_ratio="1:1")
    assert llm.prompt == "my written prompt"
    assert (
        "iced coffee cup"
        in build_prompt(c, good_plan(), None, mode="llm", aspect_ratio="1:1").prompt.lower()
    )


def test_every_written_prompt_is_blocklist_checked() -> None:
    with pytest.raises(ContentRefused):
        build_prompt(
            card("flux-2-klein-4b"), good_plan(), "a naked child", mode="llm", aspect_ratio="1:1"
        )


# ---- the whole pipeline -------------------------------------------------------------------------


def req(
    plan: Plan = Plan.PRO, prompt: str = "iced coffee ad for summer", **settings: Any
) -> GenerationRequest:
    return GenerationRequest(prompt=prompt, plan=plan, settings=ImageSettings(**settings))


async def run(service: Any, run_id: str, r: GenerationRequest) -> list[Any]:
    service.start(run_id, "u1", r)
    return [e async for e in service.stream(run_id)]


def spans(deps: Any, run_id: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sp in deps.tracer.trace_for(run_id).snapshot("done")["spans"]:
        out.setdefault(sp["name"], []).append(sp)
    return out


async def test_n_images_get_n_different_written_prompts(tmp_path: Path) -> None:
    provider = FakeProvider()
    writer = [
        written(
            "iced coffee cup, low angle",
            "iced coffee cup, dusk light",
            "iced coffee cup, props left",
        )
    ]
    deps = make_deps(
        tmp_path, provider=provider, writer=writer, critic=[critic_ok(3)], trace_ui=True
    )
    service = make_service(deps)
    events = await run(service, "w1", req(n=3))
    await service.wait("w1")
    assert events[-1].type == "ready" and len(events[-1].images) == 3
    assert {c["prompt"] for c in provider.calls} == {
        "iced coffee cup, low angle",
        "iced coffee cup, dusk light",
        "iced coffee cup, props left",
    }
    s = spans(deps, "w1")
    assert len(s["art_director"]) == 1 and len(s["prompt_writer"]) == 1
    assert s["prompt_writer"][0]["outputs"]["source"] == "llm"
    assert len(s["code_guard"][0]["outputs"]["variations"]) == 3


async def test_without_a_writer_model_the_code_renders_variations(tmp_path: Path) -> None:
    provider = FakeProvider()
    deps = make_deps(tmp_path, provider=provider, critic=[critic_ok(2)], trace_ui=True)
    service = make_service(deps)
    await run(service, "w2", req(n=2))
    await service.wait("w2")
    assert len({c["prompt"] for c in provider.calls}) == 2
    assert spans(deps, "w2")["prompt_writer"][0]["outputs"]["source"] == "template"


async def test_the_shot_plan_is_traced_and_never_sent_to_the_client(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)], trace_ui=True)
    service = make_service(deps)
    events = await run(service, "w3", req())
    await service.wait("w3")
    director = spans(deps, "w3")["art_director"][0]["outputs"]
    assert director["shot_plan"]["subject"] == "iced coffee cup" and director["look"] == "lifestyle"
    assert director["subject_guess"] and "retrieved" in director
    text = " ".join(e.model_dump_json() for e in events).lower()
    for secret in (
        "shot_plan",
        "realism_checks",
        "variation_axes",
        "lifestyle",
        "art_director",
        "lens",
    ):
        assert secret not in text, secret


async def test_delivered_plans_are_stored_with_the_critics_scores(tmp_path: Path) -> None:
    from app.agents.image_grand.workers.critic import CriticImage, CriticOutput, Scores

    out = CriticOutput(
        images=[
            CriticImage(
                id="img_1",
                scores=Scores(match=9, craft=9, interest=9, technical=9, aesthetic=9, realism=8),
            )
        ],
        highlight="x",
    )
    deps = make_deps(tmp_path, critic=[out])
    before = deps.plans.count("run")
    await run(make_service(deps), "w4", req())
    assert deps.plans.count("run") == before + 1
    hit = deps.plans.similar("an iced coffee cup", subject_type="product", k=10)
    assert any(h.source == "run" and h.aesthetic == 9 for h in hit)


async def test_the_users_look_is_used_and_an_unknown_look_fails_cleanly(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, trace_ui=True)
    service = make_service(deps)
    await run(service, "w5", req(preset="cinematic"))
    await service.wait("w5")
    assert "`cinematic`" in deps.enhancer.calls[0]["system"]  # type: ignore[attr-defined]
    assert spans(deps, "w5")["art_director"][0]["outputs"]["look"] == "cinematic"
    bad = make_deps(tmp_path / "bad")
    events = await run(make_service(bad), "w6", req(preset="nope"))
    assert events[-1].type == "error" and "preset" in (events[-1].message or "").lower()
    assert await bad.credits.balance("u1") == 100


async def test_similar_good_plans_reach_the_director_as_examples(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, trace_ui=True)
    service = make_service(deps)
    await run(service, "w7", req(prompt="perfume bottle on a marble shelf"))
    await service.wait("w7")
    assert "## Good plans for similar prompts" in deps.enhancer.calls[0]["system"]  # type: ignore[attr-defined]
    assert len(spans(deps, "w7")["art_director"][0]["outputs"]["retrieved"]) == 2


async def test_a_project_style_is_reused_when_one_is_approved(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)
    plan = good_plan().model_dump(mode="json")
    pid = deps.plans.add(
        run_id="old",
        index=0,
        subject_type="product",
        look="editorial",
        prompt="x",
        plan=plan,
        aesthetic=9,
        realism=9,
        project_id="brand1",
    )
    deps.plans.approve(pid)
    r = GenerationRequest(
        prompt="a cup", plan=Plan.PRO, settings=ImageSettings(), project_id="brand1"
    )
    events = await run(make_service(deps), "w8", r)
    assert events[-1].type == "ready"
    system = deps.enhancer.calls[0]["system"]  # type: ignore[attr-defined]
    assert "## Project style" in system and "`editorial`" in system
    plain = make_deps(tmp_path / "p")
    await run(make_service(plain), "w9", req())
    assert "## Project style" not in plain.enhancer.calls[0]["system"]  # type: ignore[attr-defined]


async def test_the_question_flow_comes_from_the_plans_missing_list(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = make_service(deps)
    service.start("w10", "u1", req(prompt="make it nice"))
    events = [e async for e in service.stream("w10")]
    assert events[-1].type == "question"
    await service.resume("w10", {"q1": "Clean product photo"})
    final = [e async for e in service.stream("w10")]
    assert final[-1].type == "ready" and len(deps.enhancer.calls) == 2  # type: ignore[attr-defined]


async def test_the_plans_endpoint_lists_the_looks(tmp_path: Path) -> None:
    import httpx

    from app.api.deps import get_create_deps
    from app.main import app

    deps = make_deps(tmp_path)
    app.dependency_overrides[get_create_deps] = lambda: deps
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            res = (await c.get("/v1/plans")).json()
    finally:
        app.dependency_overrides.clear()
    ids = [p["id"] for p in res["presets"]]
    assert ids[0] == "auto" and len(ids) == 13 and {"lifestyle", "studio", "fantasy"} <= set(ids)
