"""End to end txt2img with fake providers: events, credits, interrupt, failures."""

from pathlib import Path

from app.contracts.request import GenerationRequest, ImageSettings
from app.core.plans import Plan
from tests.unit.create_helpers import (
    FakeProvider,
    SeqVision,
    critic_ok,
    dump,
    make_deps,
    make_service,
    missing,
    ready,
)


def request(prompt: str = "iced coffee ad for summer", **settings: object) -> GenerationRequest:
    plan = settings.pop("plan", Plan.PRO)
    return GenerationRequest(
        prompt=prompt,
        plan=plan,
        settings=ImageSettings(**settings),  # type: ignore[arg-type]
    )


async def run(service, run_id: str, req: GenerationRequest):  # type: ignore[no-untyped-def]
    service.start(run_id, "u1", req)
    return [e async for e in service.stream(run_id)]


async def test_full_run_streams_status_images_ready(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2, best=2)])
    service = make_service(deps)
    events = await run(service, "r1", request(n=2))
    kinds = [e.type for e in events]
    assert "image" not in kinds  # images are not shown before the safety check
    assert kinds[-1] == "ready"
    labels = [e.label for e in events if e.type == "status"]
    assert labels[0] == "Understanding your request"
    assert "Thinking" in labels and "Compiling" in labels and "Final touches" in labels
    assert any(e.progress == "2/2" for e in events if e.type == "status")
    ready_event = events[-1]
    assert len(ready_event.images) == 2
    assert ready_event.recommendation  # every plan gets the line
    # user paid for 2 images of the chosen model's tier
    assert await deps.credits.balance("u1") < 100


async def test_events_never_expose_the_pipeline(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2)])
    service = make_service(deps)
    events = await run(service, "r2", request(n=2))
    text = dump(events).lower()
    for secret in (
        "flux",
        "lucid",
        "phoenix",
        "cloudflare",
        "enhancer",
        "critic",
        "score",
        "trace",
        "langsmith",
        "seed",
        "iced coffee cup on a marble",
        "model",
    ):
        assert secret not in text, secret


async def test_free_plan_gets_recommendation_but_keeps_generation_order(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2, best=2)])
    service = make_service(deps)
    events = await run(service, "r3", request(plan=Plan.BASIC, n=2))
    done = events[-1]
    assert [i.id for i in done.images] == ["img_1", "img_2"]  # not sorted by rank
    assert done.recommendation == "Image 2 matches your prompt best: sharp and on prompt."


async def test_paid_plan_sorts_by_rank(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2, best=2)])
    service = make_service(deps)
    events = await run(service, "r4", request(plan=Plan.PRO, n=2))
    assert [i.id for i in events[-1].images] == ["img_2", "img_1"]
    assert events[-1].recommendation.startswith("Image 1 matches")  # image 2 is now first


async def test_unsafe_image_is_dropped_and_not_charged(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(2)], vision=SeqVision(["unsafe\nS12", "safe"]))
    service = make_service(deps)
    events = await run(service, "r5", request(plan=Plan.BASIC, n=2))
    assert len(events[-1].images) == 1  # the unsafe one never reaches the user
    left = [n for n in ("img_1.jpg", "img_2.jpg") if deps.store.read_bytes("r5", n)]
    assert len(left) == 1  # the other was deleted, so it can never be fetched
    assert await deps.credits.balance("u1") == 99  # one basic-tier image


async def test_all_unsafe_rejects_run_without_charge(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, vision=SeqVision(["unsafe\nS12"]))
    service = make_service(deps)
    events = await run(service, "r6", request(plan=Plan.BASIC, n=2))
    assert events[-1].type == "error" and events[-1].code == "output_blocked"
    assert await deps.credits.balance("u1") == 100
    assert not (tmp_path / "r6").exists()


async def test_one_failed_image_does_not_cancel_the_others(tmp_path: Path) -> None:
    from app.agents.image_grand.workers.image_gen import seed_for

    provider = FakeProvider()
    deps = make_deps(tmp_path, provider=provider, critic=[critic_ok(2)])
    # models in build mode do not all take a seed, so fail by call order instead
    calls = {"n": 0}
    real = provider.generate

    async def flaky(**kw):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            from app.core.errors import ProviderCallFailed

            raise ProviderCallFailed("boom", retryable=False)
        return await real(**kw)

    provider.generate = flaky  # type: ignore[method-assign]
    _ = seed_for
    service = make_service(deps)
    events = await run(service, "r7", request(plan=Plan.BASIC, n=2))
    assert events[-1].type == "ready" and len(events[-1].images) == 1
    assert await deps.credits.balance("u1") == 99  # charged for the delivered one only


async def test_all_images_fail_means_error_and_full_refund(tmp_path: Path) -> None:
    from app.core.errors import ProviderCallFailed

    provider = FakeProvider()

    async def boom(**kw):  # type: ignore[no-untyped-def]
        raise ProviderCallFailed("down", retryable=False)

    provider.generate = boom  # type: ignore[method-assign]
    deps = make_deps(tmp_path, provider=provider)
    service = make_service(deps)
    events = await run(service, "r8", request(plan=Plan.BASIC, n=1))
    assert events[-1].type == "error"
    assert await deps.credits.balance("u1") == 100


async def test_question_pauses_then_resume_continues(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = make_service(deps)
    first = await run(service, "r9", request(prompt="make it nice", n=1, plan=Plan.BASIC))
    assert first[-1].type == "question"
    q = first[-1].questions[0]
    assert [o.id for o in q.options] == ["a", "b", "c", "other"]
    assert service.runs["r9"].status == "waiting"
    assert await deps.credits.balance("u1") < 100  # credits are held while waiting

    await service.resume("r9", {"q1": "Clean product photo"})
    second = [e async for e in service.stream("r9")]
    assert second[-1].type == "ready"
    # the enhancer saw the answer and could not ask again
    second_call = deps.enhancer.calls[1]  # type: ignore[attr-defined]
    assert "Clean product photo" in second_call["user"]
    assert '"allow_questions": false' in second_call["user"]


async def test_unanswered_question_expires_without_charge(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, enhancer=[missing()])
    service = make_service(deps)
    await run(service, "r10", request(prompt="make it nice", n=1, plan=Plan.BASIC))
    await service._expire(service.runs["r10"])
    assert service.runs["r10"].status == "expired"
    assert await deps.credits.balance("u1") == 100


async def test_plan_limit_rejects_n_without_charge(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)
    service = make_service(deps)
    events = await run(service, "r11", request(plan=Plan.BASIC, n=3))
    assert events[-1].type == "error" and events[-1].code == "plan_limit"
    assert "up to 2" in (events[-1].message or "")
    assert await deps.credits.balance("u1") == 100
    assert deps.enhancer.calls == []  # type: ignore[attr-defined]  # no LLM cost on a failed check


async def test_out_of_credits(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, credits=0)
    service = make_service(deps)
    events = await run(service, "r12", request(plan=Plan.BASIC, n=1))
    assert events[-1].code == "out_of_credits"


async def test_uploaded_image_is_not_available_yet(tmp_path: Path) -> None:
    deps = make_deps(tmp_path)
    service = make_service(deps)
    req = request(plan=Plan.BASIC, n=1)
    req.images = ["aGk="]
    events = await run(service, "r13", req)
    assert events[-1].code == "not_available"


async def test_same_prompt_twice_gives_new_images_and_is_charged_each_time(tmp_path: Path) -> None:
    provider = FakeProvider()
    deps = make_deps(tmp_path, provider=provider, critic=[critic_ok(1)])
    service = make_service(deps)
    await run(service, "r14", request(plan=Plan.BASIC, n=1))
    events = await run(service, "r15", request(plan=Plan.BASIC, n=1))
    assert events[-1].type == "ready"
    assert len(provider.calls) == 2  # no result cache: the second run really generated
    assert await deps.credits.balance("u1") == 98


async def test_seed_differs_per_image_and_is_sent_when_supported(tmp_path: Path) -> None:
    provider = FakeProvider()
    deps = make_deps(tmp_path, provider=provider, critic=[critic_ok(2)])
    service = make_service(deps)
    # phoenix supports a seed
    await run(service, "r16", request(plan=Plan.PRO, n=2, model="phoenix-1"))
    seeds = [c["seed"] for c in provider.calls]
    assert len(set(seeds)) == 2 and None not in seeds
    prompts = {c["prompt"] for c in provider.calls}
    assert len(prompts) == 1  # one final prompt for every variation


async def test_interrupt_and_resume_work_with_the_real_sqlite_checkpointer(tmp_path: Path) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.agents.grand.graph import build_grand_graph
    from app.api.run_service import RunService

    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
        service = RunService(build_grand_graph(deps, saver), deps)
        first = await run(service, "s1", request(prompt="make it nice", n=1, plan=Plan.BASIC))
        assert first[-1].type == "question"
        await service.resume("s1", {"q1": "A landscape"})
        second = [e async for e in service.stream("s1")]
        assert second[-1].type == "ready"
