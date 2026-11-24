"""B3: checkpoints are deleted when a run ends, and old ones are swept at startup."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.grand.graph import build_grand_graph
from app.api.checkpoint_cleanup import forget_thread, sweep_old_threads
from app.api.run_service import RunService
from app.contracts.request import GenerationRequest, ImageSettings
from app.core.plans import Plan
from tests.unit.create_helpers import SeqVision, critic_ok, make_deps, missing, ready


def req(prompt: str = "a fox", **kw: Any) -> GenerationRequest:
    return GenerationRequest(prompt=prompt, plan=Plan.BASIC, settings=ImageSettings(**kw))


async def has_thread(saver: Any, thread: str) -> bool:
    return await saver.aget_tuple({"configurable": {"thread_id": thread}}) is not None


def service_with(deps: Any, saver: Any) -> RunService:
    return RunService(build_grand_graph(deps, saver), deps)


async def test_the_thread_is_gone_after_done(tmp_path: Path) -> None:
    saver = InMemorySaver()
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    service = service_with(deps, saver)
    service.start("d1", "u1", req(n=1))
    assert [e async for e in service.stream("d1")][-1].type == "ready"
    await service.wait("d1")
    assert not await has_thread(saver, "d1")


async def test_the_thread_is_gone_after_failed(tmp_path: Path) -> None:
    saver = InMemorySaver()
    deps = make_deps(tmp_path, vision=SeqVision(["unsafe\nS12"]))
    service = service_with(deps, saver)
    service.start("f1", "u1", req(n=1))
    assert [e async for e in service.stream("f1")][-1].type == "error"
    await service.wait("f1")
    assert not await has_thread(saver, "f1")


async def test_the_thread_is_kept_while_a_question_waits_and_gone_after_expiry(
    tmp_path: Path,
) -> None:
    saver = InMemorySaver()
    deps = make_deps(tmp_path, enhancer=[missing(), ready()])
    service = service_with(deps, saver)
    service.start("x1", "u1", req("make it nice", n=1))
    assert [e async for e in service.stream("x1")][-1].type == "question"
    assert await has_thread(saver, "x1")  # needed for the resume
    await service._expire(service.runs["x1"])
    assert not await has_thread(saver, "x1")


async def test_the_thread_is_gone_after_a_resumed_run_finishes(tmp_path: Path) -> None:
    saver = InMemorySaver()
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(1)])
    service = service_with(deps, saver)
    service.start("r1", "u1", req("make it nice", n=1))
    [e async for e in service.stream("r1")]
    await service.resume("r1", {"q1": "A"})
    assert [e async for e in service.stream("r1")][-1].type == "ready"
    await service.wait("r1")
    assert not await has_thread(saver, "r1")


async def test_a_delete_failure_never_breaks_a_run() -> None:
    class Broken:
        async def adelete_thread(self, thread_id: str) -> None:
            raise RuntimeError("db locked")

    await forget_thread(Broken(), "x")  # logged, not raised
    await forget_thread(None, "x")


async def test_the_startup_sweep_deletes_only_threads_older_than_a_day(tmp_path: Path) -> None:
    saver = InMemorySaver()
    deps = make_deps(tmp_path, enhancer=[missing()])  # every run stops at a question
    service = service_with(deps, saver)
    for rid in ("w1", "w2"):  # two runs paused at a question
        service.start(rid, "u1", req("make it nice", n=1))
        [e async for e in service.stream(rid)]
    assert await sweep_old_threads(saver) == 0  # nothing is old yet
    assert await has_thread(saver, "w1") and await has_thread(saver, "w2")
    later = datetime.now(UTC) + timedelta(hours=25)
    assert await sweep_old_threads(saver, now=later) == 2
    assert not await has_thread(saver, "w1") and not await has_thread(saver, "w2")


async def test_cleanup_works_on_the_real_sqlite_saver(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
        service = service_with(deps, saver)
        service.start("s1", "u1", req(n=1))
        assert [e async for e in service.stream("s1")][-1].type == "ready"
        await service.wait("s1")
        assert not await has_thread(saver, "s1")
        assert await sweep_old_threads(saver) == 0
