"""LangSmith tracing: one trace per run, one span per step, no image bytes."""

import json
from pathlib import Path

from app.contracts.request import GenerationRequest, ImageSettings
from app.core.plans import Plan
from app.core.prompts import PromptStore, _find_template
from app.core.tracing import scrub
from tests.unit.create_helpers import (
    RecordingClient,
    critic_ok,
    dump,
    make_deps,
    make_service,
    missing,
    ready,
)

STEPS = {
    "code_checks",
    "router",
    "enhancer",
    "interrupt",
    "code_guard",
    "generate",
    "critic",
    "settle",
}


def by_name(client: RecordingClient) -> dict[str, list[dict]]:  # type: ignore[type-arg]
    out: dict[str, list[dict]] = {}  # type: ignore[type-arg]
    for run in client.created:
        out.setdefault(run["name"], []).append(run)
    return out


async def full_run(tmp_path: Path, client: RecordingClient, n: int = 2):  # type: ignore[no-untyped-def]
    deps = make_deps(tmp_path, enhancer=[missing(), ready()], critic=[critic_ok(n)], trace=client)
    service = make_service(deps)
    req = GenerationRequest(prompt="make it nice", plan=Plan.PRO, settings=ImageSettings(n=n))
    service.start("run1", "user1", req)
    first = [e async for e in service.stream("run1")]
    await service.resume("run1", {"q1": "Clean product photo"})
    second = [e async for e in service.stream("run1")]
    await service.wait("run1")
    return deps, first + second


async def test_one_full_run_is_one_trace_with_all_eight_step_spans(tmp_path: Path) -> None:
    client = RecordingClient()
    await full_run(tmp_path, client, n=2)
    names = by_name(client)
    assert STEPS <= set(names), STEPS - set(names)
    assert len(names["txt2img_run"]) == 1
    assert len(names["generate_image"]) == 2  # one child span per variation
    trace_ids = {r["trace_id"] for r in client.created}
    assert len(trace_ids) == 1  # one trace, even across the interrupt and the resume
    root_id = names["txt2img_run"][0]["id"]
    for run in client.created:
        if run["name"] != "txt2img_run":
            assert run["trace_id"] == names["txt2img_run"][0]["trace_id"]
    gen_id = names["generate"][0]["id"]
    assert {r["parent_run_id"] for r in names["generate_image"]} == {gen_id}
    for step in STEPS:
        assert names[step][0]["parent_run_id"] == root_id


async def test_every_run_carries_the_required_metadata(tmp_path: Path) -> None:
    client = RecordingClient()
    await full_run(tmp_path, client, n=1)
    required = {"run_id", "user_id", "plan", "model", "provider", "compile_mode"}
    spans = [
        r
        for r in client.created
        if r["name"] != "txt2img_run" and r["name"] != "code_checks" and r["name"] != "router"
    ]
    assert spans
    for run in spans:  # spans started after the router know the model and provider
        meta = run["extra"]["metadata"]
        assert required <= set(meta), (run["name"], required - set(meta))
        assert meta["run_id"] == "run1" and meta["user_id"] == "user1" and meta["plan"] == "pro"
        assert meta["compile_mode"] == "llm" and meta["provider"] == "cloudflare"
    root_end = [u for u in client.updated if u.get("extra", {}).get("metadata", {}).get("model")]
    assert root_end  # the root is updated with the real model at the end


async def test_generate_spans_log_seed_size_latency_cost_and_fallback(tmp_path: Path) -> None:
    client = RecordingClient()
    await full_run(tmp_path, client, n=2)
    ids = {str(r["id"]) for r in by_name(client)["generate_image"]}
    updates = [u for u in client.updated if u["id"] in ids]
    assert len(updates) == 2
    for u in updates:
        out = u["outputs"]
        for key in (
            "seed_sent",
            "seed_honoured",
            "size",
            "latency_ms",
            "usage_cost",
            "provider_fallback",
            "image_id",
            "thumb_url",
        ):
            assert key in out, key
        assert out["provider_fallback"] is False
        assert out["size"] == "64x64" and out["usage_cost"] == 0.001


async def test_traces_hold_ids_and_urls_never_image_bytes(tmp_path: Path) -> None:
    client = RecordingClient()
    await full_run(tmp_path, client, n=2)
    blob = json.dumps({"created": client.created, "updated": client.updated}, default=str)
    assert "data:image" not in blob and "base64" not in blob
    assert "\\xff\\xd8" not in blob and "JFIF" not in blob


async def test_prompt_versions_are_recorded_in_the_trace_metadata(tmp_path: Path) -> None:
    client = RecordingClient()
    await full_run(tmp_path, client, n=1)
    enhancer_span = by_name(client)["enhancer"][-1]
    assert enhancer_span["extra"]["metadata"]["enhancer_prompt_version"].startswith("local-")
    critic_span = by_name(client)["critic"][0]
    assert critic_span["extra"]["metadata"]["critic_prompt_version"].startswith("local-")


async def test_traces_never_reach_the_sse_stream(tmp_path: Path) -> None:
    client = RecordingClient()
    _, events = await full_run(tmp_path, client, n=1)
    text = dump(events).lower()
    for word in ("trace", "langsmith", "span", "run_tree", "prompt_version", "local-"):
        assert word not in text


async def test_tracing_off_means_no_spans_and_the_run_still_works(tmp_path: Path) -> None:
    deps = make_deps(tmp_path, critic=[critic_ok(1)])  # no client: tracing is off
    service = make_service(deps)
    service.start("r", "u", GenerationRequest(prompt="a fox", settings=ImageSettings(n=1)))
    events = [e async for e in service.stream("r")]
    assert events[-1].type == "ready"


def test_scrub_replaces_bytes_and_data_urls() -> None:
    out = scrub({"a": b"\xff\xd8\xff", "b": ["data:image/png;base64,AAAA"], "c": 1})
    assert out == {"a": "<3 bytes>", "b": ["<data url removed>"], "c": 1}


# ---- prompts from the Hub with a version tag ----------------------------------------------


class _Commit:
    commit_hash = "abcdef1234567890"
    manifest = {
        "kwargs": {"messages": [{"kwargs": {"prompt": {"kwargs": {"template": "HUB TEXT"}}}}]}
    }


class _HubClient:
    def __init__(self, fail: bool = False) -> None:
        self.refs: list[str] = []
        self.fail = fail

    def pull_prompt_commit(self, ref: str) -> _Commit:
        self.refs.append(ref)
        if self.fail:
            raise RuntimeError("offline")
        return _Commit()


async def test_prompts_are_pulled_from_the_hub_with_a_version_tag() -> None:
    client = _HubClient()
    store = PromptStore(source="hub", tag="dev", client_factory=lambda: client)  # type: ignore[arg-type,return-value]
    await store.load()
    assert client.refs == ["media-enhancer:dev", "media-critic:dev"]
    assert store.get("enhancer").text == "HUB TEXT"
    assert store.get("enhancer").version == "dev@abcdef12"


async def test_hub_failure_falls_back_to_the_local_file() -> None:
    store = PromptStore(
        source="hub",
        client_factory=lambda: _HubClient(fail=True),  # type: ignore[arg-type,return-value]
    )
    await store.load()
    got = store.get("critic")
    assert got.version.startswith("local-") and "review the results" in got.text


def test_find_template_walks_nested_manifests() -> None:
    assert _find_template({"a": [{"b": {"template": "X"}}]}) == "X"
    assert _find_template({"a": 1}) is None
