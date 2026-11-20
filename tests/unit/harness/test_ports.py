"""M1: local port implementations."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from media_engine.ports import (
    ArtifactNotFound,
    ArtifactStore,
    EventSink,
    InMemoryRunStore,
    ListEventSink,
    LocalArtifactStore,
    RunStore,
    make_event,
    safe_emit,
)
from tests.unit.harness.fakes import RaisingSink


def test_local_implementations_satisfy_protocols(tmp_path: Path) -> None:
    assert isinstance(InMemoryRunStore(), RunStore)
    assert isinstance(LocalArtifactStore(tmp_path), ArtifactStore)
    assert isinstance(ListEventSink(), EventSink)


async def test_artifact_put_get_url(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / ".artifacts")
    image_id = await store.put(b"\x89PNG-fake", "image/png")
    assert image_id.startswith("img_")
    assert await store.get(image_id) == b"\x89PNG-fake"
    assert (tmp_path / ".artifacts" / f"{image_id}.png").exists()
    assert (await store.url(image_id)).startswith("file://")


async def test_artifact_rejects_bad_ids_and_mimes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        await store.get("../../etc/passwd")
    with pytest.raises(ArtifactNotFound):
        await store.get("img_" + "0" * 32)
    with pytest.raises(ValueError):
        await store.put(b"x", "application/octet-stream")


async def test_orphaned_images_are_recorded_and_swept(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    kept = await store.put(b"keep", "image/png")
    dropped = await store.put(b"drop", "image/jpeg")

    await store.mark_orphaned(dropped, "cancelled before critic")
    assert await store.orphaned() == [dropped]

    assert await store.sweep_orphaned() == 1
    assert await store.orphaned() == []
    assert await store.get(kept) == b"keep"
    with pytest.raises(ArtifactNotFound):
        await store.get(dropped)


def test_make_event_has_exact_public_shape() -> None:
    # Serialised as safe_emit does (exclude_none): an ordinary event keeps its exact shape.
    event = make_event("run_1", 2, "Creating your image", "started").model_dump(exclude_none=True)
    assert set(event) == {"run_id", "step", "public_label", "phase", "ts"}


def test_image_ready_carries_exactly_an_image_id() -> None:
    event = make_event("run_1", 3, "Finishing up", "image_ready", image_id="img_x")
    assert event.model_dump(exclude_none=True)["image_id"] == "img_x"
    with pytest.raises(ValueError):
        make_event("run_1", 3, "Finishing up", "image_ready")  # no id
    with pytest.raises(ValueError):
        make_event("run_1", 3, "Finishing up", "finished", image_id="img_x")  # id on another


def test_make_event_refuses_internal_names() -> None:
    with pytest.raises(ValidationError):
        make_event("run_1", 2, "image_creator", "started")  # type: ignore[arg-type]


def test_list_sink_collects_serialised_events() -> None:
    sink = ListEventSink()
    safe_emit(sink, make_event("run_1", 1, "Understanding your request", "started"))
    run_id, event = sink.events[0]
    assert run_id == "run_1"
    assert event["public_label"] == "Understanding your request"


def test_safe_emit_swallows_a_sink_that_always_raises() -> None:
    sink = RaisingSink()
    for phase in ("started", "progress", "finished"):
        safe_emit(sink, make_event("run_1", 1, "Finishing up", phase))  # type: ignore[arg-type]
    assert sink.attempts == 3


def test_safe_emit_only_accepts_typed_events() -> None:
    # A hand-built dict (which could carry an internal agent name) is not an Event.
    with pytest.raises(TypeError, match="takes an Event"):
        safe_emit(ListEventSink(), {"run_id": "r", "public_label": "critic"})  # type: ignore[arg-type]
