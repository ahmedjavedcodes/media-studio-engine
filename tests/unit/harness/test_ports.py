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
    event = make_event("run_1", 2, "Creating your image", "started")
    assert set(event) == {"run_id", "step", "public_label", "phase", "ts"}


def test_make_event_refuses_internal_names() -> None:
    with pytest.raises(ValidationError):
        make_event("run_1", 2, "image_creator", "started")  # type: ignore[arg-type]


def test_list_sink_collects() -> None:
    sink = ListEventSink()
    safe_emit(sink, "run_1", make_event("run_1", 1, "Understanding your request", "started"))
    assert sink.events[0][0] == "run_1"


def test_safe_emit_swallows_a_raising_sink() -> None:
    class Exploding:
        def emit(self, run_id: str, event: dict) -> None:
            raise RuntimeError("sink down")

    safe_emit(Exploding(), "run_1", make_event("run_1", 1, "Finishing up", "finished"))
