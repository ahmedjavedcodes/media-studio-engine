"""ArtifactStore port. Image bytes live here; everything else passes `image_id` strings."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}
_ID_PATTERN = re.compile(r"^img_[0-9a-f]{32}$")
ORPHANS_FILE = "orphaned.jsonl"


class ArtifactNotFound(KeyError):
    pass


@runtime_checkable
class ArtifactStore(Protocol):
    async def put(self, data: bytes, mime: str) -> str: ...  # returns image_id
    async def get(self, image_id: str) -> bytes: ...
    async def url(self, image_id: str) -> str: ...

    async def mark_orphaned(self, image_id: str, reason: str) -> None:
        """Flag a stored image that will never be delivered (e.g. generated, then the run was
        cancelled before the Critic checked it). It was paid for and must be swept, not leaked."""
        ...


def _check_id(image_id: str) -> str:
    if not _ID_PATTERN.match(image_id):
        raise ValueError(f"Invalid image_id {image_id!r}")
    return image_id


class LocalArtifactStore:
    """Writes images to a local directory (default `./.artifacts/`).

    TODO(backend): object storage, signed URLs and the scheduled orphan sweep live in the
    backend repo behind this Protocol. `sweep_orphaned()` is the local stand-in."""

    def __init__(self, root: str | Path = ".artifacts") -> None:
        self.root = Path(root)

    def _path(self, image_id: str) -> Path:
        _check_id(image_id)
        matches = sorted(self.root.glob(f"{image_id}.*"))
        if not matches:
            raise ArtifactNotFound(image_id)
        return matches[0]

    async def put(self, data: bytes, mime: str) -> str:
        ext = _EXT_BY_MIME.get(mime)
        if ext is None:
            raise ValueError(f"Unsupported mime type {mime!r}; known: {sorted(_EXT_BY_MIME)}")
        image_id = f"img_{uuid.uuid4().hex}"
        target = self.root / f"{image_id}.{ext}"

        def write() -> None:
            self.root.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(write)
        return image_id

    async def get(self, image_id: str) -> bytes:
        path = self._path(image_id)
        return await asyncio.to_thread(path.read_bytes)

    async def url(self, image_id: str) -> str:
        return self._path(image_id).resolve().as_uri()

    async def mark_orphaned(self, image_id: str, reason: str) -> None:
        _check_id(image_id)
        line = json.dumps(
            {"image_id": image_id, "reason": reason, "ts": datetime.now(UTC).isoformat()}
        )

        def append() -> None:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / ORPHANS_FILE).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        await asyncio.to_thread(append)

    async def orphaned(self) -> list[str]:
        path = self.root / ORPHANS_FILE
        if not path.exists():
            return []
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return [json.loads(row)["image_id"] for row in text.splitlines() if row.strip()]

    async def sweep_orphaned(self) -> int:
        """Delete every orphaned image and clear the list. Returns how many files were removed."""
        ids = await self.orphaned()

        def sweep() -> int:
            removed = 0
            for image_id in ids:
                for path in self.root.glob(f"{image_id}.*"):
                    path.unlink()
                    removed += 1
            (self.root / ORPHANS_FILE).unlink(missing_ok=True)
            return removed

        return await asyncio.to_thread(sweep)
