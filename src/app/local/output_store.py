"""Local file store for generated images and run results (dev). Swap for object storage later.

Names are checked twice: by a strict pattern (no dots at the start, no slashes) and by resolving
the final path and requiring it to stay inside the store root.
"""

import json
import re
import shutil
from pathlib import Path

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}(\.[A-Za-z0-9]{1,5})?$")
RESULT_FILE = "result.json"


class OutputStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, run_id: str, name: str | None = None) -> Path | None:
        """The path for a run folder or a file in it, or None if it could leave the store."""
        if not _RUN_ID.match(run_id) or (name is not None and not _NAME.match(name)):
            return None
        root = self.root.resolve()
        path = (root / run_id / name).resolve() if name else (root / run_id).resolve()
        return path if path.is_relative_to(root) else None

    def write_bytes(self, run_id: str, name: str, data: bytes) -> None:
        path = self._path(run_id, name)
        if path is None:
            raise ValueError("bad run id or file name")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, run_id: str, name: str) -> bytes | None:
        """Bytes of a saved file, or None. Anything that could leave the store is None."""
        path = self._path(run_id, name)
        return path.read_bytes() if path is not None and path.is_file() else None

    def delete(self, run_id: str, name: str) -> None:
        path = self._path(run_id, name)
        if path is not None:
            path.unlink(missing_ok=True)

    def purge(self, run_id: str) -> None:
        """Delete every file of a run. A failed run leaves nothing that can be fetched."""
        path = self._path(run_id)
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)

    def write_json(self, run_id: str, name: str, data: object) -> None:
        self.write_bytes(run_id, name, json.dumps(data, ensure_ascii=False, indent=2).encode())

    def read_json(self, run_id: str, name: str) -> dict[str, object] | None:
        raw = self.read_bytes(run_id, name)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except ValueError:  # a half-written or corrupt file
            return None
        return data if isinstance(data, dict) else None

    def released(self, run_id: str) -> set[str]:
        """The files a user may fetch: the images of the saved result and their thumbnails.
        Nothing is released until the run has passed the safety check and been saved."""
        result = self.read_json(run_id, RESULT_FILE)
        names: set[str] = set()
        docs = result.get("documents") if result else None
        try:
            for doc in docs if isinstance(docs, list) else []:
                for layer in doc["versions"][0]["layers"]:
                    names.update({layer["file"], layer["thumb"]})
        except (KeyError, IndexError, TypeError):
            return set()  # a result we cannot read releases nothing
        return names

    def read_released(self, run_id: str, name: str) -> bytes | None:
        return self.read_bytes(run_id, name) if name in self.released(run_id) else None

    def url(self, run_id: str, name: str) -> str:
        return f"/v1/files/{run_id}/{name}"
