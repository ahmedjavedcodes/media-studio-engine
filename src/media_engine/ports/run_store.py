"""RunStore port. Agents import the Protocol only, never a concrete store."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from media_engine.harness.run_state import RunState


@runtime_checkable
class RunStore(Protocol):
    async def save(self, state: RunState) -> None: ...
    async def load(self, run_id: str) -> RunState | None: ...


class InMemoryRunStore:
    """Local implementation. Stores JSON, not objects, so every save proves the state
    serialises and every load returns an independent copy.

    TODO(backend): real persistence lives in the backend repo behind this Protocol."""

    def __init__(self) -> None:
        self._rows: dict[str, str] = {}

    async def save(self, state: RunState) -> None:
        self._rows[state.run_id] = state.model_dump_json()

    async def load(self, run_id: str) -> RunState | None:
        raw = self._rows.get(run_id)
        return RunState.model_validate_json(raw) if raw is not None else None
