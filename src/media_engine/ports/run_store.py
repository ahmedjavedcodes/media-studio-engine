"""RunStore port. Agents import the Protocol only, never a concrete store."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from media_engine.harness.run_state import RunState


@runtime_checkable
class RunStore(Protocol):
    """Where a run's state lives. TWO writers share each row:

    - the harness writes everything EXCEPT `cancel_requested` (`save`);
    - the backend writes `cancel_requested` (it cancels a run by setting it, on its side).

    So `save()` MUST NEVER write the `cancel_requested` field: the harness's in-memory copy says
    False, and writing it back would erase a cancel the backend set a moment earlier.
    `cancel_requested()` is the cheap read the harness polls while a step runs; `load()` returns
    the whole state, with the backend's flag applied."""

    async def save(self, state: RunState) -> None: ...
    async def load(self, run_id: str) -> RunState | None: ...
    async def cancel_requested(self, run_id: str) -> bool: ...


class InMemoryRunStore:
    """Local implementation. Stores JSON, not objects, so every save proves the state
    serialises and every load returns an independent copy.

    TODO(backend): real persistence lives in the backend repo behind this Protocol."""

    def __init__(self) -> None:
        self._rows: dict[str, str] = {}
        self._cancelled: set[str] = set()  # the backend's field, kept apart from the rows

    async def save(self, state: RunState) -> None:
        # Never writes `cancel_requested`: that field belongs to the backend (see RunStore).
        self._rows[state.run_id] = state.model_dump_json(exclude={"cancel_requested"})

    async def load(self, run_id: str) -> RunState | None:
        raw = self._rows.get(run_id)
        if raw is None:
            return None
        state = RunState.model_validate_json(raw)
        state.cancel_requested = run_id in self._cancelled
        return state

    async def cancel_requested(self, run_id: str) -> bool:
        return run_id in self._cancelled

    async def request_cancel(self, run_id: str) -> None:
        """What the backend does to cancel a run. TODO(backend): a column update in the real
        store; it never goes through `save()`."""
        self._cancelled.add(run_id)
