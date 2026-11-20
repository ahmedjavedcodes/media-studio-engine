"""Developer trace: what a run actually did, call by call, on a channel SEPARATE from the Envelope.

Nothing here reaches `Envelope.output`, an event or the public UI: internal agent names and the
model ids that served a call live only in `RunState.calls` / `RunState.decisions`, and the dev
shim serves them at `GET /runs/:id/trace` (only when DEBUG_TRACE=1). Capturing a trace adds NO
model call: every field comes from data the gateway already returns.

    Trace       collector for ONE run. Buffers until the RunState exists (the router runs first),
                then writes straight into `state.calls` / `state.decisions` / `state.orphaned`.
    scope()     a context variable naming the step / round / branch that is running, so an agent
                that makes a call (even in a concurrent branch task) records where it belongs.
    CallRecord  one model call, LLM or image. JSON-safe; no image bytes, no data: URLs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator

from media_engine.gateway.types import AttemptInfo

if TYPE_CHECKING:
    from media_engine.harness.run_state import RunState

OMITTED = "[image omitted]"
_DATA_URL = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=_-]*", re.IGNORECASE)
_LONG_BASE64 = re.compile(r"[A-Za-z0-9+/=_-]{400,}")

TraceStep = Literal["route", "enhance", "generate", "critique", "supervise"]


def redact(text: str | None) -> str | None:
    """Never let image data into the trace: a data: URL, or a long unbroken base64 run."""
    if text is None:
        return None
    return _LONG_BASE64.sub(OMITTED, _DATA_URL.sub(OMITTED, text))


def messages_text(messages: list[dict[str, Any]]) -> str:
    """The chat messages as readable text, with every image replaced by a marker."""
    lines: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            body = content
        else:
            parts = []
            for part in content or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(OMITTED)  # image_url or anything else non-text
            body = "\n".join(parts)
        lines.append(f"[{message.get('role', '?')}]\n{body}")
    return redact("\n\n".join(lines)) or ""


def now() -> str:
    return datetime.now(UTC).isoformat()


class ImageDetail(BaseModel):
    width_requested: int
    height_requested: int
    width: int | None = None  # what came back, read from the image itself
    height: int | None = None
    seed: int | None = None  # the seed the harness asked for
    seed_sent: bool = True  # False when the model takes no seed (it was not sent)
    seed_honoured: bool | None = None
    mime: str | None = None
    bytes: int | None = None
    steps: int | None = None  # None: the provider's default


class CallRecord(BaseModel):
    seq: int
    ts: str  # when the call STARTED (ISO, UTC)
    step: TraceStep
    role: str  # internal agent name
    attempt: int = 0  # round (0 for the router, which runs before round 1)
    branch: int = 0
    kind: Literal["llm", "image"] = "llm"
    chain: str | None = None
    provider: str | None = None
    model: str | None = None  # the model that actually served the call
    was_fallback: bool = False  # it was not the first model in the chain
    attempts: list[AttemptInfo] = Field(default_factory=list)  # gateway chain walk
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None  # None: not reported
    total_tokens: int = 0
    tokens_estimated: bool = False  # provider sent no usage: 4-chars-per-token estimate
    reasoning_effort: str | None = None
    neurons: float = 0.0
    cost_usd: float = 0.0
    duration_ms: int = 0
    was_repair: bool = False  # the ONE repair call after an unusable reply
    failure: str | None = None  # a FailureKind (or "aborted"), None when the call was fine
    request_text: str = ""
    reply_text: str | None = None
    reasoning_text: str | None = None  # None: the provider did not return it
    image: ImageDetail | None = None

    @field_validator("request_text", mode="before")
    @classmethod
    def _clean_request(cls, value: Any) -> Any:
        return redact(value) if isinstance(value, str) else value

    @field_validator("reply_text", "reasoning_text", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> Any:
        return redact(value) if isinstance(value, str) else value


class Decision(BaseModel):
    seq: int
    ts: str
    step: str
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        return redact(value) if isinstance(value, str) else value


class Orphan(BaseModel):
    image_id: str
    reason: str
    ts: str


class Trace:
    """Collector for one run. Create it, hand it to `create_images()` / the supervisor, and the
    dev shim can read `snapshot()` while the run is still going."""

    def __init__(self) -> None:
        self.state: RunState | None = None
        self.meta: dict[str, Any] = {}  # caps etc., set by the supervisor on bind
        self._calls: list[CallRecord] = []
        self._decisions: list[Decision] = []
        self._orphans: list[Orphan] = []
        self._seq = 0
        self._dseq = 0
        # The trace's own clock, so a run that never gets a RunState (escalate) still has an
        # overall time and a final status, and the router (which runs first) is inside it.
        self.started_at = datetime.now(UTC)
        self.ended_at: datetime | None = None
        self.final_status: str | None = None

    # -- writing ----------------------------------------------------------------------------

    @property
    def calls(self) -> list[CallRecord]:
        return self.state.calls if self.state is not None else self._calls

    @property
    def decisions(self) -> list[Decision]:
        return self.state.decisions if self.state is not None else self._decisions

    @property
    def orphans(self) -> list[Orphan]:
        return self.state.orphaned if self.state is not None else self._orphans

    def add_call(self, **fields: Any) -> CallRecord:
        """Append a record (seq assigned here, in call order) and return it, so the caller can
        still complete it (e.g. mark `failure` once the reply was parsed)."""
        self._seq += 1
        record = CallRecord(seq=self._seq, ts=fields.pop("ts", None) or now(), **fields)
        self.calls.append(record)
        return record

    def decide(self, step: str, text: str) -> None:
        self._dseq += 1
        self.decisions.append(Decision(seq=self._dseq, ts=now(), step=step, text=text))

    def orphan(self, image_id: str, reason: str) -> None:
        self.orphans.append(Orphan(image_id=image_id, reason=reason, ts=now()))
        self.decide("settle", f"image {image_id} orphaned: {reason}")

    def finish(self, status: str) -> None:
        """The run is over (called by `create_images`, also for a run with no RunState)."""
        self.final_status = status
        self.ended_at = datetime.now(UTC)

    def bind(self, state: RunState) -> None:
        """Attach the RunState: what was buffered (the router) moves in, and from now on records
        go straight into the state, so it round-trips through JSON with them."""
        state.calls = [*self._calls, *state.calls]
        state.decisions = [*self._decisions, *state.decisions]
        state.orphaned = [*self._orphans, *state.orphaned]
        self._calls, self._decisions, self._orphans = [], [], []
        self.state = state

    # -- reading ----------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything the inspector shows, JSON-safe, safe to call while the run is going."""
        calls = [c.model_dump(mode="json") for c in self.calls]
        llm = [c for c in self.calls if c.kind == "llm"]
        totals = {
            "calls": len(self.calls),
            "llm_calls": len(llm),
            "image_calls": len(self.calls) - len(llm),
            "prompt_tokens": sum(c.prompt_tokens for c in llm),
            "completion_tokens": sum(c.completion_tokens for c in llm),
            "reasoning_tokens": sum(c.reasoning_tokens or 0 for c in llm),
            "total_tokens": sum(c.total_tokens for c in llm),
            "neurons": sum(c.neurons for c in self.calls),
            "cost_usd": sum(c.cost_usd for c in self.calls),
        }
        state = self.state
        ended = (state.finished_at if state is not None else None) or self.ended_at
        out: dict[str, Any] = {
            "calls": calls,
            "decisions": [d.model_dump(mode="json") for d in self.decisions],
            "orphaned": [o.model_dump(mode="json") for o in self.orphans],
            "totals": totals,
            "meta": self.meta,
            "steps": [],
            "attempts": [],
            "budget": None,
            "supervisor_spend": None,
            "retries_used": 0,
            "critic_reasks_used": 0,
            "status": self.final_status or "running",
            # Overall time runs from the trace's start (before the router call), to the end.
            "created_at": self.started_at.isoformat(),
            "finished_at": ended.isoformat() if ended else None,
            "wall_ms": int(((ended or datetime.now(UTC)) - self.started_at).total_seconds() * 1000),
        }
        if state is not None:
            out.update(
                steps=[s.model_dump(mode="json") for s in state.plan],
                attempts=[a.model_dump(mode="json") for a in state.attempts],
                budget=state.budget.model_dump(mode="json"),
                supervisor_spend=state.supervisor_spend.model_dump(mode="json"),
                retries_used=state.retries_used,
                critic_reasks_used=state.critic_reasks_used,
                status=self.final_status or str(getattr(state.status, "value", state.status)),
            )
        return out


# -- who is calling: a context variable, so concurrent branch tasks each record their own ------


@dataclass(frozen=True)
class Scope:
    trace: Trace
    step: TraceStep
    attempt: int
    branch: int


_SCOPE: ContextVar[Scope | None] = ContextVar("media_engine_trace_scope", default=None)


@contextmanager
def scope(trace: Trace, step: TraceStep, attempt: int = 0, branch: int = 0) -> Iterator[None]:
    token = _SCOPE.set(Scope(trace, step, attempt, branch))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def current() -> Scope | None:
    """The scope of the call being made right now, or None (no trace: nothing is recorded)."""
    return _SCOPE.get()
