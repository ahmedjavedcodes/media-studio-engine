"""Tracing for the txt2img run. One trace per run, one span per step.

Every span is recorded in two places. LangSmith gets it when tracing is on. And it is always kept
locally (timestamps, inputs, outputs), so the dev trace panel (`GET /v1/runs/{id}/trace`) can show a
run from the input to the last step, with latency, cost, model and provider for each span.

Spans are explicit RunTree children, not callbacks. That keeps ONE trace for the whole run even
when the graph pauses at `interrupt()` and a second request resumes it. Span names:
code_checks, router, enhancer, interrupt, code_guard, generate (with one `generate_image` child
per variation), critic, settle.

Rules: never log image bytes (ids and thumbnail URLs only), and traces never go to the SSE stream.
"""

import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from langsmith import Client
from langsmith.run_trees import RunTree

log = structlog.get_logger("tracing")

TRACE_TAGS = ["agent:txt2img"]


def scrub(value: Any, _depth: int = 0) -> Any:
    """Make a value safe to send to LangSmith. Bytes become a size note, never the bytes."""
    if isinstance(value, bytes | bytearray):
        return f"<{len(value)} bytes>"
    if _depth > 6:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k): scrub(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [scrub(v, _depth + 1) for v in value]
    if isinstance(value, str) and value.startswith("data:image"):
        return "<data url removed>"
    return value


# Which kind of step each span name is. The panel uses it for colour and for the totals.
KINDS = {
    "guardrail": "guardrail",
    "code_checks": "check",
    "router": "router",
    "enhancer": "llm",
    "art_director": "llm",
    "prompt_writer": "llm",
    "interrupt": "wait",
    "code_guard": "check",
    "generate": "generate",
    "generate_image": "image",
    "safety": "safety",
    "critic": "llm",
    "settle": "settle",
}
NEURON_USD = 0.011 / 1000  # Cloudflare's price: $0.011 per 1,000 neurons


class Span:
    """One step. Collects outputs and metadata. Kept locally, and posted to LangSmith when it
    ends (if tracing is on)."""

    def __init__(
        self,
        tree: RunTree | None,
        *,
        span_id: int = 0,
        name: str = "",
        parent_id: int | None = None,
        inputs: dict[str, Any] | None = None,
        started_at: float | None = None,
    ) -> None:
        self._tree = tree
        self.id = span_id
        self.name = name
        self.parent_id = parent_id
        self.inputs: dict[str, Any] = scrub(inputs or {})
        self.outputs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.started_at = started_at if started_at is not None else time.time()
        self.ended_at: float | None = None
        self.error: str | None = None
        self.ended = False

    def log(self, **outputs: Any) -> None:
        self.outputs.update(outputs)

    def meta(self, **metadata: Any) -> None:
        self.metadata.update(metadata)

    def end(self, error: str | None = None, *, ended_at: float | None = None) -> None:
        if self.ended:
            return
        self.ended = True
        self.ended_at = ended_at if ended_at is not None else time.time()
        self.error = error
        if self._tree is None:
            return
        try:
            self._tree.end(
                outputs=scrub(self.outputs) or None,
                error=error,
                metadata=scrub(self.metadata) or None,
            )
            self._tree.patch()
        except Exception as exc:  # tracing must never break a run
            log.warning("trace_span_end_failed", error=type(exc).__name__)

    def to_dict(self, run_started: float, now: float) -> dict[str, Any]:
        end = self.ended_at if self.ended_at is not None else now
        status = "error" if self.error else ("ok" if self.ended else "running")
        outputs = scrub(self.outputs)
        meta = scrub(self.metadata)
        return {
            "id": self.id,
            "parent": self.parent_id,
            "name": self.name,
            "kind": KINDS.get(self.name, "step"),
            "status": status,
            "start_ms": round((self.started_at - run_started) * 1000),
            "duration_ms": round((end - self.started_at) * 1000),
            "model": outputs.get("model") or meta.get("model"),
            "provider": outputs.get("provider") or meta.get("provider"),
            "inputs": self.inputs,
            "outputs": outputs,
            "metadata": meta,
            "error": self.error,
        }


class RunTrace:
    """The trace of one run. Lives across an interrupt and its resume."""

    def __init__(
        self, root: RunTree | None, metadata: dict[str, Any], inputs: dict[str, Any] | None = None
    ) -> None:
        self._root = root
        self.metadata = dict(metadata)
        self.inputs: dict[str, Any] = scrub(inputs or {})
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.spans: list[Span] = []  # every span, in start order
        self._spans: dict[str, Span] = {}  # by key, for lookups. A repeated key keeps the newest

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def set_metadata(self, **metadata: Any) -> None:
        """Add run metadata (model, provider ...). New spans carry it too."""
        self.metadata.update(metadata)

    def start(
        self,
        name: str,
        *,
        parent: str | None = None,
        inputs: dict[str, Any] | None = None,
        run_type: str = "chain",
        key: str | None = None,
        started_at: float | None = None,
    ) -> Span:
        parent_span = self._spans.get(parent) if parent else None
        tree: RunTree | None = None
        if self._root is not None:
            parent_tree = parent_span._tree if parent_span is not None else None
            try:
                tree = (parent_tree or self._root).create_child(
                    name,
                    run_type,  # type: ignore[arg-type]
                    inputs=scrub(inputs or {}),
                    extra={"metadata": scrub(self.metadata)},
                    tags=TRACE_TAGS,
                )
                tree.post()
            except Exception as exc:  # tracing must never break a run
                log.warning("trace_span_start_failed", error=type(exc).__name__)
                tree = None
        span = Span(
            tree,
            span_id=len(self.spans) + 1,
            name=name,
            parent_id=parent_span.id if parent_span is not None else None,
            inputs=inputs,
            started_at=started_at,
        )
        self.spans.append(span)
        self._spans[key or name] = span
        return span

    def record(
        self,
        name: str,
        *,
        started_at: float,
        ended_at: float,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        parent: str | None = None,
        key: str | None = None,
    ) -> Span:
        """A step that already happened (the guardrail ran before this trace began, and each
        safety check is timed inside its own helper)."""
        span = self.start(name, inputs=inputs, started_at=started_at, parent=parent, key=key)
        span.log(**(outputs or {}))
        span.end(error=error, ended_at=ended_at)
        return span

    def span_named(self, key: str) -> Span | None:
        return self._spans.get(key)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        parent: str | None = None,
        inputs: dict[str, Any] | None = None,
        run_type: str = "chain",
        key: str | None = None,
    ) -> Iterator[Span]:
        span = self.start(name, parent=parent, inputs=inputs, run_type=run_type, key=key)
        try:
            yield span
        except BaseException as exc:
            span.end(error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        else:
            span.end()

    def finish(self, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        for span in self.spans:  # close anything still open (a timed-out interrupt)
            span.end(error=error)
        self.ended_at = time.time()
        if self._root is None:
            return
        try:
            self._root.end(outputs=scrub(outputs or {}), error=error, metadata=scrub(self.metadata))
            self._root.patch()
        except Exception as exc:
            log.warning("trace_finish_failed", error=type(exc).__name__)

    # ---- the dev trace panel ---------------------------------------------------------------

    def snapshot(self, status: str) -> dict[str, Any]:
        """The whole trace as plain data: the input, every span, and the totals. Scrubbed (no
        image bytes). Safe to call while the run is still going."""
        now = time.time()
        first = min([self.started_at, *[s.started_at for s in self.spans]])
        end = self.ended_at if self.ended_at is not None else now
        spans = [s.to_dict(first, now) for s in sorted(self.spans, key=lambda x: x.started_at)]
        return {
            "run_id": self.metadata.get("run_id"),
            "user_id": self.metadata.get("user_id"),
            "status": status,
            "input": self.inputs,
            "metadata": scrub(self.metadata),
            "totals": _totals(spans, round((end - first) * 1000)),
            "spans": spans,
        }


def _totals(spans: list[dict[str, Any]], duration_ms: int) -> dict[str, Any]:
    """Add up cost, time and tokens from what the spans logged."""
    images_usd = 0.0
    images_estimated = False
    neurons = 0.0
    llm_direct_usd = 0.0  # LLM calls billed per token (Groq), not in neurons
    tokens = {"prompt": 0, "completion": 0}
    held = charged = delivered = requested = 0
    for sp in spans:
        out = sp["outputs"]
        if sp["name"] == "generate_image":
            cost = out.get("cost_usd")
            if cost is not None:
                images_usd += float(cost)
            else:
                images_usd += float(out.get("cost_estimate_usd") or 0.0)
                images_estimated = True
        neurons += float(out.get("neurons") or 0.0)
        if sp["name"] != "generate_image":
            llm_direct_usd += float(out.get("cost_usd") or 0.0)
        tokens["prompt"] += int(out.get("prompt_tokens") or 0)
        tokens["completion"] += int(out.get("completion_tokens") or 0)
        held += int(out.get("held") or 0)
        charged += int(out.get("charged") or 0)
        if sp["name"] == "generate":
            requested = int(sp["inputs"].get("n") or 0)
        if sp["name"] == "settle":
            delivered = int(out.get("delivered") or 0)
    llm_usd = neurons * NEURON_USD + llm_direct_usd
    return {
        "duration_ms": duration_ms,
        "images_usd": round(images_usd, 6),
        "images_cost_estimated": images_estimated,  # true when a provider reported no cost
        "llm_usd_estimate": round(llm_usd, 6),  # neurons at Cloudflare's price, plus Groq tokens
        "total_usd": round(images_usd + llm_usd, 6),
        "neurons": round(neurons, 1),
        "tokens": tokens,
        "credits_held": held,
        "credits_charged": charged,
        "images_requested": requested,
        "images_delivered": delivered,
    }


class Tracer:
    """Makes one RunTrace per run. Off (no-op spans) unless tracing is on and a client exists."""

    def __init__(self, *, client: Client | None, project: str, enabled: bool) -> None:
        self._client = client
        self._project = project
        self._enabled = enabled and client is not None
        self._runs: dict[str, RunTrace] = {}
        self._finished: OrderedDict[str, RunTrace] = OrderedDict()  # kept for the dev panel

    def start_run(
        self, run_id: str, *, metadata: dict[str, Any], inputs: dict[str, Any]
    ) -> RunTrace:
        root: RunTree | None = None
        if self._enabled:
            try:
                root = RunTree(
                    name="txt2img_run",
                    run_type="chain",
                    inputs=scrub(inputs),
                    project_name=self._project,
                    client=self._client,
                    extra={"metadata": scrub(metadata)},
                    tags=TRACE_TAGS,
                )
                root.post()
            except Exception as exc:
                log.warning("trace_start_failed", error=type(exc).__name__)
                root = None
        trace = RunTrace(root, metadata, inputs)
        self._runs[run_id] = trace
        return trace

    def get(self, run_id: str) -> RunTrace:
        return self._runs.get(run_id) or RunTrace(None, {})

    def flush(self) -> None:
        """Send queued spans now. Call before the process exits."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:
                log.warning("trace_flush_failed", error=type(exc).__name__)

    def trace_for(self, run_id: str) -> RunTrace | None:
        """A run's trace, live or finished (the last 100 are kept in memory)."""
        return self._runs.get(run_id) or self._finished.get(run_id)

    def drop(self, run_id: str) -> None:
        trace = self._runs.pop(run_id, None)
        if trace is not None:
            self._finished[run_id] = trace
            while len(self._finished) > 100:
                self._finished.popitem(last=False)


def build_tracer(
    *, enabled: bool, api_key: str, project: str, client: Client | None = None
) -> Tracer:
    """Real tracer from settings. A client can be injected (tests)."""
    if client is None and enabled and api_key:
        client = Client(api_key=api_key)
    return Tracer(client=client, project=project, enabled=enabled)
