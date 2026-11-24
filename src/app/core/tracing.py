"""LangSmith tracing for the txt2img run. One trace per run, one span per step.

Spans are explicit RunTree children, not callbacks. That keeps ONE trace for the whole run even
when the graph pauses at `interrupt()` and a second request resumes it. Span names:
code_checks, router, enhancer, interrupt, code_guard, generate (with one `generate_image` child
per variation), critic, settle.

Rules: never log image bytes (ids and thumbnail URLs only), and traces never go to the SSE stream.
"""

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


class Span:
    """One step. Collects outputs and metadata, and is posted to LangSmith when it ends."""

    def __init__(self, tree: RunTree | None) -> None:
        self._tree = tree
        self.outputs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.ended = False

    def log(self, **outputs: Any) -> None:
        self.outputs.update(outputs)

    def meta(self, **metadata: Any) -> None:
        self.metadata.update(metadata)

    def end(self, error: str | None = None) -> None:
        if self.ended:
            return
        self.ended = True
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


class RunTrace:
    """The trace of one run. Lives across an interrupt and its resume."""

    def __init__(self, root: RunTree | None, metadata: dict[str, Any]) -> None:
        self._root = root
        self.metadata = dict(metadata)
        self._spans: dict[str, Span] = {}

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
    ) -> Span:
        if self._root is None:
            span = Span(None)
        else:
            parent_tree = self._spans[parent]._tree if parent in self._spans else self._root
            try:
                tree = (parent_tree or self._root).create_child(
                    name,
                    run_type,  # type: ignore[arg-type]
                    inputs=scrub(inputs or {}),
                    extra={"metadata": scrub(self.metadata)},
                    tags=TRACE_TAGS,
                )
                tree.post()
                span = Span(tree)
            except Exception as exc:  # tracing must never break a run
                log.warning("trace_span_start_failed", error=type(exc).__name__)
                span = Span(None)
        self._spans[key or name] = span
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
        for span in list(self._spans.values()):  # close anything still open (a timed-out interrupt)
            span.end(error=error)
        if self._root is None:
            return
        try:
            self._root.end(outputs=scrub(outputs or {}), error=error, metadata=scrub(self.metadata))
            self._root.patch()
        except Exception as exc:
            log.warning("trace_finish_failed", error=type(exc).__name__)


class Tracer:
    """Makes one RunTrace per run. Off (no-op spans) unless tracing is on and a client exists."""

    def __init__(self, *, client: Client | None, project: str, enabled: bool) -> None:
        self._client = client
        self._project = project
        self._enabled = enabled and client is not None
        self._runs: dict[str, RunTrace] = {}

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
        trace = RunTrace(root, metadata)
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

    def drop(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


def build_tracer(
    *, enabled: bool, api_key: str, project: str, client: Client | None = None
) -> Tracer:
    """Real tracer from settings. A client can be injected (tests)."""
    if client is None and enabled and api_key:
        client = Client(api_key=api_key)
    return Tracer(client=client, project=project, enabled=enabled)
