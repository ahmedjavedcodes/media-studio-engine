"""The fixed enhance -> generate -> critique sequence, as a LangGraph graph. One graph run is
ONE attempt. This is the only LangGraph in the harness (spec §2), and it is deliberately dumb:

- Linear: START -> enhance -> generate -> critique -> END. No conditional edges, so there is no
  branching here and exactly one way out. Retries WRAP the graph (the plain-Python supervisor
  picks a lever and runs it again); nothing loops back inside it.
- The graph state carries only `run_id`, a reference. The run's real state (RunState: plan,
  budget, spend, statuses) lives on the per-run context the supervisor passes through
  `config["configurable"]["run"]`, so there is one owner and one JSON round-trip, never two.
- Every early exit (cancel, budget refusal, safety block, step failure) is recorded on the
  context by the supervisor's step guard; the remaining nodes see it and return at once, so
  all outcomes leave through the single END edge. Nodes never raise past the graph: an
  unexpected exception is recorded as a step failure. (asyncio.CancelledError, a
  BaseException, still propagates: cancelling the task must cancel the run.)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from itertools import pairwise
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from media_engine.core.logging import get_logger

log = get_logger("agents.image_grand.graph")

CONTEXT_KEY = "run"
NODES = ("enhance", "generate", "critique")  # the static creation plan, in order


class AttemptState(TypedDict):
    run_id: str  # a reference only: never plan, budget, spend or status


NodeFn = Callable[[Any], Awaitable[None]]  # receives the supervisor's run context
OnCrash = Callable[[Any, str, Exception], None]


def build_attempt_graph(steps: Sequence[tuple[str, NodeFn]], on_crash: OnCrash) -> Any:
    """Compile the linear graph. `steps` must be exactly NODES, in order. `on_crash` records an
    unexpected exception from a node as that step's failure (so it never escapes the graph)."""
    names = tuple(name for name, _ in steps)
    if names != NODES:
        raise ValueError(f"attempt graph must be exactly {NODES}, got {names}")

    def wrap(
        name: str, fn: NodeFn
    ) -> Callable[[AttemptState, RunnableConfig], Awaitable[dict[str, Any]]]:
        async def node(state: AttemptState, config: RunnableConfig) -> dict[str, Any]:
            ctx = config["configurable"][CONTEXT_KEY]
            try:
                await fn(ctx)
            except Exception as err:  # never raise past the graph: record and fall through
                log.exception("graph.node_crashed", node=name, run_id=state["run_id"])
                on_crash(ctx, name, err)
            return {}

        return node

    graph = StateGraph(AttemptState)
    for name, fn in steps:
        graph.add_node(name, wrap(name, fn))  # type: ignore[call-overload]
    graph.add_edge(START, NODES[0])
    for before, after in pairwise(NODES):
        graph.add_edge(before, after)
    graph.add_edge(NODES[-1], END)
    return graph.compile()


async def run_attempt(graph: Any, run_id: str, ctx: Any) -> None:
    """One attempt: one pass through the graph. The outcome is left on `ctx`."""
    await graph.ainvoke({"run_id": run_id}, config={"configurable": {CONTEXT_KEY: ctx}})
