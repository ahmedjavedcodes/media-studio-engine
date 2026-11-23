"""Dummy graph for setup. Replaced by the real Grand Agent pass-through in phase 1."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class DummyState(TypedDict):
    prompt: str
    reply: str


def _echo(state: DummyState) -> DummyState:
    return {"prompt": state["prompt"], "reply": f"echo: {state['prompt']}"}


def build_dummy_graph() -> CompiledStateGraph[DummyState, None, DummyState, DummyState]:
    g = StateGraph(DummyState)
    g.add_node("echo", _echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g.compile()
