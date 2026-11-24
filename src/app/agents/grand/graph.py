"""The Grand Agent graph: the router chain, then the leaf agent.

Phase 1 is a pass-through. Every level is plain code (see agents/routing.py). The txt2img
sub-agent sits inside this graph, so its `interrupt()` pauses the whole graph and the
checkpointer saves it. No escalation up the tree is needed.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.generation.create.emit import emit
from app.agents.image_grand.generation.create.state import CreateState
from app.agents.image_grand.generation.create.txt2img import compile_txt2img
from app.agents.routing import RouteInput, route_request
from app.contracts.events import RunEvent


def build_grand_graph(
    deps: CreateDeps, checkpointer: BaseCheckpointSaver[Any] | None
) -> CompiledStateGraph[CreateState, None, CreateState, CreateState]:
    def entry(state: CreateState) -> str:
        return route_request(RouteInput(has_images=state.get("has_images", False)))

    async def unavailable(state: CreateState) -> dict[str, Any]:
        # img2img is not built yet. No credits were held, so nothing to refund.
        err = {
            "code": "not_available",
            "message": "Editing with your own photo is not available yet. Try a text prompt.",
        }
        emit(
            RunEvent(type="error", run_id=state["run_id"], code=err["code"], message=err["message"])
        )
        return {"error": err}

    g = StateGraph(CreateState)
    g.add_node("txt2img", compile_txt2img(deps))
    g.add_node("unavailable", unavailable)
    g.add_conditional_edges(
        START, entry, {"txt2img": "txt2img", "img2img": "unavailable", "unsupported": "unavailable"}
    )
    g.add_edge("txt2img", END)
    g.add_edge("unavailable", END)
    return g.compile(checkpointer=checkpointer)
