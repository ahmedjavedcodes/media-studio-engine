"""GET /v1/runs/{id}/events. SSE stream. Fake events for setup."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agents.grand.graph import build_dummy_graph
from app.api.runs import RUNS
from app.contracts.events import RunEvent

router = APIRouter(prefix="/v1")


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> EventSourceResponse:
    req = RUNS.get(run_id)
    if req is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def gen() -> AsyncIterator[dict[str, str]]:
        yield {
            "event": "progress",
            "data": RunEvent(type="progress", run_id=run_id, message="started").model_dump_json(),
        }
        out = await build_dummy_graph().ainvoke({"prompt": req.prompt, "reply": ""})
        yield {
            "event": "result",
            "data": RunEvent(type="result", run_id=run_id, message=out["reply"]).model_dump_json(),
        }

    return EventSourceResponse(gen())
