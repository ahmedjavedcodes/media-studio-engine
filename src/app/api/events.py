"""GET /v1/runs/{id}/events. The SSE stream for a run.

Event names: status, image, question, ready, error, expired. Only friendly words go out. No
model names, agent names, prompts, scores or traces (spec section 5).
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_run_service, get_token_signer
from app.api.run_service import RunService
from app.core.tokens import TokenSigner

router = APIRouter(prefix="/v1")


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    service: Annotated[RunService, Depends(get_run_service)],
    signer: Annotated[TokenSigner, Depends(get_token_signer)],
    t: Annotated[str, Query()] = "",
) -> EventSourceResponse:
    """`EventSource` cannot send headers, so access is a signed token in the URL (`?t=`) made by
    POST /v1/runs for this run and user. A missing, wrong, expired or other-run token is a 404."""
    record = service.runs.get(run_id)
    if record is None or not signer.verify(t, run_id, record.user_id):
        raise HTTPException(status_code=404, detail="run not found")

    async def gen() -> AsyncIterator[dict[str, str]]:
        async for event in service.stream(run_id):
            yield {
                "event": event.type,
                "data": event.model_dump_json(exclude_none=True, exclude_defaults=True),
            }

    return EventSourceResponse(gen())
