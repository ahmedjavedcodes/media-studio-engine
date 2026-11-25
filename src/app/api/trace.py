"""GET /v1/runs/{id}/trace: the dev trace panel's data. A step-by-step record of one run.

DEV ONLY. Spec section 5 says the pipeline is never exposed to users, and that still holds: this
is off unless TRACE_UI=true, it is never on in prod (APP_ENV=prod), and it is its own endpoint,
not part of the SSE stream. Access needs the same signed token as /events (`?t=`). Everything in
it is scrubbed: no image bytes, only ids and numbers.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.api.deps import get_create_deps, get_run_service, get_token_signer
from app.api.run_service import RunService
from app.core.tokens import TokenSigner

router = APIRouter(prefix="/v1")


@router.get("/runs/{run_id}/trace")
async def get_trace(
    run_id: str,
    deps: Annotated[CreateDeps, Depends(get_create_deps)],
    service: Annotated[RunService, Depends(get_run_service)],
    signer: Annotated[TokenSigner, Depends(get_token_signer)],
    t: Annotated[str, Query()] = "",
) -> JSONResponse:
    if not deps.settings.trace_ui_on:  # off looks the same as "does not exist"
        raise HTTPException(status_code=404, detail="not found")
    record = service.runs.get(run_id)
    live = deps.tracer.trace_for(run_id)
    snap: dict[str, Any] | None
    owner: str | None
    if record is not None and live is not None:
        snap, owner = live.snapshot(record.status), record.user_id
    else:  # an older run, or the server restarted: the saved copy
        stored = deps.store.read_json("traces", f"{run_id}.json")
        snap = dict(stored) if stored else None
        raw_owner = stored.get("user_id") if stored else None
        owner = raw_owner if isinstance(raw_owner, str) else None
    if snap is None or not signer.verify(t, run_id, owner):
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(snap, headers={"Cache-Control": "no-store"})
