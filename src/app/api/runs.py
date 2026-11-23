"""POST /v1/runs. HTTP layer only. Run store is in memory for setup."""

import uuid

from fastapi import APIRouter

from app.contracts.request import GenerationRequest

router = APIRouter(prefix="/v1")

RUNS: dict[str, GenerationRequest] = {}


@router.post("/runs")
async def create_run(req: GenerationRequest) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    RUNS[run_id] = req
    return {"run_id": run_id}
