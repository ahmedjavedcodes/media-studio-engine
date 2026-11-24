"""POST /v1/runs. HTTP layer only. Run store is in memory for setup.

The guardrail runs first. A request that is not approved never gets a run_id.
"""

import base64
import binascii
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.contracts.request import GenerationRequest
from app.core.errors import BadInput
from app.guardrails.agent import GuardDeps, guard
from app.guardrails.deps import get_guard_deps
from app.guardrails.schemas import GuardInput

router = APIRouter(prefix="/v1")

GuardDepends = Annotated[GuardDeps, Depends(get_guard_deps)]

RUNS: dict[str, GenerationRequest] = {}


def decode_images(images: list[str]) -> list[bytes]:
    out: list[bytes] = []
    for item in images:
        payload = item.split(",", 1)[1] if item.startswith("data:") and "," in item else item
        try:
            out.append(base64.b64decode(payload, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise BadInput("An image is not valid base64.") from exc
    return out


@router.post("/runs")
async def create_run(
    req: GenerationRequest,
    deps: GuardDepends,
    x_user_id: Annotated[str, Header()] = "local-dev",
) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    # Raises 400 / 403 / 422 / 429 / 503. Nothing below runs unless every check approved.
    await guard(
        GuardInput(
            request_id=run_id,
            user_id=x_user_id,
            prompt=req.prompt,
            images=decode_images(req.images),
        ),
        deps,
    )
    RUNS[run_id] = req  # the generator gets the raw prompt and the original images
    return {"run_id": run_id}
