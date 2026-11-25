"""POST /v1/runs, resume, and reading a finished run. HTTP layer only.

The guardrail runs first. A request that is not approved never gets a run_id.
"""

import base64
import binascii
import re
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.api.deps import get_create_deps, get_run_service, get_token_signer
from app.api.run_service import RunNotWaiting, RunService
from app.contracts.request import GenerationRequest, ResumeRequest
from app.core.errors import BadInput
from app.core.tokens import TokenSigner
from app.guardrails.agent import GuardDeps, guard, guard_detailed
from app.guardrails.deps import get_guard_deps
from app.guardrails.schemas import GuardInput

router = APIRouter(prefix="/v1")

GuardDepends = Annotated[GuardDeps, Depends(get_guard_deps)]
Service = Annotated[RunService, Depends(get_run_service)]
Deps = Annotated[CreateDeps, Depends(get_create_deps)]
Signer = Annotated[TokenSigner, Depends(get_token_signer)]

MAX_ANSWER_CHARS = 200


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
    service: Service,
    signer: Signer,
    x_user_id: Annotated[str, Header()] = "local-dev",
) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    images = decode_images(req.images)
    service.preflight(req)  # 400 if the picked model cannot take these images. No cost yet
    # Raises 400 / 403 / 422 / 429 / 503. Nothing below runs unless every check approved.
    guard_started = time.time()
    _, decision = await guard_detailed(
        GuardInput(request_id=run_id, user_id=x_user_id, prompt=req.prompt, images=images),
        deps,
    )
    guard_info = {
        "started_at": guard_started,
        "ended_at": time.time(),
        "outputs": {
            "decision": decision.decision.value,
            "category": decision.category.value,
            "confidence": decision.confidence,
            "timings_ms": decision.timings_ms,
            "checks": [  # one per model that looked at the request
                {
                    "source": v.source,
                    "model": v.model,
                    "decision": v.decision.value,
                    "category": v.category.value,
                    "latency_ms": v.latency_ms,
                    "confidence": v.confidence,
                }
                for v in decision.verdicts
            ],
            "model": ", ".join(v.model for v in decision.verdicts) or None,
        },
    }
    service.start(run_id, x_user_id, req, guard=guard_info)  # raw prompt and the images
    # The token goes in the URL of /events and of every image: they send no header.
    return {"run_id": run_id, "token": signer.sign(run_id, x_user_id)}


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    body: ResumeRequest,
    deps: GuardDepends,
    service: Service,
    signer: Signer,
    x_user_id: Annotated[str, Header()] = "local-dev",
) -> dict[str, str]:
    record = service.runs.get(run_id)
    if record is None or record.user_id != x_user_id:
        raise HTTPException(status_code=404, detail="run not found")
    if record.status == "expired":
        raise HTTPException(status_code=410, detail="This request timed out. Please start again.")
    if record.status != "waiting":  # a quick early answer. The lock in resume() is the real check
        raise HTTPException(status_code=409, detail="This run is not waiting for an answer.")

    answers: dict[str, str] = {}
    for q in record.questions:
        picked = body.answers.get(q["id"])
        if picked is None:
            raise BadInput("Please answer every question.")
        labels = {o["id"]: o["label"] for o in q["options"]}
        if picked in labels and picked != "other":
            answers[q["id"]] = labels[picked]
            continue
        text = body.answers.get(f"{q['id']}_text", picked if picked not in labels else "").strip()
        if not text:
            raise BadInput("Please type your answer.")
        text = text[:MAX_ANSWER_CHARS]
        # Typed text is user input, so it goes through the guardrail like any prompt.
        await guard(
            GuardInput(request_id=f"{run_id}-{q['id']}", user_id=x_user_id, prompt=text),
            deps,
        )
        answers[q["id"]] = text
    try:
        await service.resume(run_id, answers)  # checks again under the run's lock
    except RunNotWaiting as exc:
        if exc.status == "expired":
            raise HTTPException(
                status_code=410, detail="This request timed out. Please start again."
            ) from exc
        raise HTTPException(
            status_code=409, detail="This run is not waiting for an answer."
        ) from exc
    return {"status": "running", "run_id": run_id, "token": signer.sign(run_id, x_user_id)}


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    service: Service,
    deps: Deps,
    x_user_id: Annotated[str, Header()] = "local-dev",
) -> dict[str, Any]:
    """The saved result of a run, so it can be opened again. An expired run returns its draft."""
    record = service.runs.get(run_id)
    result = deps.store.read_json(run_id, "result.json")
    if record is None and result is None:
        raise HTTPException(status_code=404, detail="run not found")
    owner = record.user_id if record else _result_owner(result)
    if owner != x_user_id:  # same check as resume. Someone else's run does not exist for you
        raise HTTPException(status_code=404, detail="run not found")
    out: dict[str, Any] = {"run_id": run_id, "status": record.status if record else "done"}
    if result is not None:
        out["result"] = _public_result(deps, run_id, x_user_id, result)
    if record is not None and record.status == "expired":
        out["draft"] = {
            "prompt": record.request.prompt,
            "plan": record.request.plan.value,
            "settings": record.request.settings.model_dump(),
        }
    return out


def _result_owner(result: dict[str, Any] | None) -> str | None:
    docs = (result or {}).get("documents") or []
    return docs[0].get("user_id") if docs else None


def _public_result(
    deps: CreateDeps, run_id: str, user_id: str, result: dict[str, Any]
) -> dict[str, Any]:
    """The result as the frontend sees it: signed image urls and the recommendation. No model
    names, prompts, seeds or scores (spec section 5)."""

    def signed(name: str) -> str:
        return deps.tokens.signed_path(deps.store.url(run_id, name), run_id, user_id)

    images = []
    for i, doc in enumerate(result.get("documents", []), start=1):
        layer = doc["versions"][0]["layers"][0]
        images.append(
            {
                "id": f"img_{i}",
                "url": signed(layer["file"]),
                "thumb_url": signed(layer["thumb"]),
                "width": layer["width"],
                "height": layer["height"],
            }
        )
    return {"images": images, "recommendation": result.get("recommendation")}


_IMAGE_NAME = re.compile(r"^(img|thumb)_\d+\.(jpg|png|webp|gif|svg)$")


@router.get("/files/{run_id}/{name}")
async def get_file(
    run_id: str, name: str, deps: Deps, service: Service, t: Annotated[str, Query()] = ""
) -> Response:
    """Only images of a finished, safety-checked run, and only with a valid token for that run."""
    record = service.runs.get(run_id)
    owner = record.user_id if record else _result_owner(deps.store.read_json(run_id, "result.json"))
    if not deps.tokens.verify(t, run_id, owner):
        raise HTTPException(status_code=404, detail="not found")
    data = deps.store.read_released(run_id, name) if _IMAGE_NAME.match(name) else None
    if data is None:
        raise HTTPException(status_code=404, detail="not found")
    media = {
        "jpg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "svg": "image/svg+xml",
    }[name.rsplit(".", 1)[-1]]
    headers = {
        "Cache-Control": "private, max-age=3600",
        "X-Content-Type-Options": "nosniff",
        # An SVG opened on its own must not run anything. A sandboxed, script-less policy.
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
    }
    return Response(content=data, media_type=media, headers=headers)
