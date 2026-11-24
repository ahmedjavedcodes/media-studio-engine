"""[7] Settle and send. Spec v1.3 section 4 [7].

Save the documents (every image here already passed the per-image safety check), then charge for
what is delivered, then send the final event. The charge comes last: if anything before it fails
the run ends with `fail`, which releases the hold and deletes the files, so nobody is charged for
a run that errored.
"""

from typing import Any

import structlog

from app.agents.image_grand.generation.create.deps import CreateDeps
from app.agents.image_grand.generation.create.emit import emit
from app.agents.image_grand.generation.create.state import CreateState
from app.agents.image_grand.workers import critic as critic_worker
from app.contracts.events import ImageOut, RunEvent
from app.contracts.result import Document, Layer, RunResult, Version
from app.core.errors import OutputBlocked
from app.core.plans import Plan, credits_for
from app.core.tracing import Span
from app.local.output_store import RESULT_FILE

log = structlog.get_logger("settle")


def _image_out(
    deps: CreateDeps, run_id: str, user_id: str, layer: Layer, image_id: str
) -> ImageOut:
    def signed(name: str) -> str:  # every image URL carries the token (?t=)
        return deps.tokens.signed_path(deps.store.url(run_id, name), run_id, user_id)

    return ImageOut(
        id=image_id,
        url=signed(layer.file),
        thumb_url=signed(layer.thumb),
        width=layer.width,
        height=layer.height,
    )


async def settle_run(deps: CreateDeps, state: CreateState, span: Span) -> dict[str, Any]:
    run_id = state["run_id"]
    card = deps.registry.get(state["model_id"])
    assert card is not None
    # Every result here already passed the per-image safety check. Unsafe images were deleted.
    kept = sorted((r for r in state["results"] if r["ok"]), key=lambda r: r["index"])
    review = state["review"]
    verdicts = review["verdicts"]
    if not kept:  # the critic step ends the run first. This is a last guard
        raise OutputBlocked()

    # Paid plans: sort by rank (stable, so unranked images keep generation order). Free plans, and
    # a run whose critic could not run, keep the generation order.
    if Plan(state["plan"]) != Plan.BASIC and not review.get("failed"):
        kept.sort(
            key=lambda r: (verdicts.get(critic_worker.image_id(r["index"])) or {}).get("rank") or 99
        )

    compiled = state["compiled"]
    docs: list[Document] = []
    layers: dict[str, Layer] = {}
    for k, r in enumerate(kept, start=1):
        iid = critic_worker.image_id(r["index"])
        layer = Layer(
            file=r["file"], thumb=r["thumb"], mime=r["mime"], width=r["width"], height=r["height"]
        )
        layers[iid] = layer
        doc = Document(
            id=f"{run_id}-{k}",
            run_id=run_id,
            user_id=state["user_id"],
            versions=[
                Version(
                    layers=[layer],  # one image layer
                    model=card.id,
                    provider=r["provider"],
                    seed=r["seed"] if r["seed_sent"] else None,
                    prompt=compiled["prompt"],
                    negative_prompt=compiled["negative_prompt"],
                )
            ],
        )
        docs.append(doc)
        deps.store.write_json(run_id, f"doc_{k}.json", doc.model_dump(mode="json"))

    order = [critic_worker.image_id(r["index"]) for r in kept]
    rv = critic_worker.Review(
        verdicts={k: critic_worker.Verdict(**v) for k, v in verdicts.items()},
        best=review["best"],
        highlight=review["highlight"],
        failed=bool(review.get("failed")),
    )
    line = critic_worker.recommendation(rv, order)  # every plan gets the line
    images = [_image_out(deps, run_id, state["user_id"], layers[i], i) for i in order]

    charge = credits_for(card.tier, len(kept))  # only for delivered images
    result = RunResult(
        run_id=run_id, documents=docs, recommendation=line or None, credits_charged=charge
    )
    # Saving the result is what releases the images: only files named in it can be fetched.
    deps.store.write_json(run_id, RESULT_FILE, result.model_dump(mode="json"))
    charged = await deps.credits.settle(state["hold_id"], charge)  # last step before the event
    span.log(delivered=len(kept), charged=charged, critic_failed=bool(review.get("failed")))
    emit(RunEvent(type="ready", run_id=run_id, images=images, recommendation=line or None))
    return {}
