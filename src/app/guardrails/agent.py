"""Guardrail agent: check every request before any generation call. Spec sections 3, 8, 9.

0. account check   1. clean text   1b. prep images
2. Jev (text) and Llama Guard 4 (images + clean text) run together under one deadline
3. strictest answer wins   4. strikes and cooldowns
Any failure of a check fails closed (503). A blocked or failed request costs 0 credits.
"""

import asyncio
import hashlib
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from langsmith.run_helpers import trace

from app.contracts.ports import JevPort, StrikeLedger, VisionPort
from app.core.config import Settings
from app.core.errors import (
    AccountRestricted,
    BadInput,
    ContentBlocked,
    Cooldown,
    GuardrailUnavailable,
)
from app.guardrails import policy
from app.guardrails.clean import CleanText, clean_text
from app.guardrails.image_prep import PreparedImage, batches, prep_images
from app.guardrails.merge import merge
from app.guardrails.messages import blocked_by, build_reject_body, message_for
from app.guardrails.retry import call_with_retry
from app.guardrails.schemas import Decision, GuardDecision, GuardInput, GuardVerdict
from app.guardrails.strikes import StrikeOutcome, apply_rules
from app.guardrails.text_check import check_text
from app.guardrails.vision_check import check_vision_batch, combine_batches

log = structlog.get_logger("guardrail")


@dataclass(frozen=True)
class Timing:
    deadline_s: float = policy.DEADLINE_S
    jev_timeout_s: float = policy.JEV_TIMEOUT_S
    jev_retry_delay_s: float = policy.JEV_RETRY_DELAY_S
    vision_timeout_s: float = policy.VISION_TIMEOUT_S
    vision_retry_delay_s: float = policy.VISION_RETRY_DELAY_S


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class GuardDeps:
    jev: JevPort
    vision: VisionPort
    ledger: StrikeLedger
    settings: Settings
    timing: Timing = field(default_factory=Timing)
    clock: Callable[[], datetime] = _utcnow


# ---- steps -----------------------------------------------------------------------------------


async def _check_account(deps: GuardDeps, user_id: str) -> None:
    """Step 0. Frozen or banned -> 403. Cooldown -> 429. Neither counts as a reject."""
    try:
        account = await deps.ledger.get_account(user_id)
    except Exception as exc:
        log.error("guardrail_ledger_failed", step="get_account", error=type(exc).__name__)
        raise GuardrailUnavailable() from exc
    if account.status != "active":
        raise AccountRestricted()
    if account.cooldown_until is not None:
        wait = (account.cooldown_until - deps.clock()).total_seconds()
        if wait > 0:
            raise Cooldown(math.ceil(wait))


async def _run_jev(
    deps: GuardDeps, text: CleanText, has_image: bool, deadline: float
) -> GuardVerdict:
    t = deps.timing
    return await call_with_retry(
        lambda timeout: check_text(deps.jev, text, has_image=has_image, timeout=timeout),
        timeout=t.jev_timeout_s,
        retry_delay=t.jev_retry_delay_s,
        deadline=deadline,
    )


async def _run_vision(
    deps: GuardDeps, prompt_clean: str, images: list[PreparedImage], deadline: float
) -> GuardVerdict:
    t = deps.timing

    async def one(batch: list[PreparedImage]) -> GuardVerdict:
        return await call_with_retry(
            lambda timeout: check_vision_batch(deps.vision, prompt_clean, batch, timeout=timeout),
            timeout=t.vision_timeout_s,
            retry_delay=t.vision_retry_delay_s,
            deadline=deadline,
        )

    results = await asyncio.gather(*(one(b) for b in batches(images)), return_exceptions=True)
    return combine_batches(list(results))


async def _run_checks(
    deps: GuardDeps, text: CleanText, images: list[PreparedImage]
) -> tuple[list[GuardVerdict], list[BaseException]]:
    """Step 2. Both checks together. An INSTANT reject cancels the other one."""
    deadline = time.monotonic() + deps.timing.deadline_s
    tasks: dict[asyncio.Task[GuardVerdict], str] = {
        asyncio.create_task(_run_jev(deps, text, bool(images), deadline)): "jev"
    }
    if images:
        tasks[asyncio.create_task(_run_vision(deps, text.prompt_clean, images, deadline))] = (
            "vision"
        )

    verdicts: list[GuardVerdict] = []
    failures: list[BaseException] = []
    pending = set(tasks)
    try:
        while pending:
            remaining = max(0.0, deadline - time.monotonic())
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:  # whole-agent deadline hit
                failures.append(TimeoutError("guardrail deadline"))
                break
            for task in done:
                exc = task.exception()
                if exc is not None:
                    log.error(
                        "guardrail_check_failed",
                        source=tasks[task],
                        error=type(exc).__name__,
                        detail=str(exc),
                    )
                    failures.append(exc)
                else:
                    verdicts.append(task.result())
            if any(v.decision is Decision.INSTANT_REJECT for v in verdicts):
                break
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return verdicts, failures


# ---- tracing and audit -----------------------------------------------------------------------


@contextmanager
def _span(inputs: dict[str, Any]) -> Iterator[Callable[[dict[str, Any], list[str]], None]]:
    """LangSmith trace for one check. Inputs are counts and hashes only, never prompt or images."""
    if os.environ.get("LANGSMITH_TRACING") != "true":
        yield lambda outputs, tags: None
        return
    with trace(name="guardrail", run_type="chain", inputs=inputs, tags=["agent:guardrail"]) as run:

        def end(outputs: dict[str, Any], tags: list[str]) -> None:
            run.add_tags(tags)
            run.end(outputs=outputs)

        yield end


def _summary(verdicts: list[GuardVerdict]) -> list[dict[str, Any]]:
    return [
        {
            "source": v.source,
            "model": v.model,
            "decision": v.decision.value,
            "category": v.category.value,
            "latency_ms": v.latency_ms,
        }
        for v in verdicts
    ]


def _audit(
    inp: GuardInput, text: CleanText, decision: GuardDecision, prepared: list[PreparedImage]
) -> None:
    """One audit row per reject. Hashes only: no rejected image bytes, no keys."""
    log.warning(
        "guardrail_reject",
        request_id=inp.request_id,
        user_id=inp.user_id,
        decision=decision.decision.value,
        category=decision.category.value,
        blocked_by=blocked_by(decision),
        account_action=decision.account_action,
        strike_count=decision.strike_count,
        needs_review=any(v.needs_review for v in decision.verdicts),
        verdicts=[v.model_dump(mode="json") for v in decision.verdicts],
        image_sha256=[p.sha256 for p in prepared],
        prompt_sha256=hashlib.sha256(text.prompt_raw.encode("utf-8")).hexdigest(),
        prompt_chars=len(text.prompt_raw),
    )


# ---- public API ------------------------------------------------------------------------------


async def check(inp: GuardInput, deps: GuardDeps) -> GuardDecision:
    """Run the guardrail. Returns APPROVE or a reject decision.

    Raises BadInput (400), AccountRestricted (403), Cooldown (429), GuardrailUnavailable (503).
    """
    started = time.monotonic()
    inputs = {
        "request_id": inp.request_id,
        "has_image": bool(inp.images),
        "image_count": len(inp.images),
        "prompt_chars": len(inp.prompt),
    }
    with _span(inputs) as end_span:
        await _check_account(deps, inp.user_id)

        text = clean_text(inp.prompt)
        if not text.prompt_clean and not inp.images:
            raise BadInput("Prompt is empty.")
        prepared = prep_images(inp.images)

        verdicts, failures = await _run_checks(deps, text, prepared)
        rejects = [v for v in verdicts if v.decision is not Decision.APPROVE]
        if not rejects and (failures or not verdicts):
            raise GuardrailUnavailable()  # 0 credits, no strike, never fail open

        final, category, confidence = merge(verdicts)
        result = GuardDecision(
            decision=final, category=category, confidence=confidence, verdicts=verdicts
        )
        if final is not Decision.APPROVE:
            outcome = await _apply_strikes(deps, inp, final)
            result = result.model_copy(
                update={
                    "account_action": outcome.account_action,
                    "strike_count": outcome.strike_count,
                    "retry_after": outcome.retry_after,
                    "user_message": message_for(category),
                }
            )
            _audit(inp, text, result, prepared)

        end_span(
            {
                "decision": result.decision.value,
                "category": result.category.value,
                "account_action": result.account_action,
                "steps": _summary(verdicts),
                "failed_checks": len(failures),
                "total_ms": int((time.monotonic() - started) * 1000),
            },
            [f"decision:{result.decision.value}", *(f"model:{v.model}" for v in verdicts)],
        )
        return result


async def _apply_strikes(deps: GuardDeps, inp: GuardInput, decision: Decision) -> StrikeOutcome:
    try:
        return await apply_rules(
            deps.ledger,
            deps.settings,
            user_id=inp.user_id,
            request_id=inp.request_id,
            decision=decision,
            now=deps.clock(),
        )
    except Exception as exc:  # the reject still stands if the ledger is down
        log.error("guardrail_ledger_failed", step="strikes", error=type(exc).__name__)
        return StrikeOutcome()


async def guard(inp: GuardInput, deps: GuardDeps) -> GuardInput:
    """The door in front of the generator.

    Returns `inp` itself, untouched (raw prompt, original image bytes), only when every check
    approved. Any other outcome raises, so no code path can reach the generator without APPROVE.
    """
    decision = await check(inp, deps)
    if decision.decision is not Decision.APPROVE:
        raise ContentBlocked(
            build_reject_body(decision, strike_limit=deps.settings.instant_strike_limit)
        )
    return inp
