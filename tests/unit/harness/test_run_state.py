"""M1: RunState must serialise to plain JSON and reload identically at every step boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from media_engine.contracts import (
    AuthContext,
    Envelope,
    EnvelopeState,
    Spend,
    Status,
)
from media_engine.harness import Budget, PlanStep, RunState
from media_engine.ports import InMemoryRunStore


def _auth() -> AuthContext:
    return AuthContext(plan="pro", execution_mode="auto", model_demand="balanced")


def _state() -> RunState:
    return RunState(
        run_id="run_1",
        user_id="user_1",
        auth=_auth(),
        budget=Budget(
            tokens_total=10_000, cost_cap_usd=0.10, neurons_total=600, reserve_fraction=0.30
        ),
        plan=[
            PlanStep(step=1, agent="prompt_enhancer", public_label="Understanding your request"),
            PlanStep(step=2, agent="image_creator", public_label="Creating your image"),
            PlanStep(step=3, agent="critic", public_label="Finishing up"),
        ],
    )


def _assert_round_trip(state: RunState) -> None:
    raw = state.model_dump_json()
    assert RunState.model_validate_json(raw) == state
    assert RunState.model_validate_json(raw).model_dump_json() == raw


async def test_round_trips_after_every_step_boundary() -> None:
    store = InMemoryRunStore()
    state = _state()

    async def checkpoint() -> None:
        _assert_round_trip(state)
        await store.save(state)
        loaded = await store.load(state.run_id)
        assert loaded == state
        assert loaded is not state  # store hands back an independent copy

    await checkpoint()

    for step, ref in zip(state.plan, ["enh_1", "img_" + "a" * 32, "crit_1"], strict=True):
        step.status = "running"
        await checkpoint()
        step.status = "done"
        step.output_ref = ref
        step.spend = Spend(tokens=120, cost_usd=0.0, duration_ms=900, neurons=12.5)
        state.budget.charge(step.spend)
        state.events.append({"run_id": state.run_id, "step": step.step, "phase": "finished"})
        await checkpoint()

    state.status = Status.DONE
    await checkpoint()


async def test_round_trips_a_frozen_cancelled_plan() -> None:
    state = _state()
    state.plan[0].status = "done"
    state.plan[0].output_ref = "enh_1"
    state.plan[1].status = "cancelled"
    state.plan[1].branches_total = 4
    state.plan[1].branches_done = 1
    state.plan[1].branches_aborted = 3
    state.plan[1].partial_outputs = ["img_" + "b" * 32]
    state.plan[2].status = "skipped"
    state.cancel_requested = True
    state.cancelled_at_step = 2
    state.status = Status.CANCELLED
    _assert_round_trip(state)


async def test_load_unknown_run_is_none() -> None:
    assert await InMemoryRunStore().load("nope") is None


def test_running_and_terminal_statuses_both_parse() -> None:
    state = _state()
    assert RunState.model_validate_json(state.model_dump_json()).status == "running"
    state.status = Status.REJECTED
    assert RunState.model_validate_json(state.model_dump_json()).status is Status.REJECTED


def test_auth_context_is_frozen() -> None:
    auth = _auth()
    with pytest.raises(ValidationError):
        auth.plan = "ultra"  # type: ignore[misc]


def test_plan_step_rejects_non_public_label() -> None:
    with pytest.raises(ValidationError):
        PlanStep(step=1, agent="critic", public_label="critic")  # type: ignore[arg-type]


def test_spend_defaults_are_not_shared() -> None:
    a = PlanStep(step=1, agent="x", public_label="Finishing up")
    b = PlanStep(step=2, agent="y", public_label="Finishing up")
    a.spend.tokens = 5
    a.partial_outputs.append("img_1")
    assert b.spend.tokens == 0 and b.partial_outputs == []


# -- Budget -------------------------------------------------------------------------------------


def _budget(**kw: float) -> Budget:
    values: dict[str, float] = {
        "tokens_total": 1000,
        "cost_cap_usd": 1.0,
        "neurons_total": 100.0,
        "reserve_fraction": 0.30,
    }
    values.update(kw)
    return Budget(**values)  # type: ignore[arg-type]


def test_budget_has_no_code_default_for_reserve() -> None:
    with pytest.raises(ValidationError):
        Budget(tokens_total=1000, cost_cap_usd=1.0, neurons_total=100.0)  # type: ignore[call-arg]


def test_budget_holds_back_reserve_for_retries() -> None:
    budget = _budget()
    assert budget.remaining_for_steps() == (700, pytest.approx(0.70), pytest.approx(70.0))
    assert budget.can_afford(700, 0.70, 70.0)
    assert not budget.can_afford(701, 0.0)
    assert budget.can_afford(1000, 1.0, 100.0, use_reserve=True)
    assert not budget.can_afford(1001, 0.0, use_reserve=True)


def test_budget_charge_reduces_remaining_and_clamps_at_zero() -> None:
    budget = _budget()
    budget.charge(Spend(tokens=650, cost_usd=0.5, neurons=10))
    assert budget.remaining_for_steps() == (50, pytest.approx(0.20), pytest.approx(60.0))
    budget.charge(Spend(tokens=500, cost_usd=0.9, neurons=200))
    assert budget.remaining_for_steps() == (0, 0.0, 0.0)
    assert budget.remaining_total() == (0, 0.0, 0.0)
    assert not budget.can_afford(1, 0.0, use_reserve=True)


def test_zero_cost_cap_still_allows_free_steps() -> None:
    budget = _budget(cost_cap_usd=0.0)
    assert budget.can_afford(100, 0.0)
    assert not budget.can_afford(100, 0.001)


def test_neurons_cap_image_generation_even_when_tokens_and_cost_are_zero() -> None:
    # Image generation on the free path spends 0 tokens and $0: only neurons move.
    budget = _budget(neurons_total=200.0)  # 140 usable before the reserve
    assert budget.can_afford(0, 0.0, 60.0)
    budget.charge(Spend(neurons=60.0))
    budget.charge(Spend(neurons=60.0))
    assert budget.neurons_spent == 120.0
    assert not budget.can_afford(0, 0.0, 60.0)  # refused BEFORE the third generation call
    assert budget.can_afford(0, 0.0, 60.0, use_reserve=True)  # a retry may use the reserve


def test_spend_counts_llm_calls() -> None:
    assert Spend(llm_calls=1).plus(Spend(llm_calls=2)).llm_calls == 3


# -- Envelope never carries image bytes ---------------------------------------------------------


def _env(output: dict) -> Envelope:
    return Envelope(
        output=output,
        summary="Done.",
        state=EnvelopeState(status=Status.DONE, spend=Spend()),
    )


def test_envelope_accepts_references_and_prose() -> None:
    env = _env(
        {
            "image_ids": ["img_" + "c" * 32],
            "issues": ["The mug handle is missing " * 40],
            "score": 8,
        }
    )
    assert env.output["score"] == 8


@pytest.mark.parametrize(
    "output",
    [
        {"image": b"\x89PNG"},
        {"nested": {"images": [bytearray(b"x")]}},
        {"image": "data:image/png;base64,iVBORw0KGgo="},
        {"image": "iVBORw0KGgo" * 100},
        {"image": "\n".join(["iVBORw0KGgoAAAANSUhEUgAA" * 3] * 20)},  # MIME-wrapped
        {"image": "iVBOR-w0KG_go" * 100},  # base64url
    ],
)
def test_envelope_rejects_image_bytes(output: dict) -> None:
    with pytest.raises(ValidationError):
        _env(output)


def test_envelope_revalidates_on_assignment() -> None:
    env = _env({"image_ids": []})
    with pytest.raises(ValidationError):
        env.output = {"image": b"\x89PNG"}


def test_long_prose_with_newlines_is_not_base64() -> None:
    _env({"critique": "\n".join(["The cup looks fine but the light is flat."] * 40)})
