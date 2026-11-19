"""M2: Agent base. Malformed output is repaired once, then fails cleanly; the repair is budgeted
on total tokens (prompt + completion), not on the completion cap."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from media_engine.agents.base import Agent, AgentConfig, extract_json
from media_engine.agents.config import AgentPolicy
from media_engine.contracts import AuthContext, Envelope, Goal, Status, StoppingCondition
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.ports import ListEventSink
from media_engine.schemas import RouteDecision
from tests.unit.harness.fakes import AllDown, ScriptedChat, messages_of

AUTH = AuthContext(plan="free", execution_mode="auto", model_demand="balanced")
GOAL = Goal(task="route", instructions="classify")
VALID = json.dumps({"route": "simple"})
INVALID = json.dumps({"route": "medium"})


class RouterLike(Agent):
    async def run(
        self, goal: Goal, stop: StoppingCondition, auth: AuthContext, inputs: dict[str, Any]
    ) -> Envelope:
        result = await self.call_structured(auth, stop, inputs["prompt"], RouteDecision)
        if result.output is None:
            return self.failed(result.error or "unknown", result.spend)
        return self.envelope(Status.DONE, result.output.model_dump(), "Routed.", result.spend)


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "router.md"
    path.write_text("Classify the request. Reply as JSON matching:\n{{schema}}\n")
    return path


def make_agent(prompt_file: Path, backend: ScriptedChat) -> RouterLike:
    policy = AgentPolicy(
        prompt=None,
        requires=["structured_output"],
        models={"free": "fast_llm", "pro": "fast_llm", "max": "reasoning", "ultra": "reasoning"},
    )
    gateway = LLMGateway(backend, chain_resolver({"router": policy}))
    config = AgentConfig(
        name="router", prompt_file=prompt_file, model_policy=policy, output_model=RouteDecision
    )
    return RouterLike(config, gateway, ListEventSink())


def stop(
    max_tokens: int = 1000, completion: int = 300, deadline_ms: int = 5000
) -> StoppingCondition:
    return StoppingCondition(
        success="route chosen",
        max_tokens=max_tokens,
        max_completion_tokens=completion,
        max_retries=0,
        deadline_ms=deadline_ms,
    )


async def test_valid_first_reply_makes_one_call(prompt_file: Path) -> None:
    backend = ScriptedChat([VALID])
    env = await make_agent(prompt_file, backend).run(GOAL, stop(), AUTH, {"prompt": "a cat"})
    assert env.state.status is Status.DONE
    assert env.output == {"route": "simple"}
    assert env.state.spend.llm_calls == 1
    assert len(backend.requests) == 1
    assert env.state.spend.tokens == 100
    assert env.state.spend.neurons == 2.5


async def test_system_prompt_comes_from_file_with_schema(prompt_file: Path) -> None:
    backend = ScriptedChat([VALID])
    await make_agent(prompt_file, backend).run(GOAL, stop(), AUTH, {"prompt": "a cat"})
    system = messages_of(backend, 0)[0]
    assert system["role"] == "system"
    assert system["content"].startswith("Classify the request.")
    assert '"route"' in system["content"] and "{{schema}}" not in system["content"]
    assert backend.requests[0][1].json_mode is True


async def test_first_call_is_capped_by_completion_limit_not_step_total(prompt_file: Path) -> None:
    backend = ScriptedChat([VALID])
    await make_agent(prompt_file, backend).run(
        GOAL, stop(max_tokens=5000, completion=300), AUTH, {"prompt": "a cat"}
    )
    assert backend.requests[0][1].max_tokens == 300


async def test_malformed_then_valid_repairs_once(prompt_file: Path) -> None:
    backend = ScriptedChat([INVALID, VALID])
    env = await make_agent(prompt_file, backend).run(GOAL, stop(), AUTH, {"prompt": "a cat"})

    assert env.state.status is Status.DONE
    assert env.output == {"route": "simple"}
    assert env.state.spend.llm_calls == 2
    repair_msgs = messages_of(backend, 1)
    assert repair_msgs[-2] == {"role": "assistant", "content": INVALID}
    assert repair_msgs[-1]["role"] == "user"
    assert "route" in repair_msgs[-1]["content"]  # the validation error is handed back
    assert env.state.spend.tokens == 200


async def test_malformed_twice_fails_cleanly_after_exactly_two_calls(prompt_file: Path) -> None:
    backend = ScriptedChat([INVALID, "not json at all"])
    env = await make_agent(prompt_file, backend).run(GOAL, stop(), AUTH, {"prompt": "a cat"})

    assert env.state.status is Status.FAILED
    assert len(backend.requests) == 2  # one repair, never a second
    assert "after one repair" in (env.state.reason or "")
    assert env.output == {}
    assert env.state.spend.tokens == 200  # spend is still reported
    assert env.state.spend.llm_calls == 2


async def test_repair_cap_accounts_for_the_resent_prompt(prompt_file: Path) -> None:
    # First call: 400 prompt + 100 completion = 500 of a 2000 budget. The repair re-sends the
    # 400-token prompt + the 100-token reply + the repair text, so its cap must leave room for
    # that, i.e. well under the naive 2000 - 500 = 1500.
    backend = ScriptedChat([INVALID, VALID], prompt_tokens=400, completion_tokens=100)
    env = await make_agent(prompt_file, backend).run(
        GOAL, stop(max_tokens=2000, completion=1500), AUTH, {"prompt": "a cat"}
    )
    assert env.state.status is Status.DONE
    repair_cap = backend.requests[1][1].max_tokens
    assert repair_cap is not None
    assert repair_cap < 1500 - 500  # remaining minus at least prompt + first reply
    assert repair_cap >= 100  # still enough for a reply as long as the first


async def test_no_repair_when_resent_prompt_would_not_fit(prompt_file: Path) -> None:
    # 700 + 100 = 800 spent of 1000; the repair prompt alone is ~800+, so no room for a reply.
    backend = ScriptedChat([INVALID, VALID], prompt_tokens=700, completion_tokens=100)
    env = await make_agent(prompt_file, backend).run(
        GOAL, stop(max_tokens=1000, completion=300), AUTH, {"prompt": "a cat"}
    )
    assert env.state.status is Status.FAILED
    assert len(backend.requests) == 1
    assert "no token budget left" in (env.state.reason or "")


async def test_repair_never_exceeds_completion_cap(prompt_file: Path) -> None:
    backend = ScriptedChat([INVALID, VALID], prompt_tokens=50, completion_tokens=20)
    await make_agent(prompt_file, backend).run(
        GOAL, stop(max_tokens=100_000, completion=300), AUTH, {"prompt": "a cat"}
    )
    assert backend.requests[1][1].max_tokens == 300


async def test_gateway_failure_becomes_failed_envelope(prompt_file: Path) -> None:
    backend = ScriptedChat([AllDown("every provider is down")])
    env = await make_agent(prompt_file, backend).run(GOAL, stop(), AUTH, {"prompt": "a cat"})
    assert env.state.status is Status.FAILED
    assert "AllDown" in (env.state.reason or "")
    assert env.state.spend.llm_calls == 1


async def test_unexpected_exception_becomes_failed_envelope(prompt_file: Path) -> None:
    backend = ScriptedChat([ValueError("an unwrapped bug in a provider")])
    env = await make_agent(prompt_file, backend).run(GOAL, stop(), AUTH, {"prompt": "a cat"})
    assert env.state.status is Status.FAILED
    assert "internal error: ValueError" in (env.state.reason or "")


async def test_cancellation_is_not_swallowed(prompt_file: Path) -> None:
    class Hangs(ScriptedChat):
        async def chat(self, task, request):  # type: ignore[no-untyped-def]
            await asyncio.sleep(10)

    task = asyncio.create_task(
        make_agent(prompt_file, Hangs([])).run(GOAL, stop(), AUTH, {"prompt": "a cat"})
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_deadline_becomes_failed_envelope_and_counts_the_call(prompt_file: Path) -> None:
    class Slow(ScriptedChat):
        async def chat(self, task, request):  # type: ignore[no-untyped-def]
            await asyncio.sleep(1)
            return await super().chat(task, request)

    env = await make_agent(prompt_file, Slow([VALID])).run(
        GOAL, stop(deadline_ms=20), AUTH, {"prompt": "a cat"}
    )
    assert env.state.status is Status.FAILED
    assert "deadline" in (env.state.reason or "")
    assert env.state.spend.llm_calls == 1


@pytest.mark.parametrize(
    "text",
    [
        '{"route": "simple"}',
        '```json\n{"route": "simple"}\n```',
        'Sure! Here it is: {"route": "simple"} Hope that helps.',
        'Here is {the answer}: {"route": "simple"}',
        'Two objects {"route": "simple"} then {"route": "complex"}',
    ],
)
def test_extract_json_finds_the_first_decodable_object(text: str) -> None:
    assert json.loads(extract_json(text) or "") == {"route": "simple"}


def test_extract_json_keeps_nested_objects_whole() -> None:
    assert json.loads(extract_json('x {"a": {"b": 1}} y') or "") == {"a": {"b": 1}}


def test_extract_json_none_when_absent() -> None:
    assert extract_json("no braces here") is None
    assert extract_json("{not json}") is None


def test_agent_without_prompt_file_refuses_llm(prompt_file: Path) -> None:
    agent = make_agent(prompt_file, ScriptedChat([]))
    no_prompt = RouterLike(
        agent.config.model_copy(update={"prompt_file": None}), None, ListEventSink()
    )
    with pytest.raises(ValueError):
        no_prompt.system_prompt()
