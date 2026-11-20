"""D1 (spec §5.3): seeds are derived from (run_id, round, branch), so a run can be replayed
from its id, and the N branches of a run never share a seed."""

from __future__ import annotations

from media_engine.agents.image_generation.create.agent import create_image_create_agent, derive_seed
from media_engine.contracts import AuthContext
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.ports import ListEventSink
from media_engine.services.create.loader import load_create_config
from tests.unit.harness.fakes import ScriptedChat, ScriptedImages

CFG = load_create_config()
MAX = CFG.policy.max_seed


def seeds(run_id: str, n: int) -> list[int]:
    agent = create_image_create_agent(  # no seed_source: the production derivation
        CFG,
        llm=LLMGateway(ScriptedChat([]), chain_resolver(CFG.agents)),
        images=ScriptedImages([]),
        artifacts=None,  # type: ignore[arg-type]  # new_context touches no storage
        events=ListEventSink(),
    )
    auth = AuthContext(plan="max", execution_mode="auto", model_demand="balanced")
    ctx = agent.new_context(auth, {"prompt": "a fox", "n": n, "run_id": run_id})
    return [record.seed for record in ctx.state.attempts]


def test_the_same_run_id_always_gives_the_same_seeds():
    assert seeds("run_replay", 4) == seeds("run_replay", 4)


def test_different_runs_get_different_seeds():
    assert seeds("run_a", 1) != seeds("run_b", 1)


def test_the_branches_of_one_run_never_share_a_seed():
    four = seeds("run_x", 4)
    assert len(set(four)) == 4
    assert all(0 <= s <= MAX for s in four)


def test_derive_seed_is_stable_per_round_and_within_range():
    assert derive_seed("run_x", 1, MAX) == derive_seed("run_x", 1, MAX)
    assert derive_seed("run_x", 1, MAX) != derive_seed("run_x", 2, MAX)
    assert 0 <= derive_seed("run_x", 1, MAX) <= MAX
