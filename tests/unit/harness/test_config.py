"""M1: harness config loads, is complete for every plan, and only names known gateway chains."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from media_engine.gateway.router import load_chains
from media_engine.services.create.loader import PLANS, load_create_config


@pytest.fixture(scope="module")
def cfg():  # type: ignore[no-untyped-def]
    return load_create_config()


def test_every_agent_chain_exists_in_gateway(cfg) -> None:  # type: ignore[no-untyped-def]
    chains = load_chains()
    for name, agent in cfg.agents.items():
        for plan in PLANS:
            assert agent.chain_for(plan) in chains, f"{name}/{plan} -> unknown chain"


def test_vision_agents_use_a_vision_chain(cfg) -> None:  # type: ignore[no-untyped-def]
    assert "vision" in cfg.agent("critic").requires
    assert {cfg.agent("critic").chain_for(p) for p in PLANS} == {"vision"}


def test_image_creator_has_no_prompt(cfg) -> None:  # type: ignore[no-untyped-def]
    assert cfg.agent("image_creator").prompt is None
    with pytest.raises(ValueError):
        cfg.agent("image_creator").prompt_path()


def test_retry_caps_match_spec(cfg) -> None:  # type: ignore[no-untyped-def]
    assert cfg.policy.max_retries == {"free": 1, "pro": 2, "max": 2, "ultra": 2}


def test_lever_table_matches_spec(cfg) -> None:  # type: ignore[no-untyped-def]
    assert cfg.policy.levers_for("free") == ["reseed", "re_enhance"]
    assert cfg.policy.levers_for("pro") == ["reseed", "re_enhance", "switch_model"]
    assert cfg.policy.levers_for("max") == ["reseed", "re_enhance", "switch_model", "raise_tier"]


def test_parallel_cap_and_presets(cfg) -> None:  # type: ignore[no-untyped-def]
    assert cfg.policy.max_parallel_branches == 4
    d = cfg.presets.defaults
    assert cfg.presets.aspect_ratios[d.aspect_ratio] == d.size


def test_free_plan_has_zero_cost_cap(cfg) -> None:  # type: ignore[no-untyped-def]
    assert cfg.policy.budgets["free"].cost_cap_usd == 0.0


def test_missing_plan_is_rejected(tmp_path: Path) -> None:
    src = Path(load_create_config.__defaults__[0])  # type: ignore[index]
    (tmp_path / "agents.yaml").write_text((src / "agents.yaml").read_text())
    harness = (src / "harness.yaml").read_text().replace(", ultra: 2}", "}", 1)
    (tmp_path / "harness.yaml").write_text(harness)
    with pytest.raises(ValidationError, match="max_retries"):
        load_create_config(config_dir=tmp_path)
