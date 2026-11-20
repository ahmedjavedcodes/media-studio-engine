"""M1: harness config loads, is complete for every plan, and only names known gateway chains."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from media_engine.gateway.registry import get_card
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


LLM_AGENTS = ("router", "prompt_enhancer", "critic")


def test_agents_run_on_groq_first_with_a_different_model_each(cfg) -> None:  # type: ignore[no-untyped-def]
    # Groq rate limits are per model: one Groq model per agent multiplies the free quota.
    chains = load_chains()
    expected = {
        "router": "openai/gpt-oss-20b",
        "prompt_enhancer": "openai/gpt-oss-120b",
        "critic": "qwen/qwen3.8-27b",
    }
    for name, model in expected.items():
        for plan in PLANS:
            first = chains[cfg.agent(name).chain_for(plan)][0]
            assert (first.provider, first.model) == ("groq", model), f"{name}/{plan}"


def test_no_agent_chain_uses_another_agents_groq_model(cfg) -> None:  # type: ignore[no-untyped-def]
    chains = load_chains()
    groq_models: dict[str, set[str]] = {}
    for name in LLM_AGENTS:
        for plan in PLANS:
            refs = chains[cfg.agent(name).chain_for(plan)]
            groq_models.setdefault(name, set()).update(
                r.model for r in refs if r.provider == "groq"
            )
    for a in LLM_AGENTS:
        for b in LLM_AGENTS:
            if a < b:
                assert not groq_models[a] & groq_models[b], f"{a} and {b} share a Groq model"


def test_cloudflare_stays_as_the_agent_fallback() -> None:
    chains = load_chains()
    for task in ("reasoning", "fast_llm", "vision"):
        providers = [r.provider for r in chains[task]]
        assert providers[0] == "groq" and "cloudflare" in providers, task


def test_free_images_stay_on_cloudflare_flux_schnell_first() -> None:
    first = load_chains()["image_create"][0]
    assert (first.provider, first.model) == ("cloudflare", "@cf/black-forest-labs/flux-1-schnell")


def test_safeguard_model_is_off_the_hot_path(cfg) -> None:  # type: ignore[no-untyped-def]
    # Groq allows gpt-oss-safeguard-20b 3 requests/minute: only the Gate's `review` path may use
    # it, so it lives in its own chain and no agent's chain contains it.
    chains = load_chains()
    safeguard = "openai/gpt-oss-safeguard-20b"
    assert [(r.provider, r.model) for r in chains["safety_review"]] == [("groq", safeguard)]
    for task, refs in chains.items():
        if task != "safety_review":
            assert safeguard not in {r.model for r in refs}, task
    used = {cfg.agent(n).chain_for(p) for n in cfg.agents for p in PLANS}
    assert "safety_review" not in used


def test_every_vision_chain_model_accepts_images() -> None:
    for ref in load_chains()["vision"]:
        if ref.is_placeholder:
            continue
        card = get_card(ref.provider, ref.model)
        assert card is not None and card.vision, f"{ref.provider}/{ref.model}"


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
