"""Declared token budgets for the image-create system prompts (IMAGE-CREATE-OPTIMIZE §8).

The schema is sent through the provider (structured output) or appended by the gateway for a
provider without it, so it is not part of a prompt file and not counted. A prompt that grows
past its budget fails the build: shorten it, or raise the budget on purpose in this file."""

from __future__ import annotations

import pytest

from media_engine.core.prompts import PROMPTS_DIR, load_prompt
from media_engine.services.create.loader import load_create_config

# text tokens, counted with o200k_base
BUDGETS = {"gate": 200, "prompt_enhancer": 600, "critic": 150, "safety_reviewer": 200}
CFG = load_create_config()


@pytest.fixture(scope="module")
def encoding():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as err:  # noqa: BLE001 - the encoding file is downloaded on first use
        pytest.skip(f"o200k_base unavailable (needs a one-time download): {err}")


@pytest.mark.parametrize("agent", sorted(BUDGETS))
def test_the_system_prompt_is_within_its_token_budget(encoding, agent):
    text = load_prompt(PROMPTS_DIR / CFG.agent(agent).prompt)
    tokens = len(encoding.encode(text))
    assert tokens <= BUDGETS[agent], f"{agent} prompt is {tokens} tokens, budget {BUDGETS[agent]}"


@pytest.mark.parametrize("agent", sorted(BUDGETS))
def test_no_prompt_describes_or_embeds_its_schema(agent):
    text = load_prompt(PROMPTS_DIR / CFG.agent(agent).prompt)
    assert "{{schema}}" not in text and "matching this schema" not in text


@pytest.mark.parametrize("agent", sorted(BUDGETS))
def test_at_most_two_few_shot_examples(agent):
    text = load_prompt(PROMPTS_DIR / CFG.agent(agent).prompt)
    assert text.lower().count("example") <= 2


def test_reasoning_effort_is_kept_minimal_for_the_three_hot_agents():
    # gpt-oss still spends a few reasoning tokens at `low` (measured 4-28 on Groq), so this
    # pins the setting, not a token count. qwen takes `none` and spent 0.
    assert CFG.agent("gate").reasoning_effort == "low"
    assert CFG.agent("prompt_enhancer").reasoning_effort == "low"
    assert CFG.agent("critic").reasoning_effort == "none"
