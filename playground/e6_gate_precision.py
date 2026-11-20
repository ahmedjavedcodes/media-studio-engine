"""E6 (FREE: Groq gpt-oss-20b, ~60 Gate calls, no images): how precise is `prompt_is_detailed`?

    PYTHONPATH=src:playground .venv/bin/python playground/e6_gate_precision.py     (script only, not yet run)

Skipping the Brief when the Gate says "detailed" is only safe if the Gate rarely says it about a
VAGUE prompt (a false skip loses the Brief on a prompt that needed it: the costly error). Runs the
real Gate on 30 detailed and 30 vague prompts built from the lists below, paced under Groq's
8K tokens-a-minute limit for that model. Prints the true-skip rate on detailed prompts and the
false-skip rate on vague ones, and lists every false skip.

Cost: $0. About 60 Gate calls (~600 tokens each, ~36K tokens of the 200K/day), ~8 minutes.
"""

from __future__ import annotations

import asyncio
import itertools
import json

from eval_common import AUTH, MEASURE

from media_engine.agents.base import AgentConfig
from media_engine.agents.image_grand.gate import Gate
from media_engine.contracts import Goal
from media_engine.gateway.llm_gateway import LLMGateway, chain_resolver
from media_engine.gateway.router import Gateway
from media_engine.ports import ListEventSink
from media_engine.schemas import GateDecision
from media_engine.services.create.loader import load_create_config

SUBJECTS = [
    "a weathered fisherman mending a blue net",
    "a red vintage motorcycle",
    "an elderly woman reading by a window",
    "a snow leopard resting on a cliff",
    "a ceramic teapot with a chipped spout",
    "a bustling night market stall selling lanterns",
]
SETTINGS = [
    "on a misty harbour dock at first light",
    "against a bare brick wall in an alley",
    "in a sunlit room with dusty wooden floors",
    "among tall pines after fresh snowfall",
    "on a rain-soaked street lit by neon signs",
]
STYLES = [
    "shot with a 35mm lens at eye level, shallow depth of field, soft film grain",
    "wide establishing shot from a low angle, golden hour light from the left",
    "close-up portrait lens, warm tungsten lamp light, muted teal and amber palette",
    "overhead view, crisp studio flash lighting, realistic photographic style",
    "oil painting style with visible brushwork, dramatic side lighting",
    "cinematic framing, anamorphic lens flare, cool blue shadows",
]
VAGUE = [
    "a dog", "a nice house", "something cool", "a forest", "my cat", "a car", "a beach",
    "a robot", "a girl", "sunset", "a castle", "food", "a cute logo", "mountains", "a city",
    "a flower", "an old man", "a dragon", "space", "a boat", "a chair", "a bird", "a river",
    "a party", "a tree", "an island", "a bridge", "a horse", "winter", "a street",
]  # fmt: skip


def detailed() -> list[str]:
    combos = itertools.islice(itertools.product(SUBJECTS, SETTINGS, STYLES), 0, 180, 6)
    return [f"{s} {w}, {y}" for s, w, y in combos][:30]


async def main() -> None:
    cfg = load_create_config()
    gateway = Gateway()
    gate = Gate(
        AgentConfig.from_policy("gate", cfg.agent("gate"), GateDecision),
        LLMGateway(gateway, chain_resolver(cfg.agents)),
        ListEventSink(),
    )
    stop = cfg.policy.stopping_condition("route", "free", "a safety verdict and a route")
    cases = [(p, True) for p in detailed()] + [(p, False) for p in VAGUE]
    rows = []
    try:
        for prompt, truth in cases:
            env = await gate.run(
                Goal(task="gate", instructions="classify"), stop, AUTH, {"prompt": prompt}
            )
            said = bool(env.output.get("prompt_is_detailed")) if env.output else None
            rows.append(
                {
                    "prompt": prompt,
                    "truly_detailed": truth,
                    "gate_said": said,
                    "words": len(prompt.split()),
                    "status": env.state.status.value,
                }
            )
            await asyncio.sleep(8)  # ~7 calls a minute keeps gpt-oss-20b under 8K tokens a minute
    finally:
        await gateway.aclose()
    ok = [r for r in rows if r["gate_said"] is not None]
    det = [r for r in ok if r["truly_detailed"]]
    vag = [r for r in ok if not r["truly_detailed"]]
    report = {
        "detailed_prompts": len(det),
        "true_skip_rate": round(sum(r["gate_said"] for r in det) / max(1, len(det)), 3),
        "vague_prompts": len(vag),
        "false_skip_rate": round(sum(r["gate_said"] for r in vag) / max(1, len(vag)), 3),
        "false_skips": [r["prompt"] for r in vag if r["gate_said"]],
        "undecided": len(rows) - len(ok),
    }
    (MEASURE / "e6.json").write_text(json.dumps({"report": report, "rows": rows}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
