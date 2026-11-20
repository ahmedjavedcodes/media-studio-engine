"""Decider: picks one option for ambiguous free text (the routing seam, CLAUDE.md).

Used at two levels: the Image Grand Agent picks a child, the Image Generation Agent picks an
operation. Only AMBIGUOUS input reaches a Decider; an explicit choice (a button, a slash
command) is routed by plain code and never asks.

`choose` returns one of `options`, or None when it cannot decide (no key, an outage, a timeout,
an answer that is not one of the options). The caller then takes its safe default. A Decider
never raises for those cases and is not an LLM call: it is not counted in the run's LLM calls.

TODO(step 7): `JevDecider` (TypeSafe AI) is the first implementation. Until it is wired, no
Decider is passed and every ambiguous request takes the safe default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Decider(Protocol):
    async def choose(self, question: str, text: str, options: Sequence[str]) -> str | None: ...


async def decide(
    decider: Decider | None, question: str, text: str, options: Sequence[str]
) -> str | None:
    """Ask `decider`, or None without one. An answer outside `options` counts as no answer."""
    if decider is None:
        return None
    answer = await decider.choose(question, text, options)
    return answer if answer in options else None


__all__ = ["Decider", "decide"]
