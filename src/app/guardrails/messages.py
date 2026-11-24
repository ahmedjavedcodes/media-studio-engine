"""The 422 body the user sees. Spec section 11.1 (strike and cooldown text updated).

Only fixed text from prompts/guardrail/user_messages.yaml. Never the model's `reason`,
scores or thresholds.
"""

from functools import lru_cache
from typing import Any

import yaml

from app.core.config import prompts_dir
from app.guardrails.schemas import Category, Decision, GuardDecision


@lru_cache
def _table() -> dict[str, Any]:
    raw = yaml.safe_load((prompts_dir() / "guardrail" / "user_messages.yaml").read_text("utf-8"))
    assert isinstance(raw, dict)
    return raw


def message_for(category: Category) -> str:
    cat = category if category.value in _table()["categories"] else Category.OTHER_UNSAFE
    return str(_table()["categories"][cat.value]["message"])


def tip_for(category: Category) -> str | None:
    cat = category if category.value in _table()["categories"] else Category.OTHER_UNSAFE
    tip = _table()["categories"][cat.value]["tip"]
    return str(tip) if tip else None


def blocked_by(decision: GuardDecision) -> str:
    sources = {v.source for v in decision.verdicts if v.decision is not Decision.APPROVE}
    if sources == {"jev"}:
        return "prompt"
    if sources == {"vision"}:
        return "image"
    return "both"


def wait_text(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = -(-seconds // 60)  # round up
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def build_reject_body(decision: GuardDecision, *, strike_limit: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": "content_blocked",
        "category": decision.category.value,
        "blocked_by": blocked_by(decision),
        "message": message_for(decision.category),
        "tip": tip_for(decision.category),
        "warning": False,
    }
    table = _table()
    if decision.decision is Decision.INSTANT_REJECT and decision.strike_count is not None:
        n = decision.strike_count
        banned = decision.account_action == "ban"
        text = table["strike_banned" if banned else "strike_warning"]
        body["warning"] = True
        body["strike_count"] = n
        body["strike_limit"] = strike_limit
        body["strike_text"] = text.format(n=n, limit=strike_limit)
        body["banned"] = banned
    elif decision.decision is Decision.STANDARD_REJECT and decision.retry_after:
        body["retry_after"] = decision.retry_after
        body["retry_text"] = table["retry_text"].format(wait=wait_text(decision.retry_after))
    return body
