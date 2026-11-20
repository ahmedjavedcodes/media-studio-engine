"""The USD estimate is NOT wired in, and must not quietly become something a paid path relies
on. Change this test only together with wiring it into the gateway AND checking it against a
real usage.cost value (review item 8)."""

from __future__ import annotations

from pathlib import Path

from media_engine.gateway import budget

SRC = Path(budget.__file__).resolve().parents[1]  # src/media_engine


def test_the_usd_estimate_is_declared_not_enforcing():
    assert budget.USD_ESTIMATE_ENFORCED is False
    assert "NOT ENFORCING" in (budget.__doc__ or "")
    assert "NOT ENFORCING" in (budget.estimate_image_usd.__doc__ or "")


def test_no_runtime_code_calls_the_usd_estimate():
    users = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name != "budget.py" and "estimate_image_usd" in path.read_text(encoding="utf-8")
    ]
    assert users == [], f"wired in without review: {users}"
