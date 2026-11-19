"""Acceptance: no system prompt is hardcoded in a .py file; every configured prompt exists."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from media_engine.services.create.loader import load_create_config

SRC = Path(__file__).parents[3] / "src" / "media_engine"
# Longest string literal allowed in code. Anything longer reads like a prompt: move it to a file.
MAX_LITERAL_CHARS = 200
PROMPT_MARKERS = ("You are ", "Reply with ONLY")

# Agents whose prompt files land in later milestones. Remove each name as it is built.
PENDING_PROMPTS = {"router", "image_grand_agent"}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def literal_text(node: ast.AST) -> str | None:
    """The static text of a string expression: a plain literal, an f-string (with `{}` for each
    interpolation) or a `+` concatenation of those. None if it is not a string expression."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
            for v in node.values
        ]
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal_text(node.left), literal_text(node.right)
        if left is not None or right is not None:
            return (left or "{}") + (right or "{}")
    return None


def offending_literals(source: str, label: str) -> list[str]:
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    found: dict[int, str] = {}  # one report per line: outer expressions win over their parts
    for node in ast.walk(tree):
        if id(node) in docstrings or not isinstance(node, ast.Constant | ast.JoinedStr | ast.BinOp):
            continue
        text = literal_text(node)
        if text is None:
            continue
        if len(text) > MAX_LITERAL_CHARS or any(m in text for m in PROMPT_MARKERS):
            lineno = getattr(node, "lineno", 0)
            found.setdefault(lineno, f"{label}:{lineno}: {text[:60]!r}")
    return list(found.values())


def test_scanner_catches_fstrings_and_concatenation() -> None:
    sample = (
        'A = f"You are {name}, a helper"\n'
        'B = "Reply with " + "ONLY JSON"\n'
        'C = "x" * 10\n'
        f'D = "{"y" * 150}" + "{"z" * 100}"\n'
        'def f():\n    """You are a docstring, which is fine."""\n'
    )
    hits = offending_literals(sample, "sample")
    assert [h.split(":")[1] for h in hits] == ["1", "2", "4"]


def test_no_prompt_text_in_python_sources() -> None:
    offenders = [
        hit
        for path in sorted(SRC.rglob("*.py"))
        for hit in offending_literals(path.read_text(encoding="utf-8"), str(path.relative_to(SRC)))
    ]
    assert offenders == [], "move these strings into media_engine/prompts/*.md:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("name", sorted(load_create_config().agents))
def test_configured_prompt_files_exist(name: str) -> None:
    policy = load_create_config().agent(name)
    if policy.prompt is None or name in PENDING_PROMPTS:
        pytest.skip(f"{name}: no prompt yet")
    assert policy.prompt_path().is_file(), f"missing prompt for {name}: {policy.prompt}"
