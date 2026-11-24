"""System prompts for the enhancer and the critic.

Pulled from LangSmith Hub with a version tag, so a prompt can change without a deploy. When the
Hub is off, has no key or fails, the file in `prompts/create/txt2img/` is used. The version that
was used is recorded in the trace metadata either way.
"""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from langsmith import Client

from app.core.config import prompts_dir

log = structlog.get_logger("prompts")

LOCAL_FILES = {"enhancer": "create/txt2img/enhancer.md", "critic": "create/txt2img/critic.md"}


@dataclass(frozen=True)
class PromptText:
    text: str
    version: str  # "dev@1a2b3c4d" from the Hub, or "local-1a2b3c4d" from the file


def _find_template(manifest: Any) -> str | None:
    """First `template` string in a serialized LangChain prompt manifest."""
    if isinstance(manifest, dict):
        tpl = manifest.get("template")
        if isinstance(tpl, str):
            return tpl
        for value in manifest.values():
            found = _find_template(value)
            if found:
                return found
    elif isinstance(manifest, list):
        for item in manifest:
            found = _find_template(item)
            if found:
                return found
    return None


def _local(name: str) -> PromptText:
    text = (prompts_dir() / LOCAL_FILES[name]).read_text("utf-8")
    return PromptText(text=text, version="local-" + hashlib.sha256(text.encode()).hexdigest()[:8])


class PromptStore:
    """`get(name)` is cheap and sync. Call `load()` once at startup to pull from the Hub."""

    def __init__(
        self,
        *,
        source: str = "local",
        tag: str = "dev",
        refs: dict[str, str] | None = None,
        client_factory: Callable[[], Client] | None = None,
    ) -> None:
        self._source = source
        self._tag = tag
        self._refs = refs or {"enhancer": "media-enhancer", "critic": "media-critic"}
        self._client_factory = client_factory
        self._cache: dict[str, PromptText] = {}

    async def load(self) -> None:
        for name in LOCAL_FILES:
            self._cache[name] = await asyncio.to_thread(self._load_one, name)

    def get(self, name: str) -> PromptText:
        if name not in self._cache:
            self._cache[name] = self._load_one(name)  # not preloaded: this is the local file
        return self._cache[name]

    def _load_one(self, name: str) -> PromptText:
        if self._source != "hub" or self._client_factory is None:
            return _local(name)
        ref = f"{self._refs[name]}:{self._tag}"
        try:
            commit = self._client_factory().pull_prompt_commit(ref)
            text = _find_template(commit.manifest)
            if not text:
                raise ValueError("no template in the prompt")
            return PromptText(text=text, version=f"{self._tag}@{commit.commit_hash[:8]}")
        except Exception as exc:  # the Hub is a nice-to-have. Never stop a run for it
            log.warning("prompt_hub_failed_using_local", ref=ref, error=type(exc).__name__)
            return _local(name)
