"""Model text that a user will read (a question, an answer option, the critic's highlight).

A crafted prompt can steer this text toward a link, markup or the name of a model. So it is
cleaned and length-capped here. Text that names a provider or a pipeline part is always dropped,
because the pipeline must stay hidden (spec section 5). Model names that are also everyday words
(llama, phoenix, lucid ...) are dropped only when the USER did not write them.
"""

import re

_URL = re.compile(
    r"(https?://|www\.)\S*|\b[\w-]+\.(com|net|org|io|ai|app|dev|co)\b\S*", re.IGNORECASE
)
_MARKUP = re.compile(r"[<>\[\]`*_#|{}\\]")
# Providers and pipeline parts: never shown, whatever the user wrote.
_ALWAYS_HIDDEN = re.compile(
    r"\b(openai|openrouter|cloudflare|anthropic|langsmith|langgraph|enhancer|guardrail|jev|"
    r"sdxl|stable diffusion|black forest labs|bytedance|google|workers ai|meta|alibaba|tencent)\b",
    re.IGNORECASE,
)
# Model names that are also normal words. Hidden unless the user's own prompt has the word.
_MODEL_WORDS = re.compile(
    r"\b(flux|phoenix|lucid|klein|schnell|leonardo|gemini|nano banana|gpt|llama|qwen|seedream|"
    r"recraft|ideogram|claude|critic)\b",
    re.IGNORECASE,
)


def clean_ui_text(text: str, max_len: int = 140, user_text: str = "") -> str | None:
    """Plain text for the UI, or None if it must not be shown. `user_text` is what the user
    wrote: a model-like word they used themselves ("a llama in the Andes") is allowed."""
    if _ALWAYS_HIDDEN.search(text) or _URL.search(text):
        return None
    mine = user_text.lower()
    for match in _MODEL_WORDS.finditer(text):
        if re.search(rf"\b{re.escape(match.group(0).lower())}\b", mine) is None:
            return None
    cleaned = " ".join(_MARKUP.sub(" ", text).split())
    if not cleaned:
        return None
    return cleaned[:max_len].rstrip()
