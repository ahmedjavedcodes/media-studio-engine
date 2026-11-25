"""Shared JSON POST with errors mapped to ProviderCallFailed. Never puts secrets in messages."""

from typing import Any

import httpx

from app.core.errors import ProviderCallFailed


def _retry_after(resp: httpx.Response) -> float | None:
    """Seconds a 429 asks us to wait (Retry-After). None when absent or not a number."""
    raw = resp.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code == 429 or resp.status_code >= 500:
        raise ProviderCallFailed(
            f"provider returned {resp.status_code}", retryable=True, retry_after=_retry_after(resp)
        )
    if resp.status_code >= 400:
        raise ProviderCallFailed(f"provider returned {resp.status_code}", retryable=False)


async def post_raw(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    json: dict[str, Any] | None = None,
    files: dict[str, tuple[None, str]] | None = None,
) -> httpx.Response:
    """POST JSON or multipart form fields. Status errors become ProviderCallFailed."""
    try:
        resp = await client.post(url, headers=headers, json=json, files=files, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise ProviderCallFailed("provider timed out", retryable=True) from exc
    except httpx.TransportError as exc:
        raise ProviderCallFailed("provider connection failed", retryable=True) from exc
    _raise_for_status(resp)
    return resp


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    resp = await post_raw(client, url, headers=headers, json=json, timeout=timeout)
    try:
        body = resp.json()
    except ValueError as exc:
        raise ProviderCallFailed("provider returned invalid JSON", retryable=False) from exc
    if not isinstance(body, dict):
        raise ProviderCallFailed("provider returned unexpected JSON", retryable=False)
    return body


def parse_chat_completion(result: dict[str, Any]) -> tuple[str, list[tuple[str, float]] | None]:
    """OpenAI-style chat result -> (text, logprobs of the first token and its alternatives).

    Also accepts Cloudflare's plain `{"response": "..."}` form (no logprobs).
    """
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return str(result.get("response") or ""), None
    choice = choices[0]
    msg = choice.get("message")
    text = msg.get("content") if isinstance(msg, dict) else None
    return str(text or ""), _first_token_logprobs(choice.get("logprobs"))


def _first_token_logprobs(logprobs: object) -> list[tuple[str, float]] | None:
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return None
    first = content[0]
    seen: dict[str, float] = {}
    entries = [first, *[a for a in first.get("top_logprobs") or [] if isinstance(a, dict)]]
    for item in entries:
        token, lp = item.get("token"), item.get("logprob")
        if isinstance(token, str) and isinstance(lp, int | float) and token not in seen:
            seen[token] = float(lp)
    return list(seen.items()) or None
