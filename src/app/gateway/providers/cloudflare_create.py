"""Cloudflare Workers AI for image create (build mode): the LLM, the critic and image models.

UNVERIFIED until a real call (checked in playground/create.py): the FLUX.2 multipart fields,
whether the REST envelope wraps the image as `result.image`, and the gpt-oss output shape.
The parsers below accept every shape the docs hint at.
"""

import base64
import re
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from app.contracts.ports import GeneratedImage, ImageOptions
from app.core.errors import InvalidModelOutput, ProviderCallFailed
from app.gateway.providers.cloudflare import CloudflareClient
from app.gateway.providers.http import post_raw

log = structlog.get_logger("provider")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_text(result: dict[str, Any]) -> str | dict[str, Any]:
    """The model's answer from a Cloudflare result. Accepts chat, plain and Responses shapes."""
    choices = result.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"])
    response = result.get("response")
    if isinstance(response, dict):
        return response  # JSON mode can return the parsed object
    if isinstance(response, str) and response:
        return response
    output = result.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("text"):
                        return str(part["text"])
    raise InvalidModelOutput("the model returned no answer")


def parse_schema[T: BaseModel](answer: str | dict[str, Any], schema: type[T]) -> T:
    try:
        if isinstance(answer, dict):
            return schema.model_validate(answer)
        return schema.model_validate_json(_FENCE.sub("", answer.strip()))
    except (ValidationError, ValueError) as exc:
        raise InvalidModelOutput("the model answered in the wrong format") from exc


def add_usage(sink: dict[str, Any] | None, result: dict[str, Any]) -> None:
    """Add a call's tokens and neurons to `sink`. Adds up, so a retried call counts both."""
    raw = result.get("usage")
    if sink is None or not isinstance(raw, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "neurons"):
        value = raw.get(key)
        if isinstance(value, int | float):
            sink[key] = sink.get(key, 0) + value


class CloudflareLLM:
    """A text or vision model that answers in a Pydantic schema (LLMPort)."""

    provider = "cloudflare"

    def __init__(self, cf: CloudflareClient, model: str) -> None:
        self._cf = cf
        self.model = model

    async def complete_json[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        images: list[str] | None = None,
        max_tokens: int,
        temperature: float,
        timeout: float,
        reasoning_effort: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> T:
        content: str | list[dict[str, Any]] = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": u}} for u in images
            ]
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
        }
        if reasoning_effort and "gpt-oss" in self.model:  # reasoning models only
            body["reasoning_effort"] = reasoning_effort
        result = await self._cf.run(self.model, body, timeout)
        add_usage(usage, result)
        return parse_schema(extract_text(result), schema)


class CloudflareImages:
    """Image models on Workers AI (ImageProviderPort)."""

    name = "cloudflare"

    def __init__(self, cf: CloudflareClient, client: httpx.AsyncClient) -> None:
        self._cf = cf
        self._client = client

    async def generate(
        self,
        *,
        slug: str,
        family: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        aspect_ratio: str,
        seed: int | None,
        timeout: float,
        estimated_cost_usd: float = 0.0,
        options: ImageOptions | None = None,
    ) -> GeneratedImage:
        fields: dict[str, Any] = {"prompt": prompt}
        multipart = False
        if "flux-1-schnell" in slug:
            fields["steps"] = 4  # schnell takes no size. The registry limits it to 1:1
        elif "flux-2" in slug:
            multipart = True  # FLUX.2 takes multipart form fields
            fields.update(width=width, height=height)
        else:
            fields.update(width=width, height=height)
            if negative_prompt:
                fields["negative_prompt"] = negative_prompt
        if options is not None:  # only models whose card lists them ever get a value
            if options.guidance is not None:
                fields["guidance"] = options.guidance
            if options.steps is not None and "flux-1-schnell" not in slug:
                fields["steps" if "flux" in slug else "num_steps"] = options.steps
        if seed is not None:
            fields["seed"] = seed
        data = await self._call(slug, fields, multipart=multipart, timeout=timeout)
        return GeneratedImage(data=data, cost_usd=None)

    async def _call(
        self, slug: str, fields: dict[str, Any], *, multipart: bool, timeout: float
    ) -> bytes:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._cf.account_id}/ai/run/{slug}"
        headers = {"Authorization": f"Bearer {self._cf.token}"}
        if multipart:
            files = {k: (None, str(v)) for k, v in fields.items()}
            resp = await post_raw(self._client, url, headers=headers, files=files, timeout=timeout)
        else:
            resp = await post_raw(self._client, url, headers=headers, json=fields, timeout=timeout)
        if resp.headers.get("content-type", "").startswith("image/"):
            return resp.content  # some models return the file itself
        try:
            body = resp.json()
        except ValueError as exc:
            raise ProviderCallFailed("cloudflare returned invalid JSON", retryable=False) from exc
        if not isinstance(body, dict) or body.get("success") is False:
            raise ProviderCallFailed("cloudflare reported failure", retryable=False)
        result = body.get("result", body)
        image = result.get("image") if isinstance(result, dict) else None
        if not isinstance(image, str):
            raise ProviderCallFailed("cloudflare returned no image", retryable=False)
        try:
            return base64.b64decode(image, validate=False)
        except ValueError as exc:
            raise ProviderCallFailed("cloudflare returned a bad image", retryable=False) from exc
