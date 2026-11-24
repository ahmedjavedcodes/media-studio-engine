"""TypeSafe Jev, direct API (test mode). POST /v1/systemone."""

from typing import Any

import httpx

from app.core.errors import ProviderCallFailed
from app.gateway.providers.http import post_json

JEV_MODEL = "jev-1.13.0"


class TypeSafeJev:
    model = JEV_MODEL

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, base_url: str) -> None:
        self._client = client
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/v1/systemone"

    async def ask(
        self, *, state: dict[str, Any], questions: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        body = await post_json(
            self._client,
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"state": state, "model": self.model, "questions": questions},
            timeout=timeout,
        )
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ProviderCallFailed("jev response has no answers", retryable=False)
        return answers
