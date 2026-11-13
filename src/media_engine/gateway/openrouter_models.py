"""Local cache of OpenRouter image-model capabilities, refreshed MANUALLY (never on a call).

    uv run python playground/sync_openrouter_models.py

Source: GET /api/v1/images/models (supported parameters per model) plus, for the models listed
in chains.yaml, GET /api/v1/images/models/{author}/{slug}/endpoints (pricing). The cache is used
to reject a request locally (BadRequestError) instead of paying for a call that would 400.

Documented gaps (checked against OpenRouter's docs on 2026-11-13): no per-model max
`input_references` is published; the only documented limit is 16 per request, enforced here.
`supported_parameters` is "a coarse union across endpoints", so validation is deliberately
limited to clear-cut cases: unknown model, unsupported parameter, out-of-enum or out-of-range
value, references on a model without image input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from media_engine.core.logging import get_logger

log = get_logger("openrouter_models")

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "openrouter_image_models.json"
MAX_INPUT_REFERENCES = 16  # documented global limit on POST /images


@dataclass
class ImageModelInfo:
    id: str
    supported_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    pricing: list[dict[str, Any]] = field(default_factory=list)  # per-endpoint pricing entries

    @property
    def accepts_image_input(self) -> bool:
        return "image" in self.input_modalities


class ModelCache:
    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        self._models: dict[str, ImageModelInfo] | None = None
        self.synced_at: str | None = None

    def load(self) -> bool:
        """Load the cache file. Returns False when there is none (validation is then skipped)."""
        if not self.path.exists():
            self._models = None
            return False
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.synced_at = raw.get("synced_at")
        self._models = {m["id"]: ImageModelInfo(**m) for m in raw.get("models", [])}
        return True

    @property
    def loaded(self) -> bool:
        if self._models is None:
            self.load()
        return self._models is not None

    def get(self, model_id: str) -> ImageModelInfo | None:
        if self._models is None:
            self.load()
        return (self._models or {}).get(model_id)

    def model_ids(self) -> list[str]:
        if self._models is None:
            self.load()
        return sorted(self._models or {})

    def save(self, models: list[ImageModelInfo]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        synced_at = datetime.now(UTC).isoformat()
        payload = {"synced_at": synced_at, "models": [vars(m) for m in models]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._models = {m.id: m for m in models}
        self.synced_at = synced_at


async def sync_image_models(
    client: httpx.AsyncClient,
    api_key: str,
    cache: ModelCache,
    *,
    price_for: list[str] | None = None,
) -> list[ImageModelInfo]:
    """Fetch the image model list (and pricing for `price_for`) and write the cache file."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = await client.get(f"{BASE_URL}/images/models", headers=headers, timeout=30)
    response.raise_for_status()
    models: list[ImageModelInfo] = []
    for item in response.json().get("data", []):
        arch = item.get("architecture") or {}
        models.append(
            ImageModelInfo(
                id=item["id"],
                supported_parameters=item.get("supported_parameters") or {},
                input_modalities=arch.get("input_modalities") or [],
                output_modalities=arch.get("output_modalities") or [],
            )
        )
    by_id = {m.id: m for m in models}
    for model_id in price_for or []:
        info = by_id.get(model_id)
        if info is None:
            continue
        r = await client.get(
            f"{BASE_URL}/images/models/{model_id}/endpoints", headers=headers, timeout=30
        )
        if r.status_code == 200:
            for endpoint in r.json().get("endpoints", []):
                info.pricing.extend(endpoint.get("pricing") or [])
        else:
            log.warning(
                "openrouter_models.pricing_unavailable", model=model_id, status=r.status_code
            )
    cache.save(models)
    log.info("openrouter_models.synced", count=len(models), path=str(cache.path))
    return models


def _check_descriptor(name: str, value: Any, descriptor: dict[str, Any]) -> str | None:
    kind = descriptor.get("type")
    if kind == "enum":
        allowed = descriptor.get("values") or []
        if allowed and value not in allowed and str(value) not in {str(v) for v in allowed}:
            return f"{name}={value!r} not supported (allowed: {', '.join(map(str, allowed))})"
    elif kind == "range":
        lo, hi = descriptor.get("min"), descriptor.get("max")
        if isinstance(value, int | float) and (
            (lo is not None and value < lo) or (hi is not None and value > hi)
        ):
            return f"{name}={value} out of range ({lo}..{hi})"
    return None


def validate_image_params(
    cache: ModelCache, model_id: str, params: dict[str, Any], n_references: int
) -> list[str]:
    """Return human-readable problems; empty list means OK (or no cache to check against)."""
    if not cache.loaded:
        return []
    info = cache.get(model_id)
    if info is None:
        return [
            (
                f"model {model_id!r} is not in the OpenRouter image model cache "
                f"(last synced {cache.synced_at}); run playground/sync_openrouter_models.py"
            )
        ]
    problems: list[str] = []
    for name, value in params.items():
        if value is None:
            continue
        descriptor = info.supported_parameters.get(name)
        if descriptor is None:
            problems.append(f"parameter {name!r} is not supported by {model_id}")
            continue
        problem = _check_descriptor(name, value, descriptor)
        if problem:
            problems.append(problem)
    if n_references:
        if n_references > MAX_INPUT_REFERENCES:
            problems.append(
                f"{n_references} input_references exceeds the limit of {MAX_INPUT_REFERENCES}"
            )
        if info.input_modalities and not info.accepts_image_input:
            problems.append(f"{model_id} does not accept image input (input_references)")
        descriptor = info.supported_parameters.get("input_references")
        if descriptor is not None:
            problem = _check_descriptor("input_references", n_references, descriptor)
            if problem:
                problems.append(problem)
    return problems
