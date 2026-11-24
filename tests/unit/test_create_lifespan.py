"""The real app startup: providers, prompts and the SQLite checkpointer are wired together."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


def test_app_starts_with_real_wiring_and_serves_the_catalog(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GUARD_WARMUP", "false")  # no network at startup
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))  # type: ignore[operator]
    monkeypatch.setenv("LOCAL_OUTPUT_DIR", str(tmp_path / "out"))  # type: ignore[operator]
    monkeypatch.setenv("OPENROUTER_IMAGES", "false")
    get_settings.cache_clear()
    from app.main import settings as main_settings

    main_settings.guard_warmup = False
    main_settings.checkpoint_db = str(tmp_path / "cp.sqlite")  # type: ignore[operator]
    from app.main import app

    try:
        with TestClient(app) as client:
            models = client.get("/v1/models", params={"plan": "basic"}).json()["models"]
            assert {m["id"] for m in models} >= {"flux-1-schnell", "flux-2-klein-4b"}
            assert all(m["id"] != "nano-banana-2" for m in models)  # OpenRouter stays off
            assert hasattr(app.state, "run_service")
            assert app.state.create_deps.enabled_providers == ["cloudflare"]
    finally:
        get_settings.cache_clear()
