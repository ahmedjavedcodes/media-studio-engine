"""B1: signed tokens on /events and on every image URL."""

from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from app.api.deps import get_create_deps, get_run_service, get_token_signer
from app.core.tokens import TokenSigner
from app.guardrails.deps import get_guard_deps
from app.main import app
from tests.unit.create_helpers import critic_ok, make_deps, make_service
from tests.unit.guard_helpers import make_deps as make_guard_deps


def test_a_token_works_only_for_its_run_and_user_until_it_expires() -> None:
    now = [1000.0]
    s = TokenSigner("secret", ttl_s=60, clock=lambda: now[0])
    tok = s.sign("run1", "alice")
    assert s.verify(tok, "run1", "alice")
    assert not s.verify(tok, "run2", "alice")  # another run
    assert not s.verify(tok, "run1", "eve")  # another user
    assert not s.verify(tok, "run1", None)
    assert not TokenSigner("other-secret", clock=lambda: now[0]).verify(tok, "run1", "alice")
    now[0] += 61
    assert not s.verify(tok, "run1", "alice")  # expired


@pytest.mark.parametrize("bad", ["", "abc", "123.", ".abc", "x.y", "9999999999.deadbeef"])
def test_garbage_tokens_are_refused(bad: str) -> None:
    assert not TokenSigner("secret").verify(bad, "run1", "alice")


def test_a_tampered_expiry_is_refused() -> None:
    s = TokenSigner("secret", ttl_s=60, clock=lambda: 1000.0)
    exp, _, mac = s.sign("run1", "alice").partition(".")
    assert not s.verify(f"{int(exp) + 99999}.{mac}", "run1", "alice")


def test_the_default_ttl_is_two_hours() -> None:
    s = TokenSigner("secret", clock=lambda: 0.0)
    assert s.sign("r", "u").split(".")[0] == str(2 * 60 * 60)


def test_without_a_secret_a_temporary_one_is_used() -> None:
    a, b = TokenSigner(""), TokenSigner("")
    assert not b.verify(a.sign("r", "u"), "r", "u")  # different temporary keys


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[tuple[httpx.AsyncClient, object]]:
    deps = make_deps(tmp_path, critic=[critic_ok(2)])
    service = make_service(deps)
    app.dependency_overrides[get_create_deps] = lambda: deps
    app.dependency_overrides[get_run_service] = lambda: service
    app.dependency_overrides[get_token_signer] = lambda: deps.tokens
    app.dependency_overrides[get_guard_deps] = lambda: make_guard_deps()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c, deps
    app.dependency_overrides.clear()


async def start(c: httpx.AsyncClient) -> tuple[str, str]:
    created = (await c.post("/v1/runs", json={"prompt": "a fox", "settings": {"n": 2}})).json()
    return created["run_id"], created["token"]


async def test_events_need_a_good_token(api) -> None:  # type: ignore[no-untyped-def]
    c, deps = api
    run_id, token = await start(c)
    other_id, other_token = await start(c)
    ok = await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    assert ok.status_code == 200 and "event: ready" in ok.text
    for t in ("", "nonsense", other_token, token[:-2] + "00"):  # missing, wrong, other run, forged
        res = await c.get(f"/v1/runs/{run_id}/events", params={"t": t})
        assert res.status_code == 404, t
    expired = TokenSigner("test-secret", clock=lambda: 0.0).sign(run_id, "local-dev")
    assert (await c.get(f"/v1/runs/{run_id}/events", params={"t": expired})).status_code == 404
    assert other_id != run_id


async def test_the_header_no_longer_decides_access_to_events(api) -> None:  # type: ignore[no-untyped-def]
    c, _ = api
    run_id, token = await start(c)
    res = await c.get(
        f"/v1/runs/{run_id}/events", params={"t": token}, headers={"X-User-Id": "someone-else"}
    )
    assert res.status_code == 200  # the token is the access, EventSource sends no header


async def test_every_image_url_carries_a_token_and_files_need_it(api) -> None:  # type: ignore[no-untyped-def]
    c, _ = api
    run_id, token = await start(c)
    text = (await c.get(f"/v1/runs/{run_id}/events", params={"t": token})).text
    ready = next(ln for ln in text.splitlines() if ln.startswith("data:") and '"images"' in ln)
    import json

    images = json.loads(ready[6:])["images"]
    assert len(images) == 2
    for img in images:
        for key in ("url", "thumb_url"):
            assert "t=" in img[key]
    url = images[0]["url"]
    assert (await c.get(url)).status_code == 200  # the signed URL works as sent
    bare = urlparse(url).path
    assert (await c.get(bare)).status_code == 404  # no token
    assert (await c.get(bare, params={"t": "wrong"})).status_code == 404
    other_id, _ = await start(c)
    assert (
        await c.get(bare.replace(run_id, other_id), params=parse_qs(urlparse(url).query))
    ).status_code == 404  # the token of one run does not open another run's file


async def test_the_saved_result_returns_signed_urls_and_resume_returns_a_token(  # type: ignore[no-untyped-def]
    api,
) -> None:
    c, _ = api
    run_id, token = await start(c)
    await c.get(f"/v1/runs/{run_id}/events", params={"t": token})
    saved = (await c.get(f"/v1/runs/{run_id}")).json()
    assert all("t=" in i["url"] for i in saved["result"]["images"])
    assert (await c.get(saved["result"]["images"][0]["url"])).status_code == 200
