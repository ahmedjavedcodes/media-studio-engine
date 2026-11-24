"""Signed tokens for the event stream and the image files (review finding R1).

`EventSource` and `<img>` cannot send an `X-User-Id` header, so `POST /v1/runs` returns a token
that goes in the URL as `?t=`. The token is `<exp>.<hmac>`, where the HMAC-SHA256 covers
`run_id|user_id|exp`. It is tied to one run and one user and expires after 2 hours. The check
uses a constant-time compare.
"""

import hashlib
import hmac
import secrets
import time
from collections.abc import Callable

import structlog

log = structlog.get_logger("tokens")

DEFAULT_TTL_S = 2 * 60 * 60


class TokenSigner:
    def __init__(
        self,
        secret: str = "",
        *,
        ttl_s: int = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not secret:
            # Dev only: tokens stop working when the process restarts. Set STREAM_TOKEN_SECRET.
            secret = secrets.token_hex(32)
            log.warning("stream_token_secret_not_set_using_a_temporary_one")
        self._key = secret.encode()
        self._ttl = ttl_s
        self._clock = clock

    def _mac(self, run_id: str, user_id: str, exp: int) -> str:
        msg = f"{run_id}|{user_id}|{exp}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def sign(self, run_id: str, user_id: str) -> str:
        exp = int(self._clock()) + self._ttl
        return f"{exp}.{self._mac(run_id, user_id, exp)}"

    def verify(self, token: str, run_id: str, user_id: str | None) -> bool:
        """True only for an unexpired token made for exactly this run and user."""
        if not token or user_id is None:
            return False
        head, _, mac = token.partition(".")
        if not head.isdigit() or not mac:
            return False
        exp = int(head)
        if exp <= self._clock():
            return False
        return hmac.compare_digest(mac, self._mac(run_id, user_id, exp))

    def signed_path(self, path: str, run_id: str, user_id: str) -> str:
        """`path` with `?t=<token>` added. Used for every image URL we send."""
        return f"{path}?t={self.sign(run_id, user_id)}"
