"""Password hashing and signed cookie helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any


PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260_000


class LoginRateLimiter:
    """Small in-process limiter for the single-instance showcase login."""

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, key: str, now: float) -> bool:
        with self._lock:
            attempts = self._attempts[key]
            self._prune(attempts, now)
            return len(attempts) < self._max_attempts

    def record_failure(self, key: str, now: float) -> None:
        with self._lock:
            attempts = self._attempts[key]
            self._prune(attempts, now)
            attempts.append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)

    def _prune(self, attempts: deque[float], now: float) -> None:
        cutoff = now - self._window_seconds
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()


class RequestRateLimiter:
    """In-process sliding-window limiter for a single public demo instance."""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def acquire(
        self,
        key: str,
        *,
        max_requests: int,
        window_seconds: int,
        now: float,
    ) -> bool:
        with self._lock:
            requests = self._requests[key]
            cutoff = now - window_seconds
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= max_requests:
                return False
            requests.append(now)
            return True


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""

    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            PASSWORD_ALGORITHM,
            str(PASSWORD_ITERATIONS),
            _b64(salt),
            _b64(digest),
        ]
    )


def verify_password(password: str, encoded: str | None) -> bool:
    """Verify a plaintext password against the stored PBKDF2 hash."""

    if not password or not encoded:
        return False
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != PASSWORD_ALGORITHM:
            return False
        iterations = int(iterations_text)
        salt = _unb64(salt_text)
        expected = _unb64(digest_text)
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def create_auth_token(
    payload: dict[str, Any],
    secret_key: str,
    expires_delta: timedelta,
) -> str:
    """Create a compact HMAC-signed token for the HttpOnly session cookie."""

    now = datetime.now(timezone.utc)
    claims = {
        **payload,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    body = _b64url(
        json.dumps(claims, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )
    signature = _token_signature(body, secret_key)
    return f"{body}.{signature}"


def read_auth_token(token: str | None, secret_key: str) -> dict[str, Any] | None:
    """Return verified token claims, or None when the token is invalid/expired."""

    if not token or "." not in token:
        return None
    body, signature = token.rsplit(".", 1)
    expected = _token_signature(body, secret_key)
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        claims = json.loads(_unb64url(body).decode("utf-8"))
        expires_at = int(claims.get("exp", 0))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if expires_at <= int(datetime.now(timezone.utc).timestamp()):
        return None
    return claims


def _token_signature(body: str, secret_key: str) -> str:
    digest = hmac.new(secret_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
    return _b64url(digest.digest())


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))
