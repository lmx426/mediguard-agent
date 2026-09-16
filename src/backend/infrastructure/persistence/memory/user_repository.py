"""In-memory user repository for local tests and fallback demos."""

from __future__ import annotations

from uuid import uuid4

from ....core.security import hash_password
from ....domain.audit.review.entities import StoredUser


class MemoryUserRepository:
    """Store a single default active auditor in process memory."""

    def __init__(self, default_password: str) -> None:
        user_id = str(uuid4())
        self._users: dict[str, StoredUser] = {
            user_id: StoredUser(
                id=user_id,
                username="default_auditor",
                display_name="演示审核员",
                department="医保稽核",
                status="active",
                roles=["auditor"],
                password_hash=hash_password(default_password),
            )
        }
        self._by_username = {"default_auditor": user_id}
        self.auth_events: list[dict[str, object]] = []

    def get_by_username(self, username: str) -> StoredUser | None:
        user_id = self._by_username.get(username.strip())
        return self._users.get(user_id) if user_id else None

    def get_by_id(self, user_id: str) -> StoredUser | None:
        return self._users.get(user_id)

    def record_login(self, user_id: str) -> None:
        if user_id in self._users:
            self.auth_events.append({"user_id": user_id, "type": "login_success"})

    def get_case_memory_enabled(self, user_id: str) -> bool:
        user = self._users.get(user_id)
        return user.case_memory_enabled if user is not None else False

    def set_case_memory_enabled(self, user_id: str, enabled: bool) -> bool:
        user = self._users.get(user_id)
        if user is None:
            raise KeyError(user_id)
        user.case_memory_enabled = enabled
        self.auth_events.append(
            {
                "user_id": user_id,
                "type": "case_memory_preference_changed",
                "enabled": enabled,
            }
        )
        return user.case_memory_enabled

    def log_auth_event(
        self,
        username: str,
        success: bool,
        reason: str,
        user_id: str | None = None,
    ) -> None:
        self.auth_events.append(
            {
                "username": username,
                "success": success,
                "reason": reason,
                "user_id": user_id,
            }
        )
