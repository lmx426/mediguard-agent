"""PostgreSQL-backed user repository — users_repo."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .....core.security import hash_password
from .....domain.audit.review.entities import StoredUser
from ..models import AuditActionORM, RoleORM, UserORM, UserRoleORM, utc_now
from ..session import session_scope


class SqlUserRepository:
    """Read and maintain reviewer users from the shared users/roles tables."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def get_by_username(self, username: str) -> StoredUser | None:
        with self._session_factory() as session:
            user = session.scalar(
                select(UserORM)
                .where(UserORM.username == username.strip())
                .options(selectinload(UserORM.roles).selectinload(UserRoleORM.role))
            )
            return self._to_stored_user(user) if user else None

    def get_by_id(self, user_id: str) -> StoredUser | None:
        try:
            parsed_id = UUID(user_id)
        except ValueError:
            return None
        with self._session_factory() as session:
            user = session.scalar(
                select(UserORM)
                .where(UserORM.id == parsed_id)
                .options(selectinload(UserORM.roles).selectinload(UserRoleORM.role))
            )
            return self._to_stored_user(user) if user else None

    def record_login(self, user_id: str) -> None:
        try:
            parsed_id = UUID(user_id)
        except ValueError:
            return
        with session_scope(self._session_factory) as session:
            user = session.get(UserORM, parsed_id)
            if user is not None:
                user.last_login_at = utc_now()

    def get_case_memory_enabled(self, user_id: str) -> bool:
        try:
            parsed_id = UUID(user_id)
        except ValueError:
            return False
        with self._session_factory() as session:
            value = session.scalar(
                select(UserORM.case_memory_enabled).where(UserORM.id == parsed_id)
            )
            return bool(value) if value is not None else False

    def set_case_memory_enabled(self, user_id: str, enabled: bool) -> bool:
        try:
            parsed_id = UUID(user_id)
        except ValueError as exc:
            raise KeyError(user_id) from exc
        with session_scope(self._session_factory) as session:
            user = session.get(UserORM, parsed_id)
            if user is None:
                raise KeyError(user_id)
            user.case_memory_enabled = enabled
            session.add(
                AuditActionORM(
                    actor_id=parsed_id,
                    action_type="case_memory_preference_changed",
                    summary="开启个人跨案件记忆" if enabled else "关闭个人跨案件记忆",
                    payload={"enabled": enabled},
                )
            )
        return enabled

    def log_auth_event(
        self,
        username: str,
        success: bool,
        reason: str,
        user_id: str | None = None,
    ) -> None:
        actor_id = None
        if user_id:
            try:
                actor_id = UUID(user_id)
            except ValueError:
                actor_id = None
        with session_scope(self._session_factory) as session:
            session.add(
                AuditActionORM(
                    actor_id=actor_id,
                    action_type="login_success" if success else "login_failed",
                    summary="审核人员登录成功" if success else "审核人员登录失败",
                    payload={
                        "username": username,
                        "reason": reason,
                    },
                )
            )

    def upsert_user(
        self,
        *,
        username: str,
        display_name: str,
        department: str | None,
        roles: Iterable[str],
        password: str | None = None,
        status: str = "active",
    ) -> StoredUser:
        with session_scope(self._session_factory) as session:
            user = session.scalar(select(UserORM).where(UserORM.username == username))
            if user is None:
                if not password:
                    raise ValueError("password is required when creating a user")
                user = UserORM(
                    username=username,
                    display_name=display_name,
                    department=department,
                    status=status,
                    password_hash=hash_password(password or ""),
                )
                session.add(user)
                session.flush()
            else:
                user.display_name = display_name
                user.department = department
                user.status = status
                if password:
                    user.password_hash = hash_password(password)

            requested_roles = {role.strip() for role in roles if role.strip()}
            for role_code in requested_roles:
                role = session.scalar(select(RoleORM).where(RoleORM.code == role_code))
                if role is None:
                    role = RoleORM(code=role_code, name=role_code)
                    session.add(role)
                    session.flush()
                if not any(item.role_id == role.id for item in user.roles):
                    session.add(UserRoleORM(user_id=user.id, role_id=role.id))
            session.flush()
            loaded = session.scalar(
                select(UserORM)
                .where(UserORM.id == user.id)
                .options(selectinload(UserORM.roles).selectinload(UserRoleORM.role))
            )
            return self._to_stored_user(loaded or user)

    @staticmethod
    def _to_stored_user(user: UserORM) -> StoredUser:
        return StoredUser(
            id=str(user.id),
            username=user.username,
            display_name=user.display_name,
            department=user.department,
            status=user.status,
            roles=[entry.role.code for entry in user.roles if entry.role is not None],
            password_hash=user.password_hash,
            case_memory_enabled=user.case_memory_enabled,
        )
