"""SQLAlchemy engine/session helpers for PostgreSQL persistence."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ....core.config import Settings
from ....core.security import hash_password
from .models import RoleORM, UserORM, UserRoleORM


DEFAULT_AUDITOR_USERNAME = "default_auditor"
DEFAULT_AUDITOR_DISPLAY_NAME = "演示审核员"
DEFAULT_ROLE_NAMES = {
    "auditor": "医保审核员",
    "reviewer": "医保复审员",
    "admin": "系统管理员",
}


def create_database_engine(settings: Settings) -> Engine:
    """Create a PostgreSQL engine from application settings."""

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a short-lived SQLAlchemy session factory."""

    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: Callable[[], Session]) -> Iterator[Session]:
    """Provide a transactional scope for repository operations."""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_status(engine: Engine, backend: str = "postgres") -> dict[str, Any]:
    """Return a compact health-check payload without leaking credentials."""

    try:
        with engine.connect() as connection:
            connection.execute(text("select 1"))
        connected = True
    except Exception as exc:  # pragma: no cover - depends on runtime DB state
        return {
            "enabled": backend == "postgres",
            "backend": backend,
            "connected": False,
            "error": exc.__class__.__name__,
        }

    return {
        "enabled": backend == "postgres",
        "backend": backend,
        "connected": connected,
    }


def get_or_create_default_auditor(session: Session) -> UserORM:
    """Return the seeded default auditor, creating it for dev/test DBs if absent."""

    user = session.scalar(
        select(UserORM).where(UserORM.username == DEFAULT_AUDITOR_USERNAME)
    )
    if user is not None:
        if user.display_name != DEFAULT_AUDITOR_DISPLAY_NAME:
            user.display_name = DEFAULT_AUDITOR_DISPLAY_NAME
        return user

    auditor_role = session.scalar(select(RoleORM).where(RoleORM.code == "auditor"))
    if auditor_role is None:
        auditor_role = RoleORM(code="auditor", name="医保审核员")
        session.add(auditor_role)
        session.flush()

    user = UserORM(
        username=DEFAULT_AUDITOR_USERNAME,
        display_name=DEFAULT_AUDITOR_DISPLAY_NAME,
        department="医保稽核",
        status="active",
    )
    session.add(user)
    session.flush()
    session.add(UserRoleORM(user_id=user.id, role_id=auditor_role.id))
    return user


def seed_default_auditor(
    session_factory: Callable[[], Session],
    settings: Settings,
) -> None:
    """Ensure baseline roles and default auditor password hash exist."""

    with session_scope(session_factory) as session:
        roles: dict[str, RoleORM] = {}
        for code, name in DEFAULT_ROLE_NAMES.items():
            role = session.scalar(select(RoleORM).where(RoleORM.code == code))
            if role is None:
                role = RoleORM(code=code, name=name)
                session.add(role)
                session.flush()
            roles[code] = role

        user = session.scalar(
            select(UserORM).where(UserORM.username == DEFAULT_AUDITOR_USERNAME)
        )
        if user is None:
            user = UserORM(
                username=DEFAULT_AUDITOR_USERNAME,
                display_name=DEFAULT_AUDITOR_DISPLAY_NAME,
                department="医保稽核",
                status="active",
                password_hash=hash_password(settings.default_auditor_password),
            )
            session.add(user)
            session.flush()
        else:
            if user.display_name != DEFAULT_AUDITOR_DISPLAY_NAME:
                user.display_name = DEFAULT_AUDITOR_DISPLAY_NAME
            if not user.password_hash:
                user.password_hash = hash_password(settings.default_auditor_password)

        auditor_role = roles["auditor"]
        if not any(entry.role_id == auditor_role.id for entry in user.roles):
            session.add(UserRoleORM(user_id=user.id, role_id=auditor_role.id))
