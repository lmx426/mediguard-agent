"""认证路由。

提供审核人员账号密码登录、会话查询和登出端点。
登录成功后签发 HttpOnly Cookie，后续业务端点通过该 Cookie 鉴权。
"""

from __future__ import annotations

from datetime import timedelta
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from ..dependencies import get_container, get_current_user
from ...core.security import create_auth_token, verify_password
from ...domain.audit.review.entities import AuthenticatedUser, StoredUser
from ..schemas.auth import AuthResponse, LoginInput

router = APIRouter(prefix="/auth", tags=["auth"])
#: 允许登录的角色集合
ALLOWED_LOGIN_ROLES = {"auditor", "reviewer", "admin"}


@router.post("/login", response_model=AuthResponse)
def login(
    body: LoginInput,
    request: Request,
    response: Response,
    container=Depends(get_container),
) -> AuthResponse:
    """验证账号密码并签发 HttpOnly 会话 Cookie。

    参数:
        body: 登录请求体，包含 username 和 password。
        response: FastAPI Response 对象，用于设置 Cookie。
        container: 应用依赖容器。

    返回:
        AuthResponse: 包含当前登录审核人员信息。

    异常:
        HTTPException 401: 用户名或密码不正确，或账号未开通审核权限。
    """

    username = body.username.strip()
    forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    client_ip = forwarded or (request.client.host if request.client else "unknown")
    limiter_key = f"{client_ip}:{username.casefold()}"
    limiter = request.app.state.login_rate_limiter
    now = time.monotonic()
    if not limiter.is_allowed(limiter_key, now):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="登录尝试过于频繁，请稍后再试。",
        )
    user = container.users.get_by_username(username)
    if not _can_login(user) or not verify_password(body.password, user.password_hash):
        limiter.record_failure(limiter_key, now)
        container.users.log_auth_event(username, False, "invalid_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录失败 / 请确认审核人员账号、密码是否正确，且账号已开通审核权限。",
        )

    authenticated = user.to_authenticated()
    limiter.clear(limiter_key)
    token = create_auth_token(
        {
            "sub": authenticated.id,
            "username": authenticated.username,
        },
        secret_key=container.settings.auth_secret_key,
        expires_delta=timedelta(hours=container.settings.auth_session_hours),
    )
    # 签发 HttpOnly Cookie，前端不可通过 JS 读取
    response.set_cookie(
        key=container.settings.auth_cookie_name,
        value=token,
        max_age=container.settings.auth_session_hours * 3600,
        httponly=True,
        secure=container.settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )
    container.users.record_login(authenticated.id)
    container.users.log_auth_event(username, True, "ok", authenticated.id)
    return AuthResponse(user=authenticated)


@router.get("/me", response_model=AuthenticatedUser)
def me(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    """返回当前登录审核人员信息。

    参数:
        current_user: 通过 Cookie 解析注入的当前用户。

    返回:
        AuthenticatedUser: 当前审核人员。
    """

    return current_user


@router.post("/logout")
def logout(response: Response, container=Depends(get_container)) -> dict[str, str]:
    """清除会话 Cookie 并登出。

    参数:
        response: FastAPI Response 对象，用于删除 Cookie。
        container: 应用依赖容器。

    返回:
        dict: ``{"status": "ok"}``。
    """

    response.delete_cookie(
        key=container.settings.auth_cookie_name,
        path="/",
        secure=container.settings.auth_cookie_secure,
        samesite="lax",
    )
    return {"status": "ok"}


def _can_login(user: StoredUser | None) -> bool:
    """校验用户是否满足登录条件。

    参数:
        user: 从仓储中查到的存储用户对象。

    返回:
        bool: 用户存在、状态为 active、有密码哈希且角色在允许列表内。
    """

    return bool(
        user
        and user.status == "active"
        and user.password_hash
        and ALLOWED_LOGIN_ROLES.intersection(user.roles)
    )
