"""MediGuard Agent FastAPI 应用入口。

创建 FastAPI 应用实例，注册生命周期、CORS 中间件、
业务异常处理器和 API 路由。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.dependencies import AppContainer, build_container
from .api.router import router as api_router
from .api.routes import health
from .core.config import Settings, get_settings
from .core.security import LoginRateLimiter, RequestRateLimiter
from .core.exceptions import (
    ApplicationError,
    BusinessValidationError,
    ResourceConflictError,
    ResourceNotFoundError,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建一个拥有独立内存状态的 FastAPI 应用。

    参数:
        settings: 可选配置对象；未提供时从 ``MEDIGUARD_`` 环境变量读取。

    返回:
        FastAPI: 已注册生命周期、异常处理和路由的应用实例。
    """

    resolved_settings = settings or get_settings()
    container = build_container(resolved_settings)

    def _preload_runtime_assets() -> None:
        """Preload heavyweight read-only runtime assets before serving traffic."""

        container.fraud_model.status()
        if container.case_agent is not None:
            container.case_agent.prewarm_policy_rag()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """应用生命周期管理。

        启动时加载脱敏 fixture 到仓储；
        关闭时清理 Agent 线程池和内存状态（非持久化模式下）。
        """

        container.fixtures.load_all()
        _preload_runtime_assets()
        if container.case_agent is not None and container.persistent_storage:
            container.case_agent.recover_persisted_runs()
        if (
            container.case_memory is not None
            and container.persistent_storage
            and resolved_settings.case_memory_worker_enabled
        ):
            container.case_memory.start_maintenance(
                poll_seconds=resolved_settings.case_memory_worker_poll_seconds,
                reconciliation_seconds=resolved_settings.case_memory_reconciliation_seconds,
                consolidation_seconds=resolved_settings.case_memory_consolidation_seconds,
            )
        yield
        # 关闭 Agent 线程池
        if container.evidence_agent is not None:
            container.evidence_agent.shutdown()
        if container.case_agent is not None:
            container.case_agent.shutdown()
        elif container.caser_context is not None:
            container.caser_context.shutdown()
        if container.case_memory is not None:
            container.case_memory.shutdown()
        # 非持久化模式下清空内存状态
        if not container.persistent_storage:
            container.cases.clear()
            container.traces.clear()
            container.reviews.clear()
            container.notes.clear()
            container.materials.clear()

    application = FastAPI(
        title=f"{resolved_settings.app_name} v{resolved_settings.app_version}",
        description="面向医保审核员的异常申报稽核工作台",
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    application.state.container = container
    application.state.login_rate_limiter = LoginRateLimiter(
        resolved_settings.auth_login_max_attempts,
        resolved_settings.auth_login_window_seconds,
    )
    application.state.public_demo_rate_limiter = RequestRateLimiter()

    @application.middleware("http")
    async def enforce_showcase_read_only(request: Request, call_next):
        if resolved_settings.showcase_mode and request.method in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }:
            allowed = {
                f"{resolved_settings.api_prefix}/auth/login",
                f"{resolved_settings.api_prefix}/auth/logout",
            }
            if request.url.path not in allowed:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": {
                            "code": "showcase_read_only",
                            "message": "公网展示版为只读模式，写操作已关闭。",
                        }
                    },
                )
        return await call_next(request)

    @application.middleware("http")
    async def enforce_public_demo_limits(request: Request, call_next):
        if not resolved_settings.public_demo_mode:
            return await call_next(request)

        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    request_bytes = int(content_length)
                except ValueError:
                    request_bytes = resolved_settings.public_demo_max_request_bytes + 1
                if request_bytes > resolved_settings.public_demo_max_request_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": {
                                "code": "public_demo_request_too_large",
                                "message": "请求体超过公网演示环境允许的大小。",
                            }
                        },
                    )

        category, max_requests = _public_demo_rate_limit(
            request,
            resolved_settings,
        )
        if category is not None:
            forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
            client_ip = forwarded or (request.client.host if request.client else "unknown")
            allowed = application.state.public_demo_rate_limiter.acquire(
                f"{category}:{client_ip}",
                max_requests=max_requests,
                window_seconds=resolved_settings.public_demo_rate_window_seconds,
                now=time.monotonic(),
            )
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": {
                            "code": "public_demo_rate_limited",
                            "message": "公网演示操作过于频繁，请稍后再试。",
                        }
                    },
                )
        return await call_next(request)

    # CORS 中间件：允许前端开发服务器跨域访问
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 业务异常处理器 ──────────────────────────────────

    @application.exception_handler(BusinessValidationError)
    async def handle_validation_error(
        _request: Request,
        error: BusinessValidationError,
    ) -> JSONResponse:
        """将领域校验错误转换为稳定的 422 响应。

        参数:
            _request: 触发异常的请求（未使用）。
            error: 包含错误详情列表的校验异常。

        返回:
            JSONResponse: HTTP 422，body 为 ``{"detail": [...]}``。
        """

        return JSONResponse(status_code=422, content={"detail": error.errors})

    @application.exception_handler(ResourceNotFoundError)
    async def handle_not_found(
        _request: Request,
        error: ResourceNotFoundError,
    ) -> JSONResponse:
        """将资源不存在转换为 404 响应。

        参数:
            _request: 触发异常的请求（未使用）。
            error: 包含资源标识的异常。

        返回:
            JSONResponse: HTTP 404，body 为 ``{"detail": "..."}``。
        """

        return JSONResponse(status_code=404, content={"detail": str(error)})

    @application.exception_handler(ResourceConflictError)
    async def handle_conflict(
        _request: Request,
        error: ResourceConflictError,
    ) -> JSONResponse:
        """将资源状态冲突转换为稳定的 409 响应。"""

        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": error.code,
                    "message": str(error),
                    **error.context,
                }
            },
        )

    @application.exception_handler(ApplicationError)
    async def handle_application_error(
        _request: Request,
        error: ApplicationError,
    ) -> JSONResponse:
        """兜底处理可安全返回客户端的应用异常。

        参数:
            _request: 触发异常的请求（未使用）。
            error: 通用应用异常。

        返回:
            JSONResponse: HTTP 400，body 为 ``{"detail": "..."}``。
        """

        return JSONResponse(status_code=400, content={"detail": str(error)})

    # 注册 API 路由（前缀由 Settings.api_prefix 控制，默认 /api）
    application.include_router(
        api_router,
        prefix=resolved_settings.api_prefix,
    )
    application.include_router(health.router)
    return application


def _public_demo_rate_limit(
    request: Request,
    settings: Settings,
) -> tuple[str | None, int]:
    if request.method != "POST":
        return None, 0
    path = request.url.path
    prefix = settings.api_prefix.rstrip("/")
    if path.startswith(f"{prefix}/ingest-record") or path == f"{prefix}/audit-record":
        return "ingest", settings.public_demo_ingest_max_requests
    if path.endswith("/messages") and "/case-agent/sessions/" in path:
        return "case_agent", settings.public_demo_agent_max_requests
    if path.endswith("/evidence-agent/runs"):
        return "review_advisor", settings.public_demo_agent_max_requests
    return None, 0


#: 模块级默认应用实例（uvicorn 直接引用）
app = create_app()
