"""API 路由聚合。

将 health / intake / audit / agent / auth 五组路由挂载到同一个 APIRouter，
统一在 main.py 中以 ``api_prefix`` 前缀注册。
"""

from fastapi import APIRouter, Depends

from .dependencies import get_current_user
from .routes import (
    auth,
    audit,
    agent,
    case_agent,
    health,
    intake,
    memory,
)

router = APIRouter()
# 公开端点（无需登录）
router.include_router(health.router)
router.include_router(auth.router)
# Agent 状态端点公开（不暴露配置和内部错误）
router.include_router(agent.router)
router.include_router(case_agent.router)
router.include_router(memory.router)
# 业务端点（需要审核人员登录）
router.include_router(intake.router, dependencies=[Depends(get_current_user)])
router.include_router(audit.router, dependencies=[Depends(get_current_user)])
