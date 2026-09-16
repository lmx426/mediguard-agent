"""健康检查路由。

提供应用可用状态、数据库连接、模型就绪等检查端点，
不要求登录。
"""

from fastapi import APIRouter, Depends

from ..dependencies import AppContainer, get_container

router = APIRouter(tags=["health"])


@router.get("/health")
def health(container: AppContainer = Depends(get_container)) -> dict[str, object]:
    """返回应用可用状态、当前案件数量、持久化后端和模型状态。

    参数:
        container: 通过 FastAPI Depends 注入的应用依赖容器。

    返回:
        dict: 包含 status / service / cases_loaded / persistence / database / fraud_model。
    """

    return {
        "status": "ok",
        "service": f"{container.settings.app_name} v{container.settings.app_version}",
        "cases_loaded": container.cases.case_count,
        "persistence": container.settings.persistence_backend,
        "showcase": {
            "enabled": container.settings.showcase_mode,
            "read_only": container.settings.showcase_mode,
            "seed_case_count": len(container.settings.showcase_record_ids)
            if container.settings.showcase_mode
            else 0,
        },
        "public_demo": {
            "enabled": container.settings.public_demo_mode,
            "interactive": container.settings.public_demo_mode,
            "max_batch_records": container.settings.public_demo_max_batch_records
            if container.settings.public_demo_mode
            else 0,
            "max_cases": container.settings.public_demo_max_cases
            if container.settings.public_demo_mode
            else 0,
        },
        "database": container.database_status(),
        "fraud_model": container.fraud_model.status().model_dump(),
        "review_advisor": container.evidence_agent_status,
        "case_agent": container.case_agent_status,
        "case_memory": (
            {
                **container.case_memory_status,
                **container.case_memory.runtime_status(),
            }
            if container.case_memory is not None
            else container.case_memory_status
        ),
    }
