"""上游接入路由。

提供脱敏样本查询、单条记录推送、CSV 批量建案等端点。
所有端点要求审核人员登录。
"""

from fastapi import APIRouter, Depends
from fastapi import HTTPException, status

from ..agent_autostart import prestart_review_advisor
from ..dependencies import (
    AppContainer,
    get_feature_validation,
    get_container,
    get_current_user,
    get_ingest_cases,
    get_ingest_records,
    get_ingest_validation,
    get_visitor_ingest_records,
)
from ...application.intake.build_case_pipeline import AuditPipelineService
from ...core.exceptions import ResourceNotFoundError
from ...domain.intake.validation_rules import (
    FeatureValidationService,
    IngestValidationService,
)
from ...domain.audit.review.entities import AuthenticatedUser
from ...infrastructure.fixtures_loader import (
    IngestRecordCatalog,
    VisitorIngestRecordCatalog,
)
from ..schemas.ingest import (
    BatchIngestInput,
    BatchIngestResponse,
    FeatureRecordInput,
    FeatureSchemaResponse,
    IngestRecordInput,
    IngestRecordResponse,
    IngestRecordSummary,
    IngestSchemaResponse,
)
from ..schemas.cases import CaseFullResponse

router = APIRouter(tags=["intake"])


def _resolve_ingest_record(
    record_id: str,
    records: IngestRecordCatalog,
    visitor_records: VisitorIngestRecordCatalog,
    *,
    prefer_visitor: bool = False,
):
    """Resolve a record while preserving visitor metadata in public demo mode."""

    if prefer_visitor:
        record = visitor_records.get_record(record_id)
        if record is not None:
            return record

    record = records.get_record(record_id)
    if record is not None:
        return record
    return visitor_records.get_record(record_id)


@router.get("/feature-schema", response_model=FeatureSchemaResponse)
def get_feature_schema(
    validation: FeatureValidationService = Depends(get_feature_validation),
) -> FeatureSchemaResponse:
    """返回兼容统计特征输入契约（向后兼容旧版接入）。

    参数:
        validation: 特征校验服务。

    返回:
        FeatureSchemaResponse: 字段名、类型和约束列表。
    """

    return validation.get_schema()


@router.get("/ingest-schema", response_model=IngestSchemaResponse)
def get_ingest_schema(
    validation: IngestValidationService = Depends(get_ingest_validation),
) -> IngestSchemaResponse:
    """返回完整 81 字段接入契约。

    参数:
        validation: 接入校验服务。

    返回:
        IngestSchemaResponse: 81 字段的字段名、类型和安全约束。
    """

    return validation.get_schema()


@router.get("/ingest-records", response_model=list[IngestRecordSummary])
def list_ingest_records(
    records: IngestRecordCatalog = Depends(get_ingest_records),
) -> list[IngestRecordSummary]:
    """返回内置脱敏宽表记录摘要列表。

    参数:
        records: 脱敏样本记录目录。

    返回:
        list[IngestRecordSummary]: 每条记录的合成申报人编号、风险等级、风险信号分。
    """

    return records.list_records()


@router.get("/visitor-ingest-records", response_model=list[IngestRecordSummary])
def list_visitor_ingest_records(
    visitor_records: VisitorIngestRecordCatalog = Depends(get_visitor_ingest_records),
) -> list[IngestRecordSummary]:
    """返回游客访问使用的固定样本清单。"""

    return visitor_records.list_records()


@router.get(
    "/ingest-records/{record_id}",
    response_model=IngestRecordResponse,
)
def get_ingest_record(
    record_id: str,
    records: IngestRecordCatalog = Depends(get_ingest_records),
    visitor_records: VisitorIngestRecordCatalog = Depends(get_visitor_ingest_records),
    container: AppContainer = Depends(get_container),
) -> IngestRecordResponse:
    """返回一条完整脱敏 81 字段宽表记录。

    参数:
        record_id: 合成申报人编号，如 ``SIM_PERSON_000006``。
        records: 脱敏样本记录目录。

    返回:
        IngestRecordResponse: 完整 81 字段键值对。

    异常:
        ResourceNotFoundError: 指定记录不存在。
    """

    record = _resolve_ingest_record(
        record_id,
        records,
        visitor_records,
        prefer_visitor=container.settings.public_demo_mode,
    )
    if record is None:
        raise ResourceNotFoundError(f"上游记录 {record_id} 不存在")
    return record


@router.post(
    "/ingest-records/{record_id}/push",
    response_model=CaseFullResponse,
)
def push_ingest_record(
    record_id: str,
    records: IngestRecordCatalog = Depends(get_ingest_records),
    visitor_records: VisitorIngestRecordCatalog = Depends(get_visitor_ingest_records),
    use_case: AuditPipelineService = Depends(get_ingest_cases),
    validation: IngestValidationService = Depends(get_ingest_validation),
    container: AppContainer = Depends(get_container),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CaseFullResponse:
    """将内置脱敏宽表记录推送为稽核案件。

    参数:
        record_id: 合成申报人编号。
        records: 脱敏样本记录目录。
        use_case: 建案流水线用例。
        validation: 接入校验服务。

    返回:
        CaseFullResponse: 生成的完整案件详情（含证据包、Trace、九阶段流程）。
    """

    _enforce_public_demo_ingest_capacity(container, requested_records=1)
    record = _resolve_ingest_record(
        record_id,
        records,
        visitor_records,
        prefer_visitor=container.settings.public_demo_mode,
    )
    if record is None:
        raise ResourceNotFoundError(f"上游记录 {record_id} 不存在")
    metadata = record.metadata or {}
    response = use_case.run_ingest(
        IngestRecordInput(
            case_title=str(metadata.get("case_title") or f"上游申报记录 {record_id}"),
            case_type=str(metadata.get("case_type") or "上游宽表记录接入"),
            source_system=str(metadata.get("source_system") or "医保结算申报系统"),
            record_version=str(metadata.get("record_version") or "claim-wide-v1"),
            case_context=record.case_context or {},
            record=record.record,
        ),
        validation,
    )
    prestart_review_advisor(container, response.case.case_id, current_user)
    return response


@router.post("/ingest-record", response_model=CaseFullResponse)
def ingest_record(
    body: IngestRecordInput,
    use_case: AuditPipelineService = Depends(get_ingest_cases),
    validation: IngestValidationService = Depends(get_ingest_validation),
    container: AppContainer = Depends(get_container),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CaseFullResponse:
    """校验并幂等接入一条完整 81 字段记录。

    参数:
        body: 81 字段脱敏宽表记录 + 元信息。
        use_case: 建案流水线用例。
        validation: 接入校验服务。

    返回:
        CaseFullResponse: 生成的完整案件详情。相同指纹的重复提交返回已有案件。
    """

    _enforce_public_demo_ingest_capacity(container, requested_records=1)
    response = use_case.run_ingest(body, validation)
    prestart_review_advisor(container, response.case.case_id, current_user)
    return response


@router.post(
    "/ingest-records/batch",
    response_model=BatchIngestResponse,
)
def ingest_record_batch(
    body: BatchIngestInput,
    use_case: AuditPipelineService = Depends(get_ingest_cases),
    validation: IngestValidationService = Depends(get_ingest_validation),
    container: AppContainer = Depends(get_container),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> BatchIngestResponse:
    """原子处理多条完整 81 字段记录（CSV 批量建案）。

    全部校验通过后才统一写入；
    任一记录校验失败则整批拒绝，不生成部分案件。

    参数:
        body: 包含多条记录的批量接入请求。
        use_case: 建案流水线用例。
        validation: 接入校验服务。

    返回:
        BatchIngestResponse: 生成的全部案件详情列表。
    """

    _enforce_public_demo_ingest_capacity(
        container,
        requested_records=len(body.records),
    )
    cases = use_case.run_ingest_batch(body.records, validation)
    prestart_review_advisor(
        container,
        [item.case.case_id for item in cases],
        current_user,
    )
    return BatchIngestResponse(record_count=len(cases), cases=cases)


@router.post("/audit-record", response_model=CaseFullResponse)
def audit_record(
    body: FeatureRecordInput,
    use_case: AuditPipelineService = Depends(get_ingest_cases),
    container: AppContainer = Depends(get_container),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CaseFullResponse:
    """保留兼容统计特征建案能力（向后兼容旧版接入）。

    参数:
        body: 统计特征记录输入。
        use_case: 建案流水线用例。

    返回:
        CaseFullResponse: 生成的完整案件详情。
    """

    if container.settings.public_demo_mode:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "public_demo_legacy_ingest_disabled",
                "message": "公网演示环境仅接收完整 81 字段脱敏记录。",
            },
        )
    response = use_case.run(body, source="json")
    prestart_review_advisor(container, response.case.case_id, current_user)
    return response


def _enforce_public_demo_ingest_capacity(
    container: AppContainer,
    *,
    requested_records: int,
) -> None:
    settings = container.settings
    if not settings.public_demo_mode:
        return
    if requested_records < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "public_demo_empty_batch",
                "message": "至少需要一条脱敏记录。",
            },
        )
    if requested_records > settings.public_demo_max_batch_records:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "public_demo_batch_too_large",
                "message": (
                    "单次最多导入 "
                    f"{settings.public_demo_max_batch_records} 条脱敏记录。"
                ),
            },
        )
    if container.cases.case_count + requested_records > settings.public_demo_max_cases:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "public_demo_case_capacity_reached",
                "message": "公网演示案件容量已满，请联系项目所有者清理后再试。",
            },
        )
