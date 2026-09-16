"""审核管理路由。

合并审核队列浏览和单案办理两类端点：
- 队列：案件列表、分页、状态筛选
- 办理：案件详情、九阶段流程、风险/规则/证据、人工初审、工作笔记、统计材料
"""

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..agent_autostart import prestart_review_advisor
from ..dependencies import (
    AppContainer,
    get_case_detail,
    get_container,
    get_current_user,
    get_list_cases,
    get_manage_notes,
    get_review_case,
    get_statistical_materials as get_statistical_materials_use_case,
)
from ...application.audit.queue.list_cases_uc import ListCasesUseCase
from ...application.audit.review.get_case_detail_uc import GetCaseDetailUseCase
from ...application.audit.review.submit_review_uc import ReviewCaseUseCase
from ...application.audit.review.manage_notes_uc import ManageNotesUseCase
from ...application.audit.review.build_materials_uc import StatisticalMaterialUseCase
from ...core.exceptions import ResourceNotFoundError
from ...domain.audit.review.entities import AuthenticatedUser, MedicalRecordResponse
from ..schemas.cases import CaseFullResponse, CaseSummary
from ..schemas.workflow import WorkflowResponse
from ..schemas.audit import AuditNote, AuditNoteInput, ReviewInput, ReviewResponse

router = APIRouter(prefix="/cases", tags=["audit"])


# ── 审核队列 ──────────────────────────────────────────


@router.get("", response_model=list[CaseSummary])
def list_cases(
    use_case: ListCasesUseCase = Depends(get_list_cases),
) -> list[CaseSummary]:
    """返回审核队列案件摘要列表。

    参数:
        use_case: 案件列表用例。

    返回:
        list[CaseSummary]: 当前仓储中所有案件的摘要（含风险等级、审核状态）。
    """

    return use_case.execute()


# ── 案件详情与流程 ────────────────────────────────────


@router.get("/{case_id}", response_model=CaseFullResponse)
def get_case(
    case_id: str,
    use_case: GetCaseDetailUseCase = Depends(get_case_detail),
    container: AppContainer = Depends(get_container),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CaseFullResponse:
    """返回案件完整详情。

    参数:
        case_id: 案件唯一标识，如 ``CASE-...``。
        use_case: 案件详情用例。

    返回:
        CaseFullResponse: 含 CaseDetail + EvidencePackage + ReviewDecision + AuditNote 列表。
    """

    response = use_case.execute(case_id)
    prestart_review_advisor(container, response.case.case_id, current_user)
    return response


@router.get("/{case_id}/workflow", response_model=WorkflowResponse)
def get_case_workflow(
    case_id: str,
    use_case: GetCaseDetailUseCase = Depends(get_case_detail),
) -> WorkflowResponse:
    """返回后端统一维护的九阶段稽核流程。

    参数:
        case_id: 案件唯一标识。
        use_case: 案件详情用例。

    返回:
        WorkflowResponse: 九个阶段节点及其完成状态、内容摘要。
    """

    return use_case.workflow(case_id)


# ── 人工初审 ──────────────────────────────────────────


@router.post("/{case_id}/review", response_model=ReviewResponse)
def submit_review(
    case_id: str,
    body: ReviewInput,
    use_case: ReviewCaseUseCase = Depends(get_review_case),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ReviewResponse:
    """保存人工初审决定，不自动触发复审或申诉。

    参数:
        case_id: 案件唯一标识。
        body: 初审意见（决定类型、理由、附件材料元数据）。
        use_case: 初审用例。
        current_user: 当前登录审核人员。

    返回:
        ReviewResponse: 已保存的初审记录 + 更新后的 Trace 节点。
    """

    return use_case.execute(case_id, body, current_user)


# ── 工作笔记 ──────────────────────────────────────────


@router.post("/{case_id}/notes", response_model=AuditNote)
def add_note(
    case_id: str,
    body: AuditNoteInput,
    use_case: ManageNotesUseCase = Depends(get_manage_notes),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuditNote:
    """记录一条案件工作笔记。

    参数:
        case_id: 案件唯一标识。
        body: 笔记内容（来源类型、正文、关联材料元数据）。
        use_case: 笔记管理用例。
        current_user: 当前登录审核人员。

    返回:
        AuditNote: 已保存的笔记（含 note_id 和创建时间）。
    """

    return use_case.add(case_id, body, current_user)


@router.post("/{case_id}/notes/{note_id}/void", response_model=AuditNote)
def void_note(
    case_id: str,
    note_id: str,
    use_case: ManageNotesUseCase = Depends(get_manage_notes),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuditNote:
    """兼容旧作废入口：当前产品口径为直接删除。

    参数:
        case_id: 案件唯一标识。
        note_id: 笔记唯一标识。
        use_case: 笔记管理用例。
        current_user: 当前登录审核人员。

    返回:
        AuditNote: 已作废的笔记。
    """

    return use_case.void(case_id, note_id, current_user)


@router.delete("/{case_id}/notes/{note_id}", response_model=AuditNote)
def delete_note(
    case_id: str,
    note_id: str,
    use_case: ManageNotesUseCase = Depends(get_manage_notes),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuditNote:
    """删除一条案件工作笔记。

    参数:
        case_id: 案件唯一标识。
        note_id: 笔记唯一标识。
        use_case: 笔记管理用例。
        current_user: 当前登录审核人员。

    返回:
        AuditNote: 已删除的笔记。
    """

    return use_case.delete(case_id, note_id, current_user)


# ── 统计材料视图 ──────────────────────────────────────


@router.get(
    "/{case_id}/statistical-materials",
    response_model=MedicalRecordResponse,
)
def get_statistical_materials(
    case_id: str,
    use_case: StatisticalMaterialUseCase = Depends(get_statistical_materials_use_case),
    container: AppContainer = Depends(get_container),
) -> MedicalRecordResponse:
    """返回由案件 81 字段聚合生成的统计材料视图。

    注意：这是脱敏字段聚合产物，不是真实病历、票据或医学结论。

    参数:
        case_id: 案件唯一标识。
        use_case: 统计材料用例。
        container: 应用依赖容器。

    返回:
        MedicalRecordResponse: 就诊行为、药品费用、费用结构三类只读材料文档。
    """

    case = container.cases.get_case(case_id)
    if case is None:
        raise ResourceNotFoundError(f"案件 {case_id} 不存在")
    return use_case.build_for_case(case)


@router.get("/{case_id}/materials/{material_id}/assets/{asset_id}")
def preview_business_material_asset(
    case_id: str,
    material_id: str,
    asset_id: str,
    download: bool = False,
    use_case: StatisticalMaterialUseCase = Depends(get_statistical_materials_use_case),
    container: AppContainer = Depends(get_container),
) -> FileResponse:
    """返回业务材料 PNG 受控预览文件。"""

    case = container.cases.get_case(case_id)
    if case is None:
        raise ResourceNotFoundError(f"案件 {case_id} 不存在")
    path, asset = use_case.resolve_asset(case_id, material_id, asset_id)
    return FileResponse(
        path,
        media_type=asset.file_type,
        filename=asset.display_name,
        content_disposition_type="attachment" if download else "inline",
    )
