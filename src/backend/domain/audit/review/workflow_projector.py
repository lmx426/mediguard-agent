"""工作流投影模块。

合并自：
- domain/models/workflow.py —— Trace、九阶段流程、人工初审和工作笔记模型
- domain/workflow.py —— 工作流组装辅助函数

Trace 记录整个审核流程的关键节点。
人工审核意见与 Agent 建议分开展示，是最终业务判断来源。
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .entities import CaseDetail
from .evidence_packager import EvidencePackage


# ============================================================================
# 来自 domain/models/workflow.py —— 工作流实体模型
# ============================================================================


class TraceNode(BaseModel):
    """Trace 时间线中的一个节点。"""

    node_id: str = Field(description="节点唯一标识，如 trace:case_selected")
    node_name: str = Field(description="节点名称，如 案件已选择")
    status: str = Field(description="节点状态：completed | pending")
    summary: str = Field(description="节点摘要描述")
    order: int = Field(description="时间顺序序号")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化追踪元数据，用于记录规则池版本、基线版本和评估摘要",
    )


class WorkflowStep(BaseModel):
    """前端流程导航可直接渲染的业务步骤。"""

    key: str = Field(description="业务步骤唯一标识，如 risk_screening")
    title: str = Field(description="步骤标题，如 风险筛查")
    status: str = Field(
        description="业务状态：completed | current | pending | recorded | conditional"
    )
    summary: str = Field(description="步骤摘要，面向审核员展示")
    content_key: str = Field(
        description="该步骤在前端对应的内容区域，如 risk_summary"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="折叠展示的结构化追踪元数据，不含 RES",
    )


class WorkflowResponse(BaseModel):
    """GET /api/cases/{case_id}/workflow 的响应。"""

    case_id: str
    current_step: str
    steps: list[WorkflowStep]


class ReviewAttachment(BaseModel):
    """人工初审材料登记元数据。

    只登记材料名称、类型、来源、核验状态和备注，不上传或保存文件内容。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120, description="材料名称")
    material_type: str = Field(min_length=1, max_length=80, description="材料类型")
    source: str = Field(default="审核人员登记", max_length=120, description="材料来源")
    verification_status: str = Field(
        default="待核验",
        max_length=80,
        description="核验状态",
    )
    remark: str = Field(default="", max_length=500, description="备注")
    file_name: str | None = Field(default=None, max_length=180, description="本地文件名")
    file_size: int | None = Field(default=None, ge=0, description="本地文件大小，字节")
    file_type: str | None = Field(default=None, max_length=120, description="本地文件类型")
    file_last_modified: str | None = Field(
        default=None,
        max_length=80,
        description="本地文件最后修改时间",
    )


class ReviewInput(BaseModel):
    """人工审核意见输入——API 请求体。

    只包含用户需要填写的字段。
    """

    reviewer: str = Field(default="", max_length=80, description="审核员标识")
    decision: str = Field(description="审核结论，如 需进一步调查 | 常规处理")
    reason: str = Field(min_length=1, description="审核理由，必填")
    attachments: list[ReviewAttachment] = Field(
        default_factory=list,
        description="人工初审附件材料登记元数据，不包含文件内容",
    )


class ReviewDecision(ReviewInput):
    """人工审核意见——含提交时间的完整记录。

    约束：
    - 与 Agent 建议分开展示
    - 必须填写理由
    - 是最终业务判断来源
    """

    submitted_at: str = Field(description="提交时间，ISO 格式")


class AuditNoteInput(BaseModel):
    """案件级审核工作笔记输入。

    智能助手来源仅表示审核员已人工核验并摘录，不代表 Agent 自动写入。
    关联材料只保存元数据，不上传或保存文件内容。
    """

    author: str = Field(default="", max_length=80, description="笔记记录人")
    source: Literal["manual", "assistant", "external", "case_agent_adopted"] = Field(
        default="manual",
        description="笔记来源：manual | assistant | external | case_agent_adopted",
    )
    content: str = Field(
        min_length=1,
        max_length=2000,
        description="审核工作笔记内容",
    )
    source_refs: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="工作笔记采纳或引用的证据来源引用",
    )
    materials: list[ReviewAttachment] = Field(
        default_factory=list,
        max_length=10,
        description="工作笔记关联材料元数据，不包含文件内容",
    )


class AuditNote(AuditNoteInput):
    """包含标识和记录时间的完整审核工作笔记。"""

    note_id: str = Field(description="笔记唯一标识")
    created_at: str = Field(description="记录时间，ISO 格式")
    status: Literal["active", "voided"] = Field(
        default="active",
        description="笔记状态：active | voided",
    )
    voided_at: str | None = Field(default=None, description="作废时间，ISO 格式")
    voided_by: str | None = Field(default=None, description="作废操作人")


class CaseFullResponse(BaseModel):
    """GET /api/cases/{case_id} 的完整响应。"""

    case: CaseDetail
    evidence: EvidencePackage
    review: ReviewDecision | None = None
    notes: list[AuditNote] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    """POST /api/cases/{case_id}/review 响应。"""

    review: ReviewDecision
    trace_node: TraceNode


# ============================================================================
# 来自 domain/workflow.py —— 工作流组装辅助
# ============================================================================

# Trace 节点到工作流阶段的映射
WORKFLOW_NODE_MAP = {
    "case_intake": "trace:case_received",
    "fact_base": "trace:data_integrity_validated",
    "rule_check": "trace:rules_evaluated",
    "risk_screening": "trace:risk_screened",
    "evidence_package": "trace:evidence_organized",
    "initial_review": "trace:manual_review_pending",
}

# 九阶段工作流中文标题
WORKFLOW_TITLES = {
    "case_intake": "案件接入",
    "fact_base": "事实底座",
    "rule_check": "规则核验",
    "risk_screening": "风险筛查",
    "evidence_package": "证据包与审核建议",
    "initial_review": "人工初审",
    "secondary_review": "人工复审",
    "appeal_handling": "申诉处理",
    "case_result": "案件处理结果",
}

# 各阶段对应的前端内容区域标识
WORKFLOW_CONTENT_KEYS = {
    "case_intake": "case_intake",
    "fact_base": "fact_base",
    "rule_check": "rule_check",
    "risk_screening": "risk_screening",
    "evidence_package": "evidence_package",
    "initial_review": "initial_review",
    "secondary_review": "secondary_review",
    "appeal_handling": "appeal_handling",
    "case_result": "case_result",
}


def build_workflow_response(
    case_id: str,
    trace_nodes: list[TraceNode],
    review: ReviewDecision | None = None,
) -> WorkflowResponse:
    """构建前端使用的稳定九阶段工作流响应。

    参数：
        case_id: 案件唯一标识。
        trace_nodes: Trace 时间线节点列表。
        review: 人工审核意见，可选。None 表示尚未提交初审。

    返回：
        包含九阶段步骤列表的工作流响应。
    """

    # 将 trace 节点按 ID 建立索引，便于快速查找
    trace_by_id = {node.node_id: node for node in trace_nodes}
    has_review = review is not None
    current_step = "case_result" if has_review else "initial_review"
    steps: list[WorkflowStep] = []

    # 前六个阶段：按 trace 状态生成
    for key in [
        "case_intake",
        "fact_base",
        "rule_check",
        "risk_screening",
        "evidence_package",
        "initial_review",
    ]:
        node = trace_by_id.get(WORKFLOW_NODE_MAP[key])
        steps.append(
            WorkflowStep(
                key=key,
                title=WORKFLOW_TITLES[key],
                status="completed" if key != "initial_review" or has_review else "current",
                summary=(
                    node.summary
                    if node is not None
                    else f"{WORKFLOW_TITLES[key]}尚未形成记录"
                ),
                content_key=WORKFLOW_CONTENT_KEYS[key],
                metadata=node.metadata if node is not None else {},
            )
        )

    # 第七、八阶段：条件阶段，当前后端流程未接入
    for key in ["secondary_review", "appeal_handling"]:
        steps.append(
            WorkflowStep(
                key=key,
                title=WORKFLOW_TITLES[key],
                status="conditional",
                summary=f"{WORKFLOW_TITLES[key]}为条件阶段，当前后端流程未接入",
                content_key=WORKFLOW_CONTENT_KEYS[key],
                metadata={
                    "backend_connected": False,
                    "activation": "conditional",
                },
            )
        )

    # 第九阶段：案件处理结果
    review_node = trace_by_id.get("trace:manual_review_submitted")
    steps.append(
        WorkflowStep(
            key="case_result",
            title=WORKFLOW_TITLES["case_result"],
            status="recorded" if has_review else "pending",
            summary=(
                f"已记录人工初审意见：{review.decision}"
                if review is not None
                else "待人工初审后形成案件处理结果"
            ),
            content_key=WORKFLOW_CONTENT_KEYS["case_result"],
            metadata=review_node.metadata if review_node is not None else {},
        )
    )

    return WorkflowResponse(
        case_id=case_id,
        current_step=current_step,
        steps=steps,
    )
