"""Trace 与九阶段 workflow 内存服务。

管理每个案件的 Trace 时间线：
- 从 fixture 或接入流水线加载初始节点
- 根据 Trace 与人工初审记录生成九阶段业务流程
- 运行时追加节点（manual_review_submitted）
- 所有状态仅在内存中，重启后丢失
"""

from datetime import datetime

from ....domain.audit.review.workflow_projector import (
    ReviewDecision,
    TraceNode,
    WorkflowResponse,
    build_workflow_response,
)


WORKFLOW_NODE_MAP = {
    "case_intake": "trace:case_received",
    "fact_base": "trace:data_integrity_validated",
    "rule_check": "trace:rules_evaluated",
    "risk_screening": "trace:risk_screened",
    "evidence_package": "trace:evidence_organized",
    "initial_review": "trace:manual_review_pending",
}

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


class TraceService:
    """Trace 时间线管理。

    v1.0 使用内存 dict 存储每个案件的 Trace 节点列表。
    服务重启后所有运行时追加的节点丢失，仅 fixture 初始节点可恢复。
    """

    def __init__(self) -> None:
        self._traces: dict[str, list[TraceNode]] = {}

    def clear(self) -> None:
        """清空所有 Trace。"""
        self._traces.clear()

    def init_from_fixture(
        self, case_id: str, initial_nodes: list[TraceNode]
    ) -> None:
        """从 fixture 加载案件的初始 Trace 节点。"""
        self._traces[case_id] = list(initial_nodes)

    def init_many(
        self,
        entries: list[tuple[str, list[TraceNode]]],
    ) -> None:
        """批量提交已经构建完成的 Trace。"""

        for case_id, nodes in entries:
            self.init_from_fixture(case_id, nodes)

    def get_trace(self, case_id: str) -> list[TraceNode]:
        """获取指定案件的完整 Trace 时间线。

        如果案件尚未初始化，返回空列表。
        """
        return self._traces.get(case_id, [])

    def append_node(
        self,
        case_id: str,
        node_id: str,
        node_name: str,
        summary: str,
    ) -> TraceNode:
        """追加一个新的 Trace 节点到案件时间线末尾。

        Args:
            case_id: 案件 ID。
            node_id: 节点唯一标识。
            node_name: 节点名称。
            summary: 节点摘要。

        Returns:
            新创建的 TraceNode。
        """
        existing = self._traces.get(case_id, [])
        next_order = max((n.order for n in existing), default=0) + 1
        new_node = TraceNode(
            node_id=node_id,
            node_name=node_name,
            status="completed",
            summary=summary,
            order=next_order,
        )
        if case_id not in self._traces:
            self._traces[case_id] = []
        self._traces[case_id].append(new_node)
        return new_node

    def update_pending_to_completed(
        self, case_id: str, pending_node_id: str
    ) -> TraceNode | None:
        """将指定的 pending 节点标记为 completed。

        用于人工审核提交后，将 manual_review_pending 更新为 completed。
        """
        nodes = self._traces.get(case_id, [])
        for node in nodes:
            if node.node_id == pending_node_id and node.status == "pending":
                node.status = "completed"
                return node
        return None

    def build_workflow(
        self,
        case_id: str,
        review: ReviewDecision | None = None,
    ) -> WorkflowResponse:
        """构建后端统一维护的九阶段业务流程。

        人工复审与申诉处理在第一阶段始终是条件阶段，不根据风险或初审结果
        自动启动，避免把未接入流程伪装成真实生产状态。
        """

        return build_workflow_response(case_id, self.get_trace(case_id), review)

    @staticmethod
    def _workflow_status(key: str, has_review: bool) -> str:
        if key == "initial_review":
            return "completed" if has_review else "current"
        return "completed"
