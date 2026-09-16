"""TraceService 单元测试。"""

import pytest

from src.backend.domain.audit.review.workflow_projector import ReviewDecision, TraceNode
from src.backend.infrastructure.persistence.memory.trace_repository import TraceService


class TestTraceService:
    def setup_method(self):
        self.service = TraceService()

    def test_init_from_fixture(self):
        nodes = [
            TraceNode(
                node_id="trace:case_selected",
                node_name="案件已选择",
                status="completed",
                summary="已选择",
                order=1,
            ),
            TraceNode(
                node_id="trace:manual_review_pending",
                node_name="待人工审核",
                status="pending",
                summary="等待",
                order=2,
            ),
        ]
        self.service.init_from_fixture("CASE-001", nodes)

        trace = self.service.get_trace("CASE-001")
        assert len(trace) == 2
        assert trace[0].status == "completed"
        assert trace[1].status == "pending"

    def test_append_node(self):
        self.service.init_from_fixture("CASE-001", [])
        new_node = self.service.append_node(
            case_id="CASE-001",
            node_id="trace:test",
            node_name="测试节点",
            summary="测试",
        )

        assert new_node.node_id == "trace:test"
        assert new_node.status == "completed"
        assert new_node.order == 1

        trace = self.service.get_trace("CASE-001")
        assert len(trace) == 1

    def test_append_node_increments_order(self):
        nodes = [
            TraceNode(
                node_id="trace:n1", node_name="N1",
                status="completed", summary="", order=1,
            ),
            TraceNode(
                node_id="trace:n2", node_name="N2",
                status="completed", summary="", order=2,
            ),
        ]
        self.service.init_from_fixture("CASE-X", nodes)

        new_node = self.service.append_node(
            case_id="CASE-X",
            node_id="trace:n3",
            node_name="N3",
            summary="",
        )
        assert new_node.order == 3

    def test_update_pending_to_completed(self):
        nodes = [
            TraceNode(
                node_id="trace:manual_review_pending",
                node_name="待人工审核",
                status="pending",
                summary="等待",
                order=1,
            ),
        ]
        self.service.init_from_fixture("CASE-001", nodes)

        updated = self.service.update_pending_to_completed(
            "CASE-001", "trace:manual_review_pending"
        )
        assert updated is not None
        assert updated.status == "completed"

        trace = self.service.get_trace("CASE-001")
        assert trace[0].status == "completed"

    def test_update_nonexistent_node_returns_none(self):
        result = self.service.update_pending_to_completed(
            "CASE-UNKNOWN", "trace:nonexistent"
        )
        assert result is None

    def test_get_trace_unknown_case_returns_empty(self):
        trace = self.service.get_trace("CASE-NONEXISTENT")
        assert trace == []

    def test_build_workflow_pending_review(self):
        nodes = [
            TraceNode(
                node_id="trace:case_received",
                node_name="案件接收",
                status="completed",
                summary="已接收",
                order=1,
            ),
            TraceNode(
                node_id="trace:rules_evaluated",
                node_name="规则核验",
                status="completed",
                summary="已核验",
                order=4,
                metadata={"total_rules": 8},
            ),
            TraceNode(
                node_id="trace:manual_review_pending",
                node_name="待人工初审",
                status="pending",
                summary="等待审核员提交处理意见",
                order=6,
            ),
        ]
        self.service.init_from_fixture("CASE-001", nodes)

        workflow = self.service.build_workflow("CASE-001")

        assert workflow.current_step == "initial_review"
        assert len(workflow.steps) == 9
        assert workflow.steps[5].key == "initial_review"
        assert workflow.steps[5].status == "current"
        assert workflow.steps[6].status == "conditional"
        assert workflow.steps[7].status == "conditional"
        assert workflow.steps[-1].key == "case_result"
        assert workflow.steps[-1].status == "pending"
        assert workflow.steps[-1].summary == "待人工初审后形成案件处理结果"
        rule_step = next(step for step in workflow.steps if step.key == "rule_check")
        assert rule_step.metadata["total_rules"] == 8

    def test_build_workflow_with_review_recorded(self):
        self.service.init_from_fixture("CASE-001", [])
        review = ReviewDecision(
            reviewer="审核员",
            decision="进入人工复核",
            reason="规则线索需要人工确认",
            submitted_at="2026-07-22T10:00:00",
        )

        workflow = self.service.build_workflow("CASE-001", review)

        assert workflow.current_step == "case_result"
        assert workflow.steps[5].status == "completed"
        assert workflow.steps[6].status == "conditional"
        assert workflow.steps[7].status == "conditional"
        assert workflow.steps[-1].status == "recorded"
        assert "进入人工复核" in workflow.steps[-1].summary
