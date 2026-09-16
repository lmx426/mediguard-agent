"""人工初审用例。"""

from datetime import datetime

from ...ports.repositories import CaseRepository, ReviewRepository, TraceRepository
from ....core.exceptions import BusinessValidationError, ResourceNotFoundError
from ....domain.audit.review.entities import AuthenticatedUser, find_forbidden_content, review_attachment_text
from ....domain.audit.review.workflow_projector import ReviewDecision, ReviewInput, ReviewResponse


class ReviewCaseUseCase:
    """保存人工初审并更新对应 Trace，不触发复审或申诉。"""

    def __init__(
        self,
        cases: CaseRepository,
        reviews: ReviewRepository,
        traces: TraceRepository,
    ) -> None:
        self.cases = cases
        self.reviews = reviews
        self.traces = traces

    def execute(
        self,
        case_id: str,
        body: ReviewInput,
        reviewer: AuthenticatedUser | None = None,
    ) -> ReviewResponse:
        """校验并保存一条人工初审记录。

        Args:
            case_id: 被审核案件编号。
            body: 人工填写的决定、理由和附件元数据。

        Raises:
            ResourceNotFoundError: 案件不存在。
            BusinessValidationError: 理由为空或材料元数据包含禁止内容。
        """

        if self.cases.get_case(case_id) is None:
            raise ResourceNotFoundError(f"案件 {case_id} 不存在")

        reason = body.reason.strip()
        if not reason:
            raise BusinessValidationError(["审核理由不能为空"])

        forbidden = find_forbidden_content(review_attachment_text(body))
        if forbidden:
            raise BusinessValidationError(
                [f"附件材料登记不得包含禁止字段或身份信息：{forbidden}"]
            )

        reviewer_name = (
            reviewer.display_name.strip()
            if reviewer is not None
            else body.reviewer.strip() or "审核员"
        )
        review = ReviewDecision(
            reviewer=reviewer_name,
            decision=body.decision,
            reason=reason,
            attachments=body.attachments,
            submitted_at=datetime.now().isoformat(),
        )
        self.reviews.save(case_id, review, reviewer)
        self.traces.update_pending_to_completed(
            case_id,
            "trace:manual_review_pending",
        )
        trace_node = self.traces.append_node(
            case_id=case_id,
            node_id="trace:manual_review_submitted",
            node_name="人工初审已提交",
            summary=f"{reviewer_name}提交处理意见：{body.decision}",
        )
        return ReviewResponse(review=review, trace_node=trace_node)
