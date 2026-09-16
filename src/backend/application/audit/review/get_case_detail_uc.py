"""案件查询用例。"""

from ...ports.repositories import CaseRepository, NoteRepository, ReviewRepository
from ....core.exceptions import ResourceNotFoundError
from ....domain.audit.review.evidence_packager import EvidenceService
from ....domain.audit.review.workflow_projector import CaseFullResponse, WorkflowResponse
from ...ports.repositories import TraceRepository


class GetCaseDetailUseCase:
    """查询案件详情和九阶段流程。"""

    def __init__(
        self,
        cases: CaseRepository,
        reviews: ReviewRepository,
        notes: NoteRepository,
        traces: TraceRepository,
        evidence: EvidenceService,
    ) -> None:
        self.cases = cases
        self.reviews = reviews
        self.notes = notes
        self.traces = traces
        self.evidence = evidence

    def execute(self, case_id: str) -> CaseFullResponse:
        """返回案件、证据、人工初审和工作笔记。"""

        case = self.cases.get_case(case_id)
        if case is None:
            raise ResourceNotFoundError(f"案件 {case_id} 不存在")
        return CaseFullResponse(
            case=case,
            evidence=self.evidence.generate(case),
            review=self.reviews.get(case_id),
            notes=self.notes.list_for_case(case_id),
        )

    def workflow(self, case_id: str) -> WorkflowResponse:
        """返回由后端统一生成的九阶段流程。"""

        if self.cases.get_case(case_id) is None:
            raise ResourceNotFoundError(f"案件 {case_id} 不存在")
        return self.traces.build_workflow(
            case_id,
            self.reviews.get(case_id),
        )
