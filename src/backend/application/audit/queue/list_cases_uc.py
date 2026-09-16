"""审核队列查询用例。"""

from ...ports.repositories import CaseRepository, ReviewRepository
from ....domain.audit.review.entities import CaseSummary


class ListCasesUseCase:
    """返回包含人工初审状态的案件摘要。"""

    def __init__(
        self,
        cases: CaseRepository,
        reviews: ReviewRepository,
    ) -> None:
        self.cases = cases
        self.reviews = reviews

    def execute(self) -> list[CaseSummary]:
        """按仓储当前顺序返回审核队列。"""

        return self.cases.list_cases(self.reviews.case_ids())
