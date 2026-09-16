"""人工初审记录的进程内仓储。"""

from ....domain.audit.review.entities import AuthenticatedUser
from ....domain.audit.review.workflow_projector import ReviewDecision


class MemoryReviewRepository:
    """按案件保存当前人工初审记录，服务重启后清空。"""

    def __init__(self) -> None:
        self._reviews: dict[str, ReviewDecision] = {}

    def clear(self) -> None:
        """清空全部人工初审记录。"""

        self._reviews.clear()

    def get(self, case_id: str) -> ReviewDecision | None:
        """返回案件当前初审记录，不存在时返回 ``None``。"""

        return self._reviews.get(case_id)

    def save(
        self,
        case_id: str,
        review: ReviewDecision,
        actor: AuthenticatedUser | None = None,
    ) -> None:
        """保存案件当前初审记录。"""

        self._reviews[case_id] = review

    def case_ids(self) -> set[str]:
        """返回已经提交人工初审的案件编号集合。"""

        return set(self._reviews)
