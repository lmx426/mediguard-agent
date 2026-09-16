"""审核工作笔记用例。"""

from datetime import datetime
from uuid import uuid4

from ...ports.repositories import CaseRepository, NoteRepository, ReviewRepository
from ....core.exceptions import BusinessValidationError, ResourceNotFoundError
from ....domain.audit.review.entities import AuthenticatedUser, attachment_metadata_text, find_forbidden_content
from ....domain.audit.review.workflow_projector import AuditNote, AuditNoteInput


class ManageNotesUseCase:
    """记录人工输入的案件工作笔记。"""

    def __init__(
        self,
        cases: CaseRepository,
        notes: NoteRepository,
        reviews: ReviewRepository,
    ) -> None:
        self.cases = cases
        self.notes = notes
        self.reviews = reviews

    def add(
        self,
        case_id: str,
        body: AuditNoteInput,
        author: AuthenticatedUser | None = None,
    ) -> AuditNote:
        """校验并追加工作笔记，不改变案件流程状态。"""

        if self.cases.get_case(case_id) is None:
            raise ResourceNotFoundError(f"案件 {case_id} 不存在")

        content = body.content.strip()
        author_name = (
            author.display_name.strip()
            if author is not None
            else body.author.strip()
        )
        errors: list[str] = []
        if not content:
            errors.append("笔记内容不能为空")
        if not author_name:
            errors.append("记录人不能为空")
        forbidden = find_forbidden_content(
            "\n".join([content, attachment_metadata_text(body.materials)])
        )
        if forbidden:
            errors.append(f"审核工作笔记不得包含禁止字段或身份信息：{forbidden}")
        if errors:
            raise BusinessValidationError(errors)

        note = AuditNote(
            note_id=f"NOTE-{uuid4().hex[:8].upper()}",
            author=author_name,
            source=body.source,
            content=content,
            materials=body.materials,
            created_at=datetime.now().isoformat(),
        )
        self.notes.add(case_id, note, author)
        return note

    def delete(
        self,
        case_id: str,
        note_id: str,
        actor: AuthenticatedUser | None = None,
    ) -> AuditNote:
        """删除一条工作笔记。"""

        if self.cases.get_case(case_id) is None:
            raise ResourceNotFoundError(f"案件 {case_id} 不存在")
        if self.reviews.get(case_id) is not None:
            raise BusinessValidationError(["人工初审已提交，工作笔记不可删除"])
        note = self.notes.delete(case_id, note_id, actor)
        if note is None:
            raise ResourceNotFoundError(f"工作笔记 {note_id} 不存在")
        return note

    def void(
        self,
        case_id: str,
        note_id: str,
        actor: AuthenticatedUser | None = None,
    ) -> AuditNote:
        """兼容旧作废入口：当前产品口径为直接删除。"""

        return self.delete(case_id, note_id, actor)
