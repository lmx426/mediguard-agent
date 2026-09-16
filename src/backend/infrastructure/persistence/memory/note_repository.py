"""审核工作笔记的进程内仓储。"""

from ....domain.audit.review.entities import AuthenticatedUser
from ....domain.audit.review.workflow_projector import AuditNote


class MemoryNoteRepository:
    """按案件保存人工录入的审核工作笔记。"""

    def __init__(self) -> None:
        self._notes: dict[str, list[AuditNote]] = {}

    def clear(self) -> None:
        """清空全部工作笔记。"""

        self._notes.clear()

    def list_for_case(self, case_id: str) -> list[AuditNote]:
        """返回案件笔记副本，避免调用方直接修改内部列表。"""

        return list(self._notes.get(case_id, []))

    def add(
        self,
        case_id: str,
        note: AuditNote,
        actor: AuthenticatedUser | None = None,
    ) -> None:
        """向指定案件追加一条工作笔记。"""

        self._notes.setdefault(case_id, []).append(note)

    def delete(
        self,
        case_id: str,
        note_id: str,
        actor: AuthenticatedUser | None = None,
    ) -> AuditNote | None:
        """删除指定工作笔记。"""

        notes = self._notes.get(case_id, [])
        for index, note in enumerate(notes):
            if note.note_id != note_id:
                continue
            return notes.pop(index)
        return None
