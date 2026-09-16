"""应用层使用的仓储端口。

端口只描述业务需要的最小能力，内存与未来数据库实现遵循同一契约。
"""

from typing import Protocol

from ...domain.audit.review.entities import (
    AuthenticatedUser,
    CaseDetail,
    CaseSummary,
    MaterialAssetLocation,
    StoredUser,
)
from ...domain.audit.review.evidence_packager import EvidencePackage
from ...domain.audit.review.workflow_projector import (
    AuditNote,
    ReviewDecision,
    TraceNode,
    WorkflowResponse,
)


class CaseRepository(Protocol):
    """案件及接入幂等索引仓储。"""

    @property
    def case_count(self) -> int: ...

    def clear(self) -> None: ...

    def add_case(
        self,
        case: CaseDetail,
        fingerprint: str | None = None,
        evidence_package: EvidencePackage | None = None,
    ) -> None: ...

    def add_cases(
        self,
        entries: list[
            tuple[CaseDetail, str | None]
            | tuple[CaseDetail, str | None, EvidencePackage | None]
        ],
    ) -> None: ...

    def get_case(self, case_id: str) -> CaseDetail | None: ...

    def find_by_fingerprint(self, fingerprint: str) -> CaseDetail | None: ...

    def list_cases(
        self, reviewed_case_ids: set[str] | None = None
    ) -> list[CaseSummary]: ...


class TraceRepository(Protocol):
    """案件流程追踪仓储。"""

    def clear(self) -> None: ...

    def init_from_fixture(
        self, case_id: str, initial_nodes: list[TraceNode]
    ) -> None: ...

    def init_many(
        self, entries: list[tuple[str, list[TraceNode]]]
    ) -> None: ...

    def get_trace(self, case_id: str) -> list[TraceNode]: ...

    def append_node(
        self,
        case_id: str,
        node_id: str,
        node_name: str,
        summary: str,
    ) -> TraceNode: ...

    def update_pending_to_completed(
        self, case_id: str, pending_node_id: str
    ) -> TraceNode | None: ...

    def build_workflow(
        self,
        case_id: str,
        review: ReviewDecision | None = None,
    ) -> WorkflowResponse: ...


class ReviewRepository(Protocol):
    """人工初审记录仓储。"""

    def clear(self) -> None: ...

    def get(self, case_id: str) -> ReviewDecision | None: ...

    def save(
        self,
        case_id: str,
        review: ReviewDecision,
        actor: AuthenticatedUser | None = None,
    ) -> None: ...

    def case_ids(self) -> set[str]: ...


class NoteRepository(Protocol):
    """审核工作笔记仓储。"""

    def clear(self) -> None: ...

    def list_for_case(self, case_id: str) -> list[AuditNote]: ...

    def add(
        self,
        case_id: str,
        note: AuditNote,
        actor: AuthenticatedUser | None = None,
    ) -> None: ...

    def delete(
        self,
        case_id: str,
        note_id: str,
        actor: AuthenticatedUser | None = None,
    ) -> AuditNote | None: ...


class UserRepository(Protocol):
    """审核人员账号仓储。"""

    def get_by_username(self, username: str) -> StoredUser | None: ...

    def get_by_id(self, user_id: str) -> StoredUser | None: ...

    def record_login(self, user_id: str) -> None: ...

    def get_case_memory_enabled(self, user_id: str) -> bool: ...

    def set_case_memory_enabled(self, user_id: str, enabled: bool) -> bool: ...

    def log_auth_event(
        self,
        username: str,
        success: bool,
        reason: str,
        user_id: str | None = None,
    ) -> None: ...


class MaterialRepository(Protocol):
    """业务材料快照与受控资产仓储。"""

    def clear(self) -> None: ...

    def get_snapshot(self, case_id: str) -> dict | None: ...

    def save_snapshot(
        self,
        case_id: str,
        snapshot: dict,
        assets: list[MaterialAssetLocation],
    ) -> None: ...

    def get_asset(
        self,
        case_id: str,
        material_id: str,
        asset_id: str,
    ) -> MaterialAssetLocation | None: ...
