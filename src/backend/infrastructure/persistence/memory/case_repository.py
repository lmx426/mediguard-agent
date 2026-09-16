"""案件查询服务。

管理接入后生成案件的内存状态：写入 → 查询。
v1.0 不持久化，服务重启后状态重置。
"""

from ....domain.audit.review.entities import CaseDetail, CaseSummary
from ....domain.audit.review.evidence_packager import EvidencePackage


class CaseService:
    """案件查询与内存状态管理。

    v1.0 使用内存 dict 存储，服务重启后所有运行时状态重置。
    """

    def __init__(self) -> None:
        self._cases: dict[str, CaseDetail] = {}
        self._fingerprints: dict[str, str] = {}

    def clear(self) -> None:
        """清空内存案件。"""
        self._cases.clear()
        self._fingerprints.clear()

    def add_case(
        self,
        case: CaseDetail,
        fingerprint: str | None = None,
        evidence_package: EvidencePackage | None = None,
    ) -> None:
        """添加案件，并可同时登记完整宽表幂等指纹。"""

        _ = evidence_package
        self._cases[case.case_id] = case
        if fingerprint:
            self._fingerprints[fingerprint] = case.case_id

    def add_cases(
        self,
        entries: list[
            tuple[CaseDetail, str | None]
            | tuple[CaseDetail, str | None, EvidencePackage | None]
        ],
    ) -> None:
        """批量提交已经完整构建的案件。

        所有校验和业务分析必须在调用前完成，使内存写入阶段不再发生业务失败。
        """

        for entry in entries:
            case, fingerprint, *optional = entry
            evidence_package = optional[0] if optional else None
            self.add_case(case, fingerprint, evidence_package)

    def find_by_fingerprint(self, fingerprint: str) -> CaseDetail | None:
        """根据接入指纹查找已存在案件。"""

        case_id = self._fingerprints.get(fingerprint)
        return self._cases.get(case_id) if case_id else None

    def list_cases(self, reviewed_case_ids: set[str] | None = None) -> list[CaseSummary]:
        """获取所有案件的审核队列摘要列表。"""
        reviewed_case_ids = reviewed_case_ids or set()
        return [
            CaseSummary(
                case_id=c.case_id,
                case_title=c.case_title,
                case_type=c.case_type,
                risk_level=c.risk_level,
                risk_score=c.risk_score,
                review_status="reviewed" if c.case_id in reviewed_case_ids else "pending",
                rule_signal_count=self._rule_signal_count(c),
                claim_amount=self._claim_amount(c),
            )
            for c in self._cases.values()
        ]

    def get_case(self, case_id: str) -> CaseDetail | None:
        """根据 case_id 获取案件详情，不存在时返回 None。"""
        return self._cases.get(case_id)

    @property
    def case_count(self) -> int:
        return len(self._cases)

    @staticmethod
    def _rule_signal_count(case: CaseDetail) -> int:
        return len([rule for rule in case.rule_hits if rule.hit])

    @staticmethod
    def _claim_amount(case: CaseDetail) -> float | None:
        value = case.source_record.get("ALL_SUM") or case.input_features.get("ALL_SUM")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
