"""Safety helpers for Case Agent."""

from __future__ import annotations

import json
import re
from typing import Any

from src.backend.domain.case_agent.entities import CaseAgentAnswer, CaseAgentSource


class CaseAgentSafetyError(ValueError):
    """Raised when a Case Agent artifact violates hard safety boundaries."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


SENSITIVE_TERMS = ("api_key", "reasoning_content")
SENSITIVE_IDENTITY_PATTERNS = (
    re.compile(r"(?:身份证(?:件)?号码?|证件号码|证件号|公民身份号码|id_card|id_number)", re.IGNORECASE),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?:真实姓名|患者姓名|参保人姓名|姓名)\s*[：:=]\s*[\u4e00-\u9fffA-Za-z]{2,}"),
    re.compile(r"(?:住址|家庭住址|通讯地址|联系地址)\s*[：:=]\s*[^,，;；。\n]{4,}"),
)
DISPOSITION_TERMS = ("拒赔", "拒付", "不予支付", "处罚")
FINAL_STATUS_TERMS = ("自动通过", "自动拒绝", "审核通过")
FINAL_DETERMINATION_TERMS = (
    "欺诈认定",
    "认定欺诈",
    "认定骗保",
    "构成骗保",
    "骗保成立",
    "最终不合理",
    "不合理定性",
)


def _term_pattern(terms: tuple[str, ...]) -> str:
    return "(?:" + "|".join(re.escape(term) for term in terms) + ")"


_DECISION_TERM_PATTERN = _term_pattern(DISPOSITION_TERMS + FINAL_STATUS_TERMS)
_DETERMINATION_TERM_PATTERN = _term_pattern(
    FINAL_DETERMINATION_TERMS + ("欺诈", "骗保", "不合理", "违规", "通过")
)
_SAFE_BOUNDARY_PATTERNS = (
    re.compile(r"本助手.{0,80}(?:不形成|不提供|不能形成).{0,20}(?:结论|处置结论|业务处置结论)"),
    re.compile(r"(?:支付处理|审核|业务处置|后续处置|结论).{0,24}(?:需由|需要由|应由|以).{0,16}(?:审核人员|人工|人工流程).{0,16}(?:确认|为准)"),
    re.compile(r"系统仅完成辅助分析.{0,24}不改变业务状态"),
    re.compile(r"存在需关注的异常风险线索"),
    re.compile(r"存在合理性核验点.{0,24}需结合业务材料确认"),
    re.compile(r"(?:不能|不得|不应|不建议|不宜|不要|不会|无法|不可|不可以|不形成|不提供).{0,16}" + _DECISION_TERM_PATTERN),
    re.compile(r"(?:不能|不得|不应|不建议|不宜|不要|不会|无法|不可|不可以|不形成|不提供).{0,16}" + _term_pattern(FINAL_DETERMINATION_TERMS)),
)
_EXECUTIONAL_DECISION_RE = re.compile(
    r"(?:建议|应当|应该|可以|可直接|直接|系统自动|自动|予以|进行|执行|给予|作出).{0,12}"
    + _DECISION_TERM_PATTERN
)
_DETERMINATIVE_CONCLUSION_RE = re.compile(
    r"(?:认定|判定|确定|定性为|结论为|判断为).{0,12}" + _DETERMINATION_TERM_PATTERN
)
_STANDALONE_FINAL_CONCLUSION_RE = re.compile(
    _term_pattern(FINAL_DETERMINATION_TERMS + FINAL_STATUS_TERMS)
)
_FACTUAL_REVIEW_STATUS_RE = re.compile(
    r"(?:当前案件|当前记录|案件基础信息|当前案件基础信息|系统记录|已有记录)?"
    r".{0,16}(?:审核状态|办理状态|当前状态)"
    r".{0,8}"
    + _term_pattern(
        FINAL_STATUS_TERMS
        + (
            "已通过",
            "未通过",
            "待人工初审",
            "已完成人工初审",
            "pending",
            "reviewed",
            "approved",
            "rejected",
        )
    )
)


def assert_safe_text(text: str, *, allow_decision_request: bool = False) -> None:
    """Reject forbidden labels, identities, secrets and decision-like claims."""

    if re.search(r"(?<![A-Za-z0-9_])RES(?![A-Za-z0-9_])", text, re.IGNORECASE):
        raise CaseAgentSafetyError("sensitive_output", "内容包含禁止标签 RES")
    lowered = text.lower()
    for term in SENSITIVE_TERMS:
        if term.lower() in lowered:
            raise CaseAgentSafetyError("sensitive_output", "内容包含禁止身份或内部字段")
    for pattern in SENSITIVE_IDENTITY_PATTERNS:
        if pattern.search(text):
            raise CaseAgentSafetyError("sensitive_output", "内容包含禁止身份或内部字段")
    if not allow_decision_request:
        assert_no_prohibited_conclusion(text)


def assert_no_prohibited_conclusion(text: str) -> None:
    """Reject executional or determinative business conclusions."""

    normalized = re.sub(r"\s+", "", text)
    for pattern in _SAFE_BOUNDARY_PATTERNS:
        normalized = pattern.sub("", normalized)
    if _EXECUTIONAL_DECISION_RE.search(normalized) or _DETERMINATIVE_CONCLUSION_RE.search(normalized):
        raise CaseAgentSafetyError("prohibited_conclusion", "内容包含裁决性或越权结论")
    normalized = _FACTUAL_REVIEW_STATUS_RE.sub("", normalized)
    if _STANDALONE_FINAL_CONCLUSION_RE.search(normalized):
        raise CaseAgentSafetyError("prohibited_conclusion", "内容包含裁决性或越权结论")


def assert_safe_payload(value: Any, *, allow_decision_request: bool = False) -> None:
    """Serialize and validate a JSON-like payload."""

    assert_safe_text(
        json.dumps(value, ensure_ascii=False, default=str),
        allow_decision_request=allow_decision_request,
    )


def validate_answer(answer: CaseAgentAnswer, available_sources: list[CaseAgentSource]) -> None:
    """Validate final structured answer against available case-scoped refs."""

    payload = answer.model_dump(mode="json")
    is_general_help = (
        answer.display_mode == "plain"
        and answer.metadata.get("intent") == "general_help"
    )
    assert_safe_payload(payload, allow_decision_request=is_general_help)
    available_refs = {source.source_ref for source in available_sources}
    answer_source_refs = {source.source_ref for source in answer.sources}
    block_refs = {ref for block in answer.content_blocks for ref in block.source_refs}
    claim_refs = {ref for claim in answer.claims for ref in claim.source_refs}
    citation_refs = {ref for citation in answer.citations for ref in citation.source_refs}
    inline_refs = block_refs | claim_refs | citation_refs

    if answer.display_mode in {"plain", "unavailable", "error"}:
        if answer.sources or inline_refs or answer.claims or answer.citations:
            raise CaseAgentSafetyError("citation_invalid", "非证据型回答不应包含引用")
        return

    if answer.display_mode == "grounded" and (not answer.sources or not inline_refs):
        raise CaseAgentSafetyError("citation_missing_for_grounded", "证据型回答缺少来源")

    unknown = sorted((answer_source_refs | inline_refs) - available_refs)
    if unknown:
        raise CaseAgentSafetyError("citation_invalid", "回答引用不属于当前案件上下文")
    missing_from_answer_sources = sorted(inline_refs - answer_source_refs)
    if missing_from_answer_sources:
        raise CaseAgentSafetyError("citation_invalid", "行内引用未出现在 sources 列表")
