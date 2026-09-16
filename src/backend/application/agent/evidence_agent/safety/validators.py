"""Agent 输出本地安全、引文和数值验证。

对 Agent 生成的证据分析内容进行硬性安全校验，
包括禁止字段、裁决性结论、引文有效性和数值一致性。
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from src.backend.domain.agent.entities import (
    AgentCitation,
    AgentGeneratedContent,
    EvidenceLedgerItem,
    ReviewAdvisoryGeneratedContentV2,
)


class AgentOutputValidationError(ValueError):
    """Agent 输出验证失败时抛出的安全异常。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


PROHIBITED_CONCLUSIONS = (
    "已认定欺诈",
    "认定欺诈",
    "欺诈认定",
    "构成欺诈",
    "确认欺诈",
    "确定欺诈",
    "欺诈成立",
    "骗保成立",
    "已认定骗保",
    "认定骗保",
    "骗保认定",
    "构成骗保",
    "拒赔",
    "拒付",
    "不予支付",
    "给予处罚",
    "建议处罚",
    "应予处罚",
    "审核通过",
    "通过审核",
    "建议驳回",
    "自动通过",
    "自动拒绝",
    "认定违规",
    "确定违规",
)
SENSITIVE_MARKERS = ("API_KEY", "reasoning_content", "身份证", "真实姓名")
SECRET_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{4,}", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9_-])")


def validate_generated_content(
    content: AgentGeneratedContent | ReviewAdvisoryGeneratedContentV2,
    ledger: dict[str, EvidenceLedgerItem],
) -> list[AgentCitation]:
    """严格验证 Agent 输出，无法关联到当前证据账本时闭合失败。

    Args:
        content: Agent 生成的分析内容。
        ledger: 当前可用的证据账本。

    Returns:
        验证通过的引文列表。

    Raises:
        AgentOutputValidationError: 检测到敏感内容、裁决性结论或引文无效。
    """

    serialized = content.model_dump_json()
    # 禁止的敏感标记检查
    for marker in SENSITIVE_MARKERS:
        if marker.lower() in serialized.lower():
            raise AgentOutputValidationError("sensitive_output", "Agent 输出包含禁止的敏感标记")
    if SECRET_KEY_PATTERN.search(serialized):
        raise AgentOutputValidationError("sensitive_output", "Agent 输出包含禁止的敏感标记")
    # RES 标签检查
    if re.search(r"(?<![A-Za-z0-9_])RES(?![A-Za-z0-9_])", serialized, re.IGNORECASE):
        raise AgentOutputValidationError("sensitive_output", "Agent 输出包含禁止的标签字段")
    # 裁决性结论检查
    for phrase in PROHIBITED_CONCLUSIONS:
        if phrase in serialized:
            raise AgentOutputValidationError("prohibited_conclusion", "Agent 输出包含裁决性或定性结论")

    # 引文验证
    referenced: list[str] = []
    if isinstance(content, ReviewAdvisoryGeneratedContentV2):
        _validate_evidence_review_consistency(content)
        _validate_text_refs(
            content.evidence_review.summary,
            content.evidence_review.source_refs,
            ledger,
        )
        referenced.extend(content.evidence_review.source_refs)
        for item in content.clue_reviews:
            _validate_text_refs(f"{item.title} {item.explanation}", item.source_refs, ledger)
            referenced.extend(item.source_refs)
        for item in content.verification_checklist:
            _validate_text_refs(f"{item.title} {item.action} {item.rationale}", item.source_refs, ledger)
            referenced.extend(item.source_refs)
        for item in [*content.conflicts, *content.missing_information]:
            _validate_text_refs(item.statement, item.source_refs, ledger)
            referenced.extend(item.source_refs)
    else:
        if content.ai_risk_label_source_refs:
            _validate_text_refs(content.ai_risk_label, content.ai_risk_label_source_refs, ledger)
            referenced.extend(content.ai_risk_label_source_refs)
        if content.risk_judgement_source_refs:
            _validate_text_refs(content.risk_judgement, content.risk_judgement_source_refs, ledger)
            referenced.extend(content.risk_judgement_source_refs)
        if content.evidence_strength_source_refs:
            _validate_text_refs(content.evidence_strength, content.evidence_strength_source_refs, ledger)
            referenced.extend(content.evidence_strength_source_refs)
        for item in content.key_risk_signals:
            _validate_text_refs(item.statement, item.source_refs, ledger)
            referenced.extend(item.source_refs)
        for item in content.human_review_focus:
            _validate_text_refs(f"{item.title} {item.action} {item.rationale}", item.source_refs, ledger)
            referenced.extend(item.source_refs)
        _validate_text_refs(content.risk_overview, content.risk_overview_source_refs, ledger)
        referenced.extend(content.risk_overview_source_refs)
        for item in [*content.supporting_evidence, *content.conflicts]:
            _validate_text_refs(item.statement, item.source_refs, ledger)
            referenced.extend(item.source_refs)
        if content.signal_review.case_review_hint:
            _validate_text_refs(
                content.signal_review.case_review_hint,
                content.signal_review.case_review_hint_source_refs,
                ledger,
            )
            referenced.extend(content.signal_review.case_review_hint_source_refs)
        for item in [
            *content.signal_review.supported_clues,
            *content.signal_review.needs_review,
            *content.signal_review.supplementary_review_hints,
        ]:
            _validate_text_refs(item.statement, item.source_refs, ledger)
            referenced.extend(item.source_refs)
        for item in content.verification_checklist:
            _validate_text_refs(f"{item.title} {item.action} {item.rationale}", item.source_refs, ledger)
            referenced.extend(item.source_refs)

    # 构建引文
    citations: list[AgentCitation] = []
    for ref in dict.fromkeys(referenced):
        source = ledger[ref]
        citations.append(
            AgentCitation(
                citation_id=source.ledger_ref,
                source_type=source.source_type,
                source_ref=source.source_ref,
                label=source.label,
                version=source.version,
                current_value=source.current_value,
                threshold=source.threshold,
                metadata=source.metadata,
            )
        )
    return citations


def validate_generated_content_relaxed(
    content: AgentGeneratedContent | ReviewAdvisoryGeneratedContentV2,
    ledger: dict[str, EvidenceLedgerItem],
) -> list[AgentCitation]:
    """宽松验证 Agent 输出，暂时跳过严格引文和数值检查，仅保留硬性停止项。

    Args:
        content: Agent 生成的分析内容。
        ledger: 当前可用的证据账本。

    Returns:
        验证通过的引文列表。

    Raises:
        AgentOutputValidationError: 检测到敏感内容或裁决性结论。
    """

    serialized = content.model_dump_json()
    for marker in SENSITIVE_MARKERS:
        if marker.lower() in serialized.lower():
            raise AgentOutputValidationError("sensitive_output", "Agent output contains a forbidden sensitive marker")
    if SECRET_KEY_PATTERN.search(serialized):
        raise AgentOutputValidationError("sensitive_output", "Agent output contains a forbidden sensitive marker")
    if re.search(r"(?<![A-Za-z0-9_])RES(?![A-Za-z0-9_])", serialized, re.IGNORECASE):
        raise AgentOutputValidationError("sensitive_output", "Agent output contains the forbidden RES label field")
    for phrase in PROHIBITED_CONCLUSIONS:
        if phrase in serialized:
            raise AgentOutputValidationError("prohibited_conclusion", "Agent output contains a prohibited decision-like conclusion")

    # 收集引用（不进行严格验证）
    referenced: list[str] = []
    if isinstance(content, ReviewAdvisoryGeneratedContentV2):
        referenced.extend(content.evidence_review.source_refs)
        for item in content.clue_reviews:
            referenced.extend(item.source_refs)
        for item in content.verification_checklist:
            referenced.extend(item.source_refs)
        for item in [*content.conflicts, *content.missing_information]:
            referenced.extend(item.source_refs)
    else:
        referenced.extend(content.ai_risk_label_source_refs)
        referenced.extend(content.risk_judgement_source_refs)
        referenced.extend(content.evidence_strength_source_refs)
        for item in content.key_risk_signals:
            referenced.extend(item.source_refs)
        for item in content.human_review_focus:
            referenced.extend(item.source_refs)
        referenced.extend(content.risk_overview_source_refs)
        for item in [*content.supporting_evidence, *content.conflicts]:
            referenced.extend(item.source_refs)
        referenced.extend(content.signal_review.case_review_hint_source_refs)
        for item in [
            *content.signal_review.supported_clues,
            *content.signal_review.needs_review,
            *content.signal_review.supplementary_review_hints,
        ]:
            referenced.extend(item.source_refs)
        for item in content.verification_checklist:
            referenced.extend(item.source_refs)

    citations: list[AgentCitation] = []
    for ref in dict.fromkeys(referenced):
        source = ledger.get(ref)
        if source is None:
            continue
        citations.append(
            AgentCitation(
                citation_id=source.ledger_ref,
                source_type=source.source_type,
                source_ref=source.source_ref,
                label=source.label,
                version=source.version,
                current_value=source.current_value,
                threshold=source.threshold,
                metadata=source.metadata,
            )
        )
    return citations


def _validate_evidence_review_consistency(
    content: ReviewAdvisoryGeneratedContentV2,
) -> None:
    """阻止复核关系和结论支撑度出现相互矛盾的组合。"""

    expected = {
        "supports": "高",
        "partially_supports": "中",
        "weakly_supports": "低",
        "insufficient_evidence": "证据不足",
        "inconsistent": "低",
    }[content.evidence_review.relation]
    if content.evidence_review.support_level != expected:
        raise AgentOutputValidationError(
            "evidence_review_inconsistent",
            "证据复核关系与结论支撑度不一致",
        )


def _validate_text_refs(
    text: str,
    refs: Iterable[str],
    ledger: dict[str, EvidenceLedgerItem],
) -> None:
    """验证文本中引用的有效性和数值一致性。

    Args:
        text: 待验证的文本内容。
        refs: 引用的证据项 ID 列表。
        ledger: 证据账本。

    Raises:
        AgentOutputValidationError: 引用缺失、无效或数值不匹配。
    """

    refs = list(dict.fromkeys(refs))
    if not refs:
        raise AgentOutputValidationError("citation_missing", "事实性内容缺少证据引用")
    unknown = [ref for ref in refs if ref not in ledger]
    if unknown:
        raise AgentOutputValidationError("citation_invalid", "引用不属于本次案件证据账本")
    cited_text = json.dumps(
        [ledger[ref].value for ref in refs],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).replace(",", "")
    cited_numbers = [
        float(token.replace(",", "").rstrip("%"))
        for token in NUMBER_PATTERN.findall(cited_text)
    ]
    for token in NUMBER_PATTERN.findall(text):
        normalized = token.replace(",", "")
        value = float(normalized.rstrip("%"))
        candidates = [value, value / 100] if normalized.endswith("%") else [value]
        if not any(
            abs(candidate - cited) <= max(1e-9, abs(cited) * 1e-6)
            for candidate in candidates
            for cited in cited_numbers
        ):
            raise AgentOutputValidationError("numeric_mismatch", f"数值 {token} 无法由所引证据核对")
