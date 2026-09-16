"""Deterministic shortcuts for product-level Case Agent questions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentContentBlock,
    CaseAgentDisplayMode,
)


ShortcutKind = Literal[
    "greeting",
    "capability_overview",
    "session_help",
    "capability_status",
    "assistant_boundary",
    "recent_answer_count",
]


@dataclass(frozen=True, slots=True)
class CaseAgentShortcut:
    """A deterministic route for product/system-state questions."""

    kind: ShortcutKind
    intent: str
    display_mode: CaseAgentDisplayMode
    label: str = ""
    capability: str = ""
    fallback_notice: str = ""

    def to_state(self) -> dict[str, str]:
        return asdict(self)


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "你好",
    "您好",
    "在吗",
    "你在吗",
    "早上好",
    "上午好",
    "下午好",
    "晚上好",
}
_CAPABILITY_TERMS = (
    "你能做什么",
    "你可以做什么",
    "你有什么功能",
    "你有哪些功能",
    "案件助手有什么功能",
    "你是谁",
    "功能说明",
    "能力说明",
    "使用说明",
)
_SESSION_TERMS = (
    "新建会话",
    "历史会话",
    "删除会话",
    "恢复会话",
    "重命名",
    "会话隔离",
    "跨案件",
    "共享吗",
    "切换案件",
    "关掉系统",
    "重新打开",
    "会话会不会丢",
    "记忆会不会丢",
)
_STATUS_MARKERS = (
    "能查吗",
    "能不能查",
    "现在能查",
    "能用吗",
    "能不能用",
    "现在能用",
    "可以用吗",
    "可用吗",
    "支持吗",
    "接入了吗",
    "接了吗",
    "有没有接入",
    "是否接入",
    "能力状态",
    "能判断吗",
)
_BOUNDARY_TERMS = (
    "直接给审核结论",
    "给最终结论",
    "替我处理案件",
    "自动处理",
    "修改规则结果",
    "改变业务状态",
    "直接通过",
    "自动通过",
    "能不能直接处理",
)
_BUSINESS_CONTEXT_TERMS = (
    "当前案件有哪些",
    "这个案件",
    "规则命中",
    "为什么命中",
    "风险提示",
    "Evidence",
    "evidence",
    "工作笔记内容",
    "OP-",
    "诊断下",
    "处方",
    "就诊记录",
    "费用",
)


def detect_shortcut(text: str) -> CaseAgentShortcut | None:
    """Return a shortcut only for high-confidence product/system questions."""

    normalized = _normalize(text)
    if not normalized:
        return None

    if _is_greeting(normalized):
        return CaseAgentShortcut(
            kind="greeting",
            intent="general_help",
            display_mode="plain",
        )

    if _has_any(normalized, _BOUNDARY_TERMS):
        return CaseAgentShortcut(
            kind="assistant_boundary",
            intent="general_help",
            display_mode="plain",
        )

    if _has_business_context(text, normalized):
        return None

    if _has_any(normalized, _SESSION_TERMS):
        return CaseAgentShortcut(
            kind="session_help",
            intent="general_help",
            display_mode="plain",
        )

    capability = _capability_status(normalized)
    if capability is not None:
        return capability

    if _has_any(normalized, _CAPABILITY_TERMS):
        return CaseAgentShortcut(
            kind="capability_overview",
            intent="general_help",
            display_mode="plain",
        )

    return None


def build_shortcut_answer(shortcut: CaseAgentShortcut) -> CaseAgentAnswer:
    """Build a validated answer payload without calling the LLM."""

    if shortcut.kind == "greeting":
        text = (
            "你好，当前案件助手已就绪。你可以查询当前案件事实、规则依据、"
            "Evidence 分析和审核工作笔记，也可以询问会话的新建、历史、重命名、删除和隔离规则。"
        )
    elif shortcut.kind == "session_help":
        text = (
            "会话按当前案件、当前审核人员和当前会话隔离；新建会话会开启独立对话，"
            "历史会话可切换查看，标题支持重命名，删除后可通过恢复入口撤回归档。"
        )
    elif shortcut.kind == "capability_status":
        text = (
            f"当前{shortcut.label}尚未接入，暂不能提供对应查询结果。"
            "你仍可以查询当前案件事实、规则依据、Evidence 分析和审核工作笔记。"
        )
    elif shortcut.kind == "assistant_boundary":
        text = (
            "本助手仅基于当前可用数据提供查询和辅助分析，不形成最终业务处置结论；"
            "系统仅完成辅助分析，不改变业务状态，审核结论以人工处理结果为准。"
        )
    elif shortcut.kind == "recent_answer_count":
        text = shortcut.label or "上一轮回答中没有可统计的条目。"
    else:
        text = (
            "我可以协助查询当前案件事实、规则依据、已有 Evidence 分析和审核工作笔记；"
            "也可以说明会话的新建、历史、重命名、删除和隔离规则。"
            "涉及政策库、用药/诊疗知识库、同类口径库的能力以当前接入状态为准。"
        )

    return CaseAgentAnswer(
        display_mode=shortcut.display_mode,
        content_blocks=[CaseAgentContentBlock(text=text, source_refs=[])],
        sources=[],
        fallback_notice=shortcut.fallback_notice,
        metadata={
            "intent": shortcut.intent,
            "requires_citation": False,
            "capabilities_used": [],
            "shortcut": True,
            "shortcut_kind": shortcut.kind,
            "capability": shortcut.capability,
        },
    )


def detect_recent_answer_count_shortcut(
    text: str,
    recent_messages: Any,
) -> CaseAgentShortcut | None:
    """Answer simple count follow-ups from the latest assistant answer payload."""

    normalized = _normalize(text)
    if not normalized or not _looks_like_count_followup(normalized):
        return None
    latest_answer = _latest_answer_payload(recent_messages)
    if latest_answer is None:
        return None

    answer_text = "\n".join(
        str(getattr(block, "text", "") or "")
        for block in list(getattr(latest_answer, "content_blocks", []) or [])
        if str(getattr(block, "text", "") or "").strip()
    )
    count, target_label, items = _extract_recent_count(answer_text)
    if count <= 0:
        return None
    details = f"，分别是：{'；'.join(items[:5])}" if items else ""
    if len(items) > 5:
        details += "等"
    return CaseAgentShortcut(
        kind="recent_answer_count",
        intent="general_help",
        display_mode="plain",
        label=f"上一轮回答中的{target_label}一共有 {count} 项{details}。",
    )


def _capability_status(normalized: str) -> CaseAgentShortcut | None:
    if not _has_any(normalized, _STATUS_MARKERS):
        return None
    if _has_any(normalized, ("政策库", "政策知识库", "医保目录", "政策rag")):
        return CaseAgentShortcut(
            kind="capability_status",
            intent="domain_expert_query",
            display_mode="unavailable",
            label="政策知识库",
            capability="policy_knowledge",
            fallback_notice="政策知识库能力未接入。",
        )
    if _has_any(normalized, ("用药库", "用药", "诊疗", "合理性", "药品知识库")):
        return CaseAgentShortcut(
            kind="capability_status",
            intent="domain_expert_query",
            display_mode="unavailable",
            label="用药/诊疗合理性知识库",
            capability="drug_rationality",
            fallback_notice="用药/诊疗合理性知识库能力未接入。",
        )
    if _has_any(normalized, ("同类口径", "同类案件", "历史口径", "口径库")):
        return CaseAgentShortcut(
            kind="capability_status",
            intent="domain_expert_query",
            display_mode="unavailable",
            label="人工确认同类口径库",
            capability="verified_precedents",
            fallback_notice="人工确认同类口径库能力未接入。",
        )
    return None


def _is_greeting(normalized: str) -> bool:
    if normalized in _GREETINGS:
        return True
    return len(normalized) <= 12 and any(item in normalized for item in _GREETINGS)


def _has_business_context(original: str, normalized: str) -> bool:
    return any(term in original or _normalize(term) in normalized for term in _BUSINESS_CONTEXT_TERMS)


def _has_any(normalized: str, terms: tuple[str, ...]) -> bool:
    return any(_normalize(term) in normalized for term in terms)


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：\"'`]+", "", text).lower()


def _looks_like_count_followup(normalized: str) -> bool:
    count_markers = ("几项", "几条", "多少项", "多少条", "一共有几", "共有几", "总共有几")
    target_markers = (
        "审核建议",
        "建议",
        "核验事项",
        "核验任务",
        "风险线索",
        "异常风险线索",
        "线索",
        "事项",
        "上面",
        "刚才",
        "上一轮",
        "这些",
    )
    return any(marker in normalized for marker in count_markers) and any(
        marker in normalized for marker in target_markers
    )


def _latest_answer_payload(recent_messages: Any) -> Any | None:
    for message in reversed(list(recent_messages or [])):
        if getattr(message, "role", "") != "assistant":
            continue
        payload = getattr(message, "answer_payload", None)
        if payload is not None:
            return payload
    return None


def _extract_recent_count(answer_text: str) -> tuple[int, str, list[str]]:
    patterns = (
        (r"命中\s*(\d+)\s*条需关注的异常风险线索[：:，,]?\s*([^。\n]*)", "审核建议"),
        (r"(\d+)\s*条需关注的异常风险线索[：:，,]?\s*([^。\n]*)", "审核建议"),
        (r"(\d+)\s*项(?:审核建议|核验建议|人工核验参考|核验事项)[：:，,]?\s*([^。\n]*)", "审核建议"),
        (r"(?:审核建议|核验建议|人工核验参考|核验事项).*?(\d+)\s*项[：:，,]?\s*([^。\n]*)", "审核建议"),
    )
    for pattern, label in patterns:
        match = re.search(pattern, answer_text)
        if not match:
            continue
        count = int(match.group(1))
        items = _split_count_items(match.group(2) if match.lastindex and match.lastindex >= 2 else "")
        return count, label, items

    lines = [
        line.strip(" -\t")
        for line in answer_text.splitlines()
        if line.strip(" -\t")
    ]
    enumerated = [
        re.sub(r"^\d+[.、)\s]+", "", line).strip()
        for line in lines
        if re.match(r"^\d+[.、)\s]+", line)
    ]
    if enumerated:
        return len(enumerated), "条目", enumerated
    return 0, "条目", []


def _split_count_items(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    parts = [
        part.strip(" ；;、，,。.")
        for part in re.split(r"[；;、，,]", cleaned)
    ]
    return [part for part in parts if part]
