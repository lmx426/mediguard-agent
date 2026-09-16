"""Case Agent final-answer prompt builder."""

from __future__ import annotations

import json

from .output_contract import OUTPUT_CONTRACT


ANSWER_EXAMPLES = {
    "plain": {
        "display_mode": "plain",
        "content_blocks": [
            {
                "text": (
                    "当前案件助手可以帮助查询当前案件事实、规则依据、已有 Evidence 分析和审核工作笔记；"
                    "也可以说明新建会话、历史会话与会话隔离方式。本助手仅基于当前可用数据提供查询和辅助分析，"
                    "不形成最终业务处置结论。"
                ),
                "source_refs": [],
            }
        ],
        "sources": [],
        "fallback_notice": "",
        "metadata": {
            "intent": "general_help",
            "requires_citation": False,
            "capabilities_used": [],
        },
    },
    "unavailable": {
        "display_mode": "unavailable",
        "content_blocks": [
            {
                "text": (
                    "当前未检索到足够可引用的政策证据回答该问题。"
                    "本助手不会使用通用知识补充政策事实。"
                ),
                "source_refs": [],
            }
        ],
        "sources": [],
        "fallback_notice": "政策证据不足，未形成可追溯政策口径。",
        "metadata": {
            "intent": "policy_expert_query",
            "requires_citation": False,
            "capabilities_used": ["ask_policy_expert"],
        },
    },
    "grounded": {
        "display_mode": "grounded",
        "content_blocks": [
            {
                "text": "当前案件摘要显示该案件为门诊类型，申报摘要提示需结合规则结果继续核验。",
                "source_refs": ["case:CASE-001:snapshot"],
            }
        ],
        "sources": [
            {
                "source_ref": "case:CASE-001:snapshot",
                "source_type": "case",
                "title": "案件摘要",
                "version": None,
                "detail": {
                    "fields": [
                        {"label": "案件类型", "value": "门诊"},
                        {"label": "申报摘要", "value": "脱敏统计记录提示需要人工核验。"},
                    ]
                },
                "metadata": {},
            }
        ],
        "fallback_notice": "",
        "metadata": {
            "intent": "case_basic_info_query",
            "requires_citation": True,
            "capabilities_used": ["query_case_basic_info"],
        },
    },
    "error": {
        "display_mode": "error",
        "content_blocks": [
            {
                "text": "当前未形成可采信的案件助手回答。",
                "source_refs": [],
            }
        ],
        "sources": [],
        "fallback_notice": "当前回答暂不可用，请稍后重试或查看当前案件已有资料。",
        "metadata": {
            "intent": "unknown",
            "requires_citation": False,
            "capabilities_used": [],
        },
    },
}


def build_final_answer_prompt(
    display_mode: str | None = None,
    *,
    fast_mode_enabled: bool = False,
    draft_only: bool = False,
) -> str:
    """Build the final controlled JSON prompt."""

    if draft_only:
        source_rule = (
            "每个 source_refs 只能使用 allowed_source_refs 中的值，且事实块必须至少引用一个来源。"
            if display_mode == "grounded"
            else "所有 source_refs 必须为空数组。"
        )
        return (
            "请只根据受控回答上下文生成简洁回答草稿。"
            "只返回一个 JSON 对象，顶层只允许 content_blocks 和 fallback_notice。"
            "content_blocks 是 1 到 4 个对象，每个对象只允许 text 和 source_refs；"
            "text 每块不超过 600 个中文字符，source_refs 是字符串数组。"
            f"{source_rule}"
            "不得生成 sources、claims、citations、metadata、display_mode 或额外字段；"
            "这些治理字段由后端确定性组装。不得补充受控上下文以外的事实。"
        )

    if fast_mode_enabled:
        mode = display_mode if display_mode in ANSWER_EXAMPLES else "plain"
        examples = {mode: ANSWER_EXAMPLES[mode]}
        mode_rules = _fast_mode_rules(mode)
    else:
        examples = ANSWER_EXAMPLES
        mode_rules = _legacy_mode_rules()

    return (
        "请基于上方受控上下文回答，并输出合法 JSON 对象。"
        "字段契约如下："
        f"{json.dumps(OUTPUT_CONTRACT, ensure_ascii=False)}。"
        "仅允许顶层字段 display_mode, content_blocks, sources, answer_markdown, claims, citations, fallback_notice, metadata；"
        "禁止输出旧字段或额外字段 brief_answer, references, summary, suggestions。"
        "必须绝对遵循以下 TypeScript 结构："
        "interface ExpectedResponse {"
        'display_mode: "grounded" | "plain" | "unavailable" | "error";'
        "content_blocks: Array<{ text: string; source_refs: string[]; }>; "
        "sources: Array<{ source_ref: string; source_type: string; title: string; "
        "version: string | null; detail: { fields: Array<{ label: string; value: string; }>; }; "
        "metadata: Record<string, unknown>; }>; "
        "answer_markdown: string | null; "
        "claims: Array<{ claim_id: string; need_id: string; need_text: string; text: string; "
        "fact_refs: string[]; source_refs: string[]; citation_ids: string[]; support_status: string; }>; "
        "citations: Array<{ citation_id: string; label: number; claim_id: string | null; "
        "fact_refs: string[]; evidence_refs: string[]; source_refs: string[]; }>; "
        "fallback_notice: string; "
        "metadata: { intent: string; requires_citation: boolean; capabilities_used: string[]; "
        "[key: string]: unknown; };"
        "}"
        "如输出 answer_markdown，正文可为一个自然段，每个事实句后用 [1]、[2] 绑定 citations；"
        "不要把同一句没有支撑的内容挂到引用角标上。"
        "根据问题复杂度决定 content_blocks 数量：单一事实查询只写 1 个 block；"
        "普通功能说明写 1 到 2 个 block；案件综合分析写 2 到 4 个 block；"
        "复杂多来源综述最多 4 个 block。不要为了凑数量主动展开用户未询问的内容。"
        "回答不要机械罗列所有字段；先判断用户真正要的是一个字段、一个清单、一个材料详情还是风险解释。"
        "如果 query_semantics.user_goal 是单一字段查询，例如只问性别、审核状态、案件类型、过敏史、参保类型，"
        "正文只能回答该字段和值；不要顺带输出同一 section 的其他字段。"
        "回答正文不要直接复述 source_ref、内部 ID、payload_hash 或技术来源标识；"
        "引用只放在 source_refs 数组中，前端会显示业务简称。"
        "模式规则：plain 的 sources 必须为 []，所有 source_refs 必须为 []，fallback_notice 必须为空字符串；"
        "grounded 的 sources 必须从 available_sources 中复制完整对象，不能只写 source_ref，"
        "source_refs 只能引用 sources 中存在的 source_ref；"
        "unavailable 的 sources 必须为 []，所有 source_refs 必须为 []，"
        "fallback_notice 填写对应能力未接入、数据未生成或不可用的简短原因，不得用通用知识补答；"
        "error 的 sources 必须为 []，所有 source_refs 必须为 []，"
        "fallback_notice 填写友好降级说明，不暴露后端异常、校验失败细节或内部错误原文。"
        "回答正文和 fallback_notice 必须使用系统提示中的标准业务表达，"
        "不要主动展开处置性、责任定性或审批结果式边界词；正常回答不要把边界说明塞进 fallback_notice。"
        "普通 grounded 案件事实查询不要主动追加“本助手仅...”这类边界说明；"
        "只有用户询问助手边界、能力未接入、错误降级或要求业务处置结论时，才使用边界说明。"
        "If controlled context contains answer_policy, strictly follow its display_mode, requires_citation, "
        "source_policy, capability_policy, no_general_knowledge_fallback, capabilities_used, "
        "unavailable_capabilities and not_ready_sections; do not rewrite the answer mode or source policy. "
        "如果受控上下文包含 answer_context，优先使用 answer_context 回答，"
        "不要绕回 capability_results 做数据库字段式复述；answer_context 已经是后端按本问题裁剪后的事实包。"
        "如果 answer_context.analysis_inputs 中包含“政策专家分析”，可以重述或压缩 expert_answer，"
        "但不得改写其中的政策事实、policy_evidence、answerability_summary 或 limits；"
        "不得把 Policy Expert 明确排除的材料缺口差集补成最终回答。"
        "如果 answer_context.analysis_inputs 中包含 mode=reuse_previous_answer 或 section=上一轮回答，"
        "本轮只能基于 previous_answer 做解释、举例、简化、翻译、总结、列表化或换句话表达；"
        "不得新增政策事实、案件事实、药品目录事实或审核结论；"
        "sources 必须只从 available_sources 复制，content_blocks.source_refs 只能使用 previous_source_refs 中的值；"
        "不得因为 execution_plan 为空就改成能力未接入或证据不足。"
        "如果受控上下文包含 answer_style_policy，必须遵循其中的 answer_tone、max_blocks、"
        "max_sentences_per_block、include_interpretation、include_next_focus、forbid_extra_fields 和 business_expression。"
        "你不是数据库结果展示器；应把 facts、lists、details、analysis_inputs 转成审核人员容易阅读的业务表达。"
        "single_field 或 forbid_extra_fields=true 时，只回答用户请求的字段和值，正文不要出现其他字段。"
        "list 时优先完整列出用户要求的清单或明细，不要只写“共有若干项”的泛泛摘要。"
        "analysis 时可以解释当前事实的审核含义，但不得引入受控上下文之外的医学、政策或处置知识。"
        f"{mode_rules}"
        "格式要求：直接输出 JSON 对象，不输出解释性文字。"
        "以下是合法输出示例，必须模仿结构而不是照抄来源内容："
        f"{json.dumps(examples, ensure_ascii=False)}。"
    )


def _legacy_mode_rules() -> str:
    return (
        "Caser L2/L3 能力在 v1 返回 unavailable 时，"
        "必须输出 display_mode=unavailable，并按 answer_policy.capability_policy 区分原因："
        "expert_evidence_insufficient 表示专家能力已运行但可引用政策证据不足，只能说明证据不足；"
        "capability_unavailable 才说明对应能力未接入；不得用通用知识补答。"
        "general_help 必须输出 display_mode=plain，只说明用户问到的助手能力、会话操作或功能解释；"
        "不要生成来源引用，fallback_notice 必须为空字符串。"
        "unavailable/error 只给审核人员可理解的友好原因，不暴露后端异常、校验失败细节或内部错误原文。"
    )


def _fast_mode_rules(display_mode: str) -> str:
    if display_mode == "plain":
        return (
            "本轮是普通说明回答：输出 display_mode=plain；只写 1 到 2 个 content_blocks；"
            "总字数建议 120 到 220 个中文字符；不要查案件、不要引用来源。"
        )
    if display_mode == "unavailable":
        return (
            "本轮是不可回答降级：输出 display_mode=unavailable；只写 1 到 2 个 content_blocks；"
            "总字数建议 80 到 160 个中文字符；必须按 answer_policy.capability_policy 区分原因："
            "expert_evidence_insufficient 只能说明政策证据不足，不能说能力未接入；"
            "capability_unavailable 才能说明对应能力尚未接入；"
            "不得使用通用知识补答，不要引用来源。"
        )
    if display_mode == "grounded":
        return (
            "本轮是证据型案件回答：优先使用 context_digest，而不是展开完整工具结果；"
            "只能引用 digest_source_refs 和 available_sources 中存在的 source_ref；"
            "单一事实查询只写 1 个 content_block；多字段查询或规则解释写 2 到 3 个 content_blocks；"
            "综合分析最多 4 个 content_blocks；每个 block 1 到 2 句；"
            "只回答用户问到的部分，不主动输出完整长报告。"
        )
    return (
        "本轮是错误或降级回答：输出 display_mode=error；只给友好原因，"
        "不要暴露内部异常、校验失败细节或后端错误原文。"
    )
