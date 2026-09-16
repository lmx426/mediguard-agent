"""Prompt for Caser business semantic planning."""

from __future__ import annotations

import json

from src.backend.application.agent.case_agent.tools.registry import (
    capability_manifest_for_prompt,
)


ALLOWED_PLANNER_INTENTS = (
    "general_help",
    "case_task",
    "expert_task",
    "clarification_required",
)
ALLOWED_GRANULARITIES = (
    "single_field",
    "field_group",
    "list",
    "detail",
    "overview",
    "analysis",
)

PLAN_CONTRACT = {
    "type": "object",
    "required": ["query_semantics", "execution_plan"],
    "additionalProperties": False,
    "properties": {
        "query_semantics": {
            "type": "object",
            "required": ["intent", "user_goal", "granularity", "information_needs", "missing_slots"],
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": list(ALLOWED_PLANNER_INTENTS)},
                "user_goal": {"type": "string"},
                "granularity": {"type": "string", "enum": list(ALLOWED_GRANULARITIES)},
                "information_needs": {"type": "array", "items": {"type": "string"}},
                "missing_slots": {"type": "array", "items": {"type": "string"}},
            },
        },
        "execution_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step", "layer", "capability", "arguments", "covers", "reason"],
                "additionalProperties": False,
                "properties": {
                    "step": {"type": "integer", "minimum": 1, "maximum": 8},
                    "layer": {"type": "string", "enum": ["L1", "L2", "L3"]},
                    "capability": {"type": "string"},
                    "arguments": {"type": "object"},
                    "depends_on": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                            ]
                        },
                    },
                    "covers": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

SEMANTIC_PARSE_CONTRACT = {
    "type": "object",
    "required": [
        "intent",
        "action",
        "answer_shape",
        "target_layer_hint",
        "target_objects",
        "information_needs",
        "rewritten_query",
        "filled_slots",
        "missing_slots",
        "confidence",
        "need_clarification",
        "safety_flags",
    ],
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "description": "general_help / case_task / expert_task / clarification_required",
        },
        "action": {
            "type": "string",
            "description": "query_fact / count_items / list_records / explain_reason / query_policy_basis / general_help",
        },
        "answer_shape": {
            "type": "string",
            "enum": list(ALLOWED_GRANULARITIES),
        },
        "target_layer_hint": {
            "type": "string",
            "enum": ["L1", "L2", "L3", "none"],
        },
        "target_objects": {"type": "array", "items": {"type": "string"}},
        "information_needs": {"type": "array", "items": {"type": "string"}},
        "rewritten_query": {"type": "string"},
        "filled_slots": {"type": "object"},
        "missing_slots": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "need_clarification": {"type": "boolean"},
        "safety_flags": {"type": "array", "items": {"type": "string"}},
    },
}


def build_semantic_parse_prompt(case_id: str) -> str:
    """Build the lightweight perception prompt for pure semantic parsing."""

    routing_hint = {
        "L1": "查询当前案件事实，例如就诊、材料、费用、结算、状态、申报人信息。",
        "L2": "基于案件事实做汇总、计数、解释、比对、风险原因或审核建议整理。",
        "L3": "查询政策依据、医保规定、报销口径、专家政策分析。",
        "none": "普通产品说明、寒暄、无业务目标或需要澄清时使用。",
    }
    return (
        "你是医保稽核案件助手 Caser 的感知层语义解析器。"
        "你的任务是把用户输入解析成结构化语义，不执行任何业务能力，不生成执行计划，不回答用户问题。"
        "必须只返回一个严格 JSON 对象，不输出 Markdown、解释文字或多余字段。"
        f"当前绑定案件 case_id 是 {case_id}，只允许理解当前案件范围内的问题。"
        "你需要完成：意图与动作分类、指代消解、Query 改写、实体与槽位抽取、安全前置质检。"
        "如果用户问题是“这个/它/为什么这么高/为什么会触发/一共有几项”等追问，"
        "必须优先结合 semantic_hint_pack.structured_task_state、turn_relation、conversation_focus "
        "和 active_reference_objects 补全 rewritten_query。"
        "structured_task_state 是当前结构化任务结论，recent_messages 只是补充证据；"
        "不得在用户未明确切换任务时静默改变 task_id 对应的核心目标。"
        "只有上下文仍无法定位对象时，才设置 need_clarification=true 并填写 missing_slots。"
        "target_layer_hint 只是能力倾向，不是执行计划。"
        "information_needs 只写回答该问题不可缺少的信息点，禁止把可能相关但未被询问的内容加入。"
        "严禁输出 execution_plan、capability、depends_on、SQL、数据库表名、文件路径、内部实体 ID 或敏感字段。"
        "不要引用政策原文、案件完整资料或证据内容；这些属于执行层和回答层。"
        "能力倾向说明如下："
        f"{json.dumps(routing_hint, ensure_ascii=False, sort_keys=True)}"
        "输出协议如下："
        f"{json.dumps(SEMANTIC_PARSE_CONTRACT, ensure_ascii=False, sort_keys=True)}"
        "示例："
        f"{json.dumps(_example_semantic_parse(), ensure_ascii=False, sort_keys=True)}"
    )


def build_business_semantic_planning_prompt(case_id: str) -> str:
    """Build a strict JSON planning prompt."""

    capabilities = capability_manifest_for_prompt()
    return (
        "你是医保稽核案件助手 Caser 的业务语义规划器。"
        "你只负责理解用户问题并生成后端执行计划，不直接回答用户问题。"
        "必须只返回一个严格 JSON 对象，不输出 Markdown、解释文字或多余字段。"
        f"当前绑定案件 case_id 是 {case_id}，所有 execution_plan[].arguments.case_id 都必须使用这个 case_id。"
        "禁止输出 SQL、数据库表名、文件路径、数据库路径、内部实体 ID 或敏感字段。"
        "允许的 intent 只有 general_help、case_task、expert_task、clarification_required。"
        "intent 只是粗分类；真正决定后端调用什么能力的是 execution_plan[].capability。"
        "必须判断 query_semantics.granularity："
        "single_field=只问一个字段或一个值，例如性别、审核状态、案件类型、过敏史；"
        "field_group=问同一对象的一组基础字段，例如申报人基础信息；"
        "list=问清单、有哪些、全部明细、来源分别是什么；"
        "detail=问某个材料、规则、线索的具体内容；"
        "overview=问整体情况或摘要；analysis=问为什么、是否偏高、怎么看、怎么核验等需要解释。"
        "execution_plan[].arguments.fields 可以写英文顶层字段名辅助后端裁剪；不确定时留空。"
        "普通产品说明、功能说明、会话说明、按钮说明、你好等问题使用 intent=general_help，execution_plan=[]。"
        "当前案件相关查询、材料查询、规则查询、风险评分、证据包、研判、核验事项等使用 intent=case_task，并给出一个或多个 L1/L2 执行步骤。"
        "recent_context.structured_task_state 是当前任务结论；recent_messages 仅用于指代和原因补充。"
        "澄清回复应继续 pending_clarification 对应任务，纠错应使用新值，不得继续依赖旧值。"
        "政策、用药诊疗合理性、同类口径、复杂专家分析等使用 intent=expert_task，并给出对应 L3 执行步骤；第一版这些能力可能不可用，但仍应规划到正确 capability。"
        "缺少关键槽位时使用 intent=clarification_required，把缺失项写入 missing_slots，并保持 execution_plan=[]。"
        "例如用户要求查看某份具体材料详情但没有说明材料名称、材料 ID 或材料分类时，应追问。"
        "capability 必须填写能力卡片中的 capability 字段。"
        "section_key 只是内部 View Store 分区，只用于理解和调试，禁止填入 execution_plan[].capability。"
        "每个 execution_plan 按真实执行顺序填写 step，从 1 开始递增。"
        "query_semantics.information_needs 必须列出回答所需信息点；每个步骤的 covers 只能填写该步骤实际覆盖的信息点。"
        "优先选择覆盖全部 information_needs 的最少能力集合，禁止为了补充背景调用不必要的宽泛能力。"
        "如果某个步骤必须读取前序步骤 artifact，再填写 depends_on，例如 [1] 或 ['step_1']；没有依赖时填写空数组。"
        "能力卡片如下："
        f"{json.dumps(capabilities, ensure_ascii=False, sort_keys=True)}"
        "输出协议如下："
        f"{json.dumps(PLAN_CONTRACT, ensure_ascii=False, sort_keys=True)}"
        "示例："
        f"{json.dumps(_example_plan(case_id), ensure_ascii=False, sort_keys=True)}"
    )


def _example_plan(case_id: str) -> dict[str, object]:
    return {
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "查询申报人性别",
            "granularity": "single_field",
            "information_needs": ["claimant_gender"],
            "missing_slots": [],
        },
        "execution_plan": [
            {
                "step": 1,
                "layer": "L1",
                "capability": "query_claimant_profile",
                "arguments": {"case_id": case_id},
                "depends_on": [],
                "covers": ["claimant_gender"],
                "reason": "申报人性别属于申报人基础信息。",
            }
        ],
    }


def _example_semantic_parse() -> dict[str, object]:
    return {
        "intent": "case_task",
        "action": "explain_reason",
        "answer_shape": "analysis",
        "target_layer_hint": "L2",
        "target_objects": ["risk_score"],
        "information_needs": ["risk_score"],
        "rewritten_query": "当前案件综合风险提示强度为什么是 69/100？",
        "filled_slots": {"business_module": "risk_score", "field": "overall_strength"},
        "missing_slots": [],
        "confidence": 0.86,
        "need_clarification": False,
        "safety_flags": [],
    }
