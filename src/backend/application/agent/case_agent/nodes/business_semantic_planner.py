"""Caser business semantic planner node."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.backend.application.agent.case_agent.prompts.planning import (
    build_business_semantic_planning_prompt,
)
from src.backend.application.agent.case_agent.memory.task_state import (
    compact_task_state,
)
from src.backend.application.agent.case_agent.schemas.plan import (
    CaserBusinessSemanticPlan,
)
from src.backend.application.agent.case_agent.shortcut import (
    detect_recent_answer_count_shortcut,
    detect_shortcut,
)
from src.backend.application.agent.case_agent.tools.registry import (
    is_l2_or_l3_capability,
    l1_section_for_capability,
    manifest_item_for_capability,
    normalize_capability_name,
)

from ._errors import state_error


def business_semantic_planner_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Let the model infer query semantics and a controlled execution plan."""

    try:
        run = state["run"]
        user_message = state["user_message"]
        service._repository.update_run(
            run.run_id,
            current_node="business_semantic_planner",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            run.run_id,
            "business_semantic_planning",
            "Understanding the audit question",
            {},
        )

        shortcut = detect_shortcut(user_message.content)
        if shortcut is None:
            shortcut = detect_recent_answer_count_shortcut(
                user_message.content,
                state.get("recent_messages", []),
            )
        if shortcut is not None:
            plan = _shortcut_plan(shortcut)
            state.update(_plan_payload(plan))
            state["shortcut"] = shortcut.to_state()
            state["intent"] = plan.query_semantics.intent
            state["intent_confidence"] = 1.0
            state["missing_slots"] = []
            state["context_plan"] = {"capabilities": []}
            state["next_action"] = "validate_execution_plan"
            return state

        deterministic_plan = _deterministic_expert_plan(
            run.case_id,
            user_message.content,
        )
        if deterministic_plan is None:
            deterministic_plan = _deterministic_followup_plan(
                run.case_id,
                user_message.content,
                state.get("recent_messages", []),
                state.get("session"),
            )
        if deterministic_plan is None:
            deterministic_plan = _deterministic_single_field_plan(
                run.case_id,
                user_message.content,
            )
        if deterministic_plan is not None:
            state.update(_plan_payload(deterministic_plan))
            state["intent"] = deterministic_plan.query_semantics.intent
            state["intent_confidence"] = 1.0
            state["slots"] = {}
            state["missing_slots"] = []
            state["context_plan"] = {
                "capabilities": [
                    {
                        "step": item.step,
                        "name": item.capability,
                        "arguments": item.arguments,
                        "layer": item.layer,
                        "depends_on": item.depends_on,
                        "reason": item.reason,
                    }
                    for item in deterministic_plan.execution_plan
                ],
            }
            state["next_action"] = "validate_execution_plan"
            return state

        response = service._gateway.complete(
            messages=[
                {
                    "role": "system",
                    "content": build_business_semantic_planning_prompt(run.case_id),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_question": user_message.content,
                            "recent_context": _recent_context_for_planner(
                                state.get("recent_messages", []),
                                state.get("session"),
                            ),
                            "instruction": (
                                "如果 current_question 是“这个/它/为什么会触发/为什么会这样”等追问，"
                                "先使用 recent_context 中最近的规则编号、字段名、来源引用或 current_topic 补全对象；"
                                "只有 recent_context 仍无法定位对象时，才返回 missing_slots。"
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            tools=[],
            require_json=True,
            model=service._classifier_model,
            thinking_enabled=False,
            max_tokens=getattr(service, "_planner_max_tokens", 768),
            timeout_seconds=getattr(service, "_planner_timeout_seconds", 15),
        )
        state["model_call_count"] = state.get("model_call_count", 0) + 1
        recorder = getattr(service, "record_model_call_metrics", None)
        if callable(recorder):
            recorder(state, "business_semantic_planner", response)
        plan = _parse_plan(response.content or "{}")
        state.update(_plan_payload(plan))
        state["intent"] = plan.query_semantics.intent
        state["intent_confidence"] = 0.85
        state["slots"] = {}
        state["missing_slots"] = plan.query_semantics.missing_slots
        state["context_plan"] = {
            "capabilities": [
                {
                    "step": item.step,
                        "name": item.capability,
                        "arguments": item.arguments,
                        "layer": item.layer,
                        "depends_on": item.depends_on,
                        "reason": item.reason,
                    }
                    for item in plan.execution_plan
            ],
        }
        state["next_action"] = "validate_execution_plan"
        return state
    except (json.JSONDecodeError, ValidationError) as exc:
        state.update(_plan_payload(_fallback_general_help_plan()))
        state["intent"] = "general_help"
        state["intent_confidence"] = 0.0
        state["missing_slots"] = []
        state["context_plan"] = {"capabilities": []}
        state["planning_error"] = f"{exc.__class__.__name__}: {exc}"
        state["next_action"] = "validate_execution_plan"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _plan_payload(plan: CaserBusinessSemanticPlan) -> dict[str, Any]:
    """Keep optional planner metadata out of legacy plans when it is empty."""

    payload = plan.model_dump(mode="json")
    for item in payload.get("execution_plan", []):
        if not item.get("covers"):
            item.pop("covers", None)
    return payload


def _shortcut_plan(shortcut: Any) -> CaserBusinessSemanticPlan:
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": _shortcut_intent(shortcut),
                "user_goal": shortcut.kind,
                "granularity": "overview",
                "missing_slots": [],
            },
            "execution_plan": [],
        }
    )


def _fallback_general_help_plan() -> CaserBusinessSemanticPlan:
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "general_help",
                "user_goal": "planner fallback",
                "granularity": "overview",
                "missing_slots": [],
            },
            "execution_plan": [],
        }
    )


def _deterministic_expert_plan(case_id: str, text: str) -> CaserBusinessSemanticPlan | None:
    user_goal = str(text or "").strip()
    drug_price_filters = _drug_price_reference_filters(user_goal)
    if not drug_price_filters and not _looks_like_policy_expert_goal(user_goal):
        return None
    arguments: dict[str, Any] = {
        "case_id": case_id,
        "question": user_goal,
        "goal": user_goal,
    }
    reason = "该问题需要查询政策口径，由 Policy Expert 子 Agent 处理。"
    if drug_price_filters:
        arguments["filters"] = drug_price_filters
        reason = "该问题需要查询上海药品价格参考数据，由 Policy RAG MCP 返回非政策依据的 reference 证据。"
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "expert_task",
                "user_goal": user_goal,
                "granularity": "analysis",
                "missing_slots": [],
            },
            "execution_plan": [
                {
                    "step": 1,
                    "layer": "L3",
                    "capability": "ask_policy_expert",
                    "arguments": arguments,
                    "depends_on": [],
                    "reason": reason,
                }
            ],
        }
    )


def _deterministic_single_field_plan(case_id: str, text: str) -> CaserBusinessSemanticPlan | None:
    user_goal = str(text or "").strip()
    if not user_goal:
        return None
    capability: str | None = None
    if _looks_like_case_basic_field_goal(user_goal):
        capability = "query_case_basic_info"
    elif _looks_like_medical_visit_goal(user_goal):
        capability = "query_medical_materials"
    elif _looks_like_claimant_profile_field_goal(user_goal):
        capability = "query_claimant_profile"
    elif _looks_like_risk_score_field_goal(user_goal):
        capability = "query_risk_score"
    elif _looks_like_settlement_field_goal(user_goal):
        capability = "query_settlement_materials"
    if capability is None:
        return None
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "case_task",
                "user_goal": user_goal,
                "granularity": (
                    "analysis" if _looks_like_scene_definition_question(user_goal) else "single_field"
                ),
                "missing_slots": [],
            },
            "execution_plan": [
                {
                    "step": 1,
                    "layer": "L1",
                    "capability": capability,
                    "arguments": {"case_id": case_id},
                    "depends_on": [],
                    "reason": (
                        "该问题是当前案件字段的场景化解释，应结合结构化案件事实回答。"
                        if _looks_like_scene_definition_question(user_goal)
                        else "该问题可由当前案件的结构化字段直接回答。"
                    ),
                }
            ],
        }
    )


def capability_plan_from_hint(
    case_id: str,
    question: str,
    hint: dict[str, Any],
) -> CaserBusinessSemanticPlan | None:
    """Compile a safe read-only L1 hint into a minimal one-step plan."""

    capability = normalize_capability_name(str(hint.get("capability_hint") or ""))
    manifest = manifest_item_for_capability(capability)
    if (
        manifest is None
        or manifest.layer != "L1"
        or manifest.status != "available"
        or l1_section_for_capability(capability) is None
    ):
        return None
    if not manifest.can_answer_directly:
        return None
    information_needs = [
        str(item)[:80]
        for item in (
            hint.get("information_needs")
            or manifest.information_needs
            or ((manifest.section_key,) if manifest.section_key else (capability,))
        )
        if str(item or "").strip()
    ][:12]
    granularity = _normalize_granularity(
        hint.get("answer_shape"),
        user_goal=question,
        execution_plan=[],
    )
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "case_task",
                "user_goal": str(question or "")[:500],
                "granularity": granularity,
                "information_needs": information_needs,
                "missing_slots": [],
            },
            "execution_plan": [
                {
                    "step": 1,
                    "layer": "L1",
                    "capability": capability,
                    "arguments": {"case_id": case_id},
                    "depends_on": [],
                    "covers": information_needs,
                    "reason": "高置信只读能力提示可直接编译为最小执行计划。",
                }
            ],
        }
    )


def _deterministic_followup_plan(
    case_id: str,
    text: str,
    recent_messages: Any,
    session: Any,
) -> CaserBusinessSemanticPlan | None:
    user_goal = str(text or "").strip()
    if not _looks_like_trigger_followup(user_goal):
        return None
    rule_id = _extract_recent_rule_id(recent_messages, session)
    if not rule_id:
        return None
    enriched_goal = f"{user_goal}（承接上一轮对象：{rule_id}）"
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "case_task",
                "user_goal": enriched_goal,
                "granularity": "analysis",
                "missing_slots": [],
            },
            "execution_plan": [
                {
                    "step": 1,
                    "layer": "L1",
                    "capability": "query_rule_verification",
                    "arguments": {
                        "case_id": case_id,
                        "filters": {"rule_id": rule_id},
                    },
                    "depends_on": [],
                    "reason": "短追问承接上一轮规则对象，读取当前案件规则核验清单解释触发原因。",
                }
            ],
        }
    )


def _parse_plan(content: str) -> CaserBusinessSemanticPlan:
    text = content.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1].strip()
    payload = json.loads(text or "{}")
    payload = _normalize_plan_payload(payload)
    return CaserBusinessSemanticPlan.model_validate(payload)


def _shortcut_intent(shortcut: Any) -> str:
    if getattr(shortcut, "intent", "") == "general_help":
        return "general_help"
    if getattr(shortcut, "display_mode", "") == "unavailable":
        return "expert_task"
    return "case_task"


def _normalize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize old planner drift before strict schema validation."""

    if not isinstance(payload, dict):
        return payload
    semantics = payload.get("query_semantics")
    execution_plan = payload.get("execution_plan")
    if not isinstance(semantics, dict):
        semantics = {}
    if not isinstance(execution_plan, list):
        execution_plan = []
    missing_slots = semantics.get("missing_slots")
    if not isinstance(missing_slots, list):
        missing_slots = []

    semantics = {
        "intent": _normalize_intent(
            semantics.get("intent"),
            execution_plan=execution_plan,
            missing_slots=missing_slots,
        ),
        "user_goal": str(semantics.get("user_goal") or "")[:500],
        "granularity": _normalize_granularity(
            semantics.get("granularity"),
            user_goal=str(semantics.get("user_goal") or ""),
            execution_plan=execution_plan,
        ),
        "information_needs": [
            str(item)[:80]
            for item in (
                semantics.get("information_needs")
                if isinstance(semantics.get("information_needs"), list)
                else []
            )[:12]
            if str(item or "").strip()
        ],
        "missing_slots": [str(item)[:80] for item in missing_slots[:8]],
    }
    normalized_plan = []
    for index, item in enumerate(execution_plan[:8], start=1):
        if not isinstance(item, dict):
            continue
        normalized_plan.append(
            {
                "step": item.get("step") or index,
                "layer": item.get("layer") or _infer_layer(item.get("capability")),
                "capability": item.get("capability") or item.get("name") or "",
                "arguments": item.get("arguments")
                if isinstance(item.get("arguments"), dict)
                else {},
                "depends_on": item.get("depends_on")
                if isinstance(item.get("depends_on"), list)
                else [],
                "covers": [
                    str(value)[:80]
                    for value in (
                        item.get("covers")
                        if isinstance(item.get("covers"), list)
                        else []
                    )[:12]
                    if str(value or "").strip()
                ],
                "reason": str(item.get("reason") or "")[:300],
            }
        )
    return {
        "query_semantics": semantics,
        "execution_plan": normalized_plan,
    }


def _normalize_intent(value: Any, *, execution_plan: list[Any], missing_slots: list[Any]) -> str:
    intent = str(value or "").strip()
    if intent in {"general_help", "case_task", "expert_task", "clarification_required"}:
        return intent
    if missing_slots:
        return "clarification_required"
    if execution_plan:
        capabilities = [
            normalize_capability_name(str(item.get("capability") or item.get("name") or ""))
            for item in execution_plan
            if isinstance(item, dict)
        ]
        if any(is_l2_or_l3_capability(capability) for capability in capabilities):
            return "expert_task"
        return "case_task"
    return "general_help"


def _infer_layer(capability: Any) -> str:
    name = normalize_capability_name(str(capability or ""))
    if l1_section_for_capability(name) is not None:
        return "L1"
    if is_l2_or_l3_capability(name):
        return "L3" if name.startswith("ask_") else "L2"
    return "L1"


def _normalize_granularity(value: Any, *, user_goal: str, execution_plan: list[Any]) -> str:
    granularity = str(value or "").strip()
    if granularity in {
        "single_field",
        "field_group",
        "list",
        "detail",
        "overview",
        "analysis",
    }:
        return granularity
    text = user_goal.strip()
    if _looks_like_review_status_goal(text):
        return "single_field"
    if any(marker in text for marker in ("为什么", "怎么看", "是否", "偏高", "合理", "核验", "分析")):
        return "analysis"
    if any(marker in text for marker in ("所有", "全部", "有哪些", "清单", "列表", "分别")):
        return "list"
    if any(marker in text for marker in ("详情", "明细", "具体", "内容")):
        return "detail"
    if _looks_like_single_field_goal(text):
        return "single_field"
    if execution_plan:
        return "overview"
    return "overview"


def _looks_like_single_field_goal(text: str) -> bool:
    if not text:
        return False
    single_field_markers = (
        "性别",
        "年龄",
        "年龄段",
        "参保类型",
        "参保地",
        "就医地",
        "就医类型",
        "报销方式",
        "备案状态",
        "急诊材料",
        "直接结算",
        "过敏史",
        "慢病",
        "案件类型",
        "审核状态",
        "审核通过",
        "通过了吗",
        "过了吗",
        "是否通过",
        "有没有通过",
        "人工初审",
        "人工复审",
        "申诉",
        "案件编号",
        "接入方式",
        "风险等级",
        "风险评分",
        "申报总费用",
        "药品费",
        "检查费",
        "治疗费",
    )
    return any(marker in text for marker in single_field_markers) and not any(
        marker in text for marker in ("哪些", "所有", "全部", "清单", "分析", "为什么")
    )


def _looks_like_scene_definition_question(text: str) -> bool:
    if not text:
        return False
    definition_markers = ("什么是", "是什么意思", "代表什么", "怎么理解", "含义")
    scene_fields = (
        "备案状态",
        "审核状态",
        "报销方式",
        "直接结算",
        "急诊材料",
        "风险等级",
        "综合风险提示强度",
    )
    return any(marker in text for marker in definition_markers) and any(
        field in text for field in scene_fields
    )


def _looks_like_trigger_followup(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    markers = (
        "为什么会触发",
        "为什么触发",
        "怎么触发",
        "为何触发",
        "为什么命中",
        "怎么命中",
        "为什么会命中",
        "为什么会这样",
        "为什么这样",
        "原因是什么",
        "这个为什么",
        "这个呢",
        "它为什么",
    )
    return any(marker in compact for marker in markers)


def _looks_like_review_status_goal(text: str) -> bool:
    if not text:
        return False
    markers = (
        "审核状态",
        "审核通过",
        "通过了",
        "通过了吗",
        "过了吗",
        "是否通过",
        "有没有通过",
        "人工初审",
        "人工复审",
        "申诉",
    )
    return any(marker in text for marker in markers) and not any(
        marker in text for marker in ("为什么", "原因", "材料", "政策")
    )


def _looks_like_case_basic_field_goal(text: str) -> bool:
    markers = (
        "参保地",
        "参保地区",
        "在哪里参保",
        "在哪参保",
        "哪里参保",
        "就医地",
        "在哪里就医",
        "在哪就医",
        "哪里就医",
        "就医类型",
        "报销方式",
        "备案状态",
        "急诊材料",
        "直接结算",
        "案件类型",
        "审核状态",
        "审核通过",
        "通过了",
        "通过了吗",
        "过了吗",
        "是否通过",
        "有没有通过",
        "人工初审",
        "人工复审",
        "申诉",
        "案件编号",
        "接入方式",
    )
    return _looks_like_atomic_field_question(text, markers)


def _looks_like_medical_visit_goal(text: str) -> bool:
    markers = (
        "就诊信息",
        "就诊记录",
        "就诊过",
        "哪里就诊",
        "去哪就诊",
        "去了哪里",
        "都去哪里",
        "都在哪",
        "都去过",
    )
    return any(marker in text for marker in markers) and not any(
        marker in text for marker in ("政策", "目录", "待遇", "支付标准")
    )


def _looks_like_claimant_profile_field_goal(text: str) -> bool:
    markers = (
        "性别",
        "年龄",
        "年龄段",
        "参保类型",
        "过敏史",
        "慢病",
    )
    return _looks_like_atomic_field_question(text, markers)


def _looks_like_risk_score_field_goal(text: str) -> bool:
    markers = ("风险等级", "风险评分", "综合风险提示强度")
    return _looks_like_atomic_field_question(text, markers)


def _looks_like_settlement_field_goal(text: str) -> bool:
    markers = ("申报总费用", "药品费", "检查费", "治疗费")
    return _looks_like_atomic_field_question(text, markers)


def _looks_like_policy_expert_goal(text: str) -> bool:
    if not text:
        return False
    if _drug_price_reference_filters(text):
        return True
    policy_markers = (
        "政策",
        "口径",
        "规定",
        "目录",
        "待遇",
        "支付标准",
        "支付范围",
        "分工",
        "适用范围",
    )
    domain_markers = (
        "医保",
        "报销",
        "跨省",
        "异地",
        "异地就医",
        "参保地",
        "就医地",
        "备案",
        "支付",
        "药品",
        "耗材",
    )
    policy_domain_markers = (
        "基金支付",
        "基金监管",
        "监督检查",
        "拒不配合",
        "暂停联网结算",
        "定点医疗机构",
        "定点零售药店",
        "定点医药机构",
        "机构编码",
        "药店编码",
        "长期处方",
        "长处方",
        "特殊病备案",
        "特殊疾病范围",
        "门诊特殊疾病",
        "预付金",
        "费用协查",
        "银行手续费",
        "银行票据",
        "住院床位费",
        "急诊观察室床位费",
        "双通道",
        "谈判药品",
    )
    policy_question_markers = (
        "能否",
        "是否",
        "能不能",
        "怎么",
        "如何",
        "核验",
        "处理",
        "依据",
        "规则",
        "范围",
        "支付",
        "报销",
        "纳入",
        "不纳入",
    )
    if any(marker in text for marker in policy_markers) and any(
        marker in text for marker in domain_markers
    ):
        return True
    if any(marker in text for marker in policy_domain_markers) and any(
        marker in text for marker in policy_question_markers
    ):
        return True
    if any(marker in text for marker in ("跨省异地", "异地就医")) and any(
        marker in text for marker in ("如何", "怎么", "分工", "目录", "待遇", "规定")
    ):
        return True
    return False


def _drug_price_reference_filters(text: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or "上海" not in compact:
        return {}
    if _has_drug_catalog_payment_intent(compact):
        return {}
    price_markers = (
        "药价",
        "价格",
        "均价",
        "平均价",
        "平均价格",
        "周均价",
        "价格区间",
        "多少钱",
    )
    drug_markers = (
        "药",
        "药品",
        "药店",
        "布洛芬",
        "阿莫西林",
        "阿托伐他汀",
        "二甲双胍",
        "氨氯地平",
        "缬沙坦",
        "连花清瘟",
        "抗病毒口服液",
        "头孢",
        "胶囊",
        "片",
        "口服液",
        "颗粒",
    )
    if not any(marker in compact for marker in price_markers):
        return {}
    if not any(marker in compact for marker in drug_markers):
        return {}
    return {
        "jurisdiction": "shanghai",
        "policy_domain": "drug_product_price_reference",
        "content_type": "table_row",
        "can_cite_as_policy_basis": False,
    }


def _has_drug_catalog_payment_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "医保类别",
            "目录编号",
            "医保目录",
            "药品目录",
            "医保药品目录",
            "支付比例",
            "本地支付比例",
            "报销比例",
            "甲类",
            "乙类",
            "限定支付范围",
        )
    )


def _looks_like_atomic_field_question(text: str, markers: tuple[str, ...]) -> bool:
    if not any(marker in text for marker in markers):
        return False
    return not any(
        marker in text for marker in ("哪些", "所有", "全部", "清单", "列表", "分析", "为什么", "核验")
    )


def _recent_context_for_planner(recent_messages: Any, session: Any) -> dict[str, Any]:
    task_state = compact_task_state(getattr(session, "task_state", None) or {})
    message_limit = 2 if task_state.get("task_id") else 6
    snippets: list[dict[str, Any]] = []
    for message in list(recent_messages or [])[-message_limit:]:
        content = str(getattr(message, "content", "") or "")
        if not content:
            continue
        answer_payload = getattr(message, "answer_payload", None)
        source_refs = list(getattr(message, "source_refs", []) or [])
        if answer_payload is not None:
            try:
                source_refs.extend(
                    ref
                    for block in getattr(answer_payload, "content_blocks", []) or []
                    for ref in getattr(block, "source_refs", []) or []
                )
            except TypeError:
                pass
        snippets.append(
            {
                "role": getattr(message, "role", ""),
                "content": content[:500],
                "source_refs": list(dict.fromkeys(ref for ref in source_refs if isinstance(ref, str)))[:12],
            }
        )
    return {
        "current_topic": str(getattr(session, "current_topic", "") or "")[:200]
        if session is not None
        else "",
        "referenced_source_refs": list(getattr(session, "referenced_source_refs", []) or [])[:12]
        if session is not None
        else [],
        "structured_task_state": task_state,
        "recent_messages": snippets,
    }


def _extract_recent_rule_id(recent_messages: Any, session: Any) -> str:
    candidates: list[str] = []
    if session is not None:
        candidates.extend(str(ref) for ref in getattr(session, "referenced_source_refs", []) or [])
        candidates.append(str(getattr(session, "current_topic", "") or ""))
    for message in reversed(list(recent_messages or [])):
        candidates.append(str(getattr(message, "content", "") or ""))
        answer_payload = getattr(message, "answer_payload", None)
        if answer_payload is not None:
            try:
                candidates.extend(str(block.text) for block in answer_payload.content_blocks)
                candidates.extend(str(source.source_ref) for source in answer_payload.sources)
                candidates.extend(str(source.title) for source in answer_payload.sources)
            except AttributeError:
                pass
        candidates.extend(str(ref) for ref in getattr(message, "source_refs", []) or [])
    for text in candidates:
        match = re.search(r"\b(OP-R\d{3,})\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""
