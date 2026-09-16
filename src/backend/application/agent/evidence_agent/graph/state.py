"""Evidence Agent 运行时状态定义。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """在 StateGraph 各节点间流转的运行时状态。

    字段说明：
        run_id: 本次 Agent 运行的唯一标识。
        case_id: 被分析案件的唯一标识。
        request: EvidenceAgentRequest 序列化后的字典。
        input_fingerprint: 案件输入数据的哈希指纹，用于判断是否需要重新运行。
        started_monotonic: 运行开始的 monotonic 时间戳，用于超时检测。
        messages: LLM 对话消息列表。
        ledger: 工具收集的证据账本，键为 ledger_ref，值为 EvidenceLedgerItem 字典。
        system_risk_prompt: 本次分析使用的后端综合风险提示只读快照。
        tool_catalog: 从工具注册表动态生成的模型工具选择说明。
        pending_response: 上一次模型调用返回的标准化响应。
        model_call_count: 已调用的模型次数。
        tool_call_count: 已调用的工具次数。
        planning_rounds: 规划轮数。
        argument_error_count: 工具参数错误累计次数。
        repeat_count: 重复工具调用次数。
        no_new_evidence_rounds: 连续未收集到新证据的轮数。
        structure_repair_count: 结构修复重试次数。
        validation_repair_count: 验证修复重试次数。
        seen_signatures: 已见过的工具调用签名列表，用于去重。
        force_final_json: 是否强制要求模型输出最终 JSON。
        final_content: 模型生成的最终分析内容。
        next_action: StateGraph 条件路由下一步动作。
        error_code: 终止错误代码。
        error_message: 终止错误描述。
    """

    run_id: str
    case_id: str
    request: dict[str, Any]
    input_fingerprint: str
    started_monotonic: float
    messages: list[dict[str, Any]]
    ledger: dict[str, dict[str, Any]]
    system_risk_prompt: dict[str, Any]
    tool_catalog: dict[str, Any]
    pending_response: Any  # ModelResponse，避免循环导入
    model_call_count: int
    tool_call_count: int
    planning_rounds: int
    argument_error_count: int
    repeat_count: int
    no_new_evidence_rounds: int
    structure_repair_count: int
    validation_repair_count: int
    seen_signatures: list[str]
    force_final_json: bool
    final_content: dict[str, Any]
    next_action: str
    error_code: str
    error_message: str
