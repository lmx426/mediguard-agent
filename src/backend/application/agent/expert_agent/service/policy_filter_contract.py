"""Strict structured contract for LLM-generated Policy RAG filters."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.backend.domain.policy_rag_taxonomy import (
    CONTENT_TYPES,
    JURISDICTIONS,
    POLICY_DOMAINS,
)

Jurisdiction = Enum(
    "Jurisdiction",
    {value.upper(): value for value in JURISDICTIONS},
    type=str,
)
PolicyDomain = Enum(
    "PolicyDomain",
    {value.upper(): value for value in POLICY_DOMAINS},
    type=str,
)
ContentType = Enum(
    "ContentType",
    {value.upper(): value for value in CONTENT_TYPES},
    type=str,
)


JURISDICTION_ALIASES = {
    "国家": "national",
    "国家级": "national",
    "全国": "national",
    "国家政策": "national",
    "北京": "beijing",
    "北京市": "beijing",
    "北京医保": "beijing",
    "上海": "shanghai",
    "上海市": "shanghai",
    "上海医保": "shanghai",
}

POLICY_DOMAIN_ALIASES = {
    "待遇政策": "benefit",
    "医保待遇": "benefit",
    "长期处方": "chronic_disease_long_prescription",
    "慢病长处方": "chronic_disease_long_prescription",
    "定点机构": "designated_institution",
    "定点医药机构": "designated_institution",
    "定点零售药店": "designated_institution",
    "医保药品目录": "drug_catalog",
    "药品目录": "drug_catalog",
    "药品价格参考": "drug_product_price_reference",
    "急诊": "emergency",
    "基金监管": "fund_supervision",
    "手工报销": "manual_reimbursement",
    "零星报销": "manual_reimbursement",
    "医疗服务价格": "medical_service_price",
    "诊疗项目价格": "medical_service_price",
    "异地就医": "remote_medical",
    "跨省异地就医": "remote_medical",
    "异地手工报销": "remote_medical_manual_reimbursement",
    "上海支付范围": "shanghai_payment_scope",
    "特殊病备案": "special_disease_filing",
    "特殊病范围": "special_disease_scope",
    "门诊特殊疾病范围": "special_disease_scope",
}

CONTENT_TYPE_ALIASES = {
    "policy": "policy_text",
    "policy_file": "policy_text",
    "policy_document": "policy_text",
    "policy_interpretation": "policy_text",
    "policy_pdf": "policy_text",
    "document_markdown": "policy_text",
    "application/pdf": "policy_text",
    "政策": "policy_text",
    "政策文件": "policy_text",
    "政策正文": "policy_text",
    "政策文本": "policy_text",
    "table": "table_row",
    "table_row": "table_row",
    "表格": "table_row",
    "表格行": "table_row",
    "目录表": "table_row",
    "结构化表格": "table_row",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_enum_values(value: Any, aliases: dict[str, str]) -> list[str]:
    normalized: list[str] = []
    for item in _as_list(value):
        text = str(item or "").strip()
        if not text:
            continue
        candidate = aliases.get(text, aliases.get(text.lower(), text.lower()))
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


class PolicyFilters(BaseModel):
    """The exact metadata boundary sent to Policy RAG MCP."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    jurisdiction: list[Jurisdiction] = Field(
        default_factory=list,
        description=(
            "政策适用地区。只允许 national、beijing、shanghai；跨地区问题可多选。"
        ),
    )
    policy_domain: list[PolicyDomain] = Field(
        default_factory=list,
        description="问题实际需要检索的政策领域；不得为了提高召回而追加无关领域。",
    )
    content_type: list[ContentType] = Field(
        default_factory=list,
        description=(
            "可过滤的 chunk 结构只包括规则正文 policy_text 和结构化行 table_row；"
            "办理指南和问答属于 doc_type，不属于 content_type。"
        ),
    )
    can_cite_as_policy_basis: bool = Field(
        default=True,
        description="是否允许把结果作为政策依据引用。价格参考数据必须为 false。",
    )

    @field_validator("jurisdiction", mode="before")
    @classmethod
    def normalize_jurisdiction(cls, value: Any) -> list[str]:
        return _normalize_enum_values(value, JURISDICTION_ALIASES)

    @field_validator("policy_domain", mode="before")
    @classmethod
    def normalize_policy_domain(cls, value: Any) -> list[str]:
        return _normalize_enum_values(value, POLICY_DOMAIN_ALIASES)

    @field_validator("content_type", mode="before")
    @classmethod
    def normalize_content_type(cls, value: Any) -> list[str]:
        return _normalize_enum_values(value, CONTENT_TYPE_ALIASES)

    @model_validator(mode="after")
    def require_retrieval_boundary(self) -> "PolicyFilters":
        if not self.jurisdiction and not self.policy_domain:
            raise ValueError(
                "filters must contain at least jurisdiction or policy_domain"
            )
        return self


class PolicyRetrievalPlanDraft(BaseModel):
    """Single structured LLM submission used by task understanding and rewrite."""

    model_config = ConfigDict(extra="forbid")

    policy_question: str = Field(min_length=1, max_length=1200)
    answer_mode: Literal["fact_lookup", "list", "process_rule", "policy_explanation", "comparison"] = "process_rule"
    information_needs: list[str] = Field(default_factory=list, max_length=12)
    filters: PolicyFilters
    top_k: int | None = Field(default=None, ge=1, le=20)
    fetch_k: int = Field(default=40, ge=1, le=200)
    rerank: bool = False
    need_case_context: bool = False
    case_context_requests: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=3,
    )
    reason: str = Field(default="", max_length=500)
    filter_reason: str = Field(default="", max_length=500)
    filter_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_information_needs(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if isinstance(payload.get("information_needs"), str):
            payload["information_needs"] = [payload["information_needs"]]
        elif isinstance(payload.get("information_needs"), dict):
            payload["information_needs"] = [
                payload["information_needs"].get("need_text")
                or payload["information_needs"].get("label")
                or payload["information_needs"].get("question_span")
                or ""
            ]
        return payload

    @model_validator(mode="after")
    def _require_information_needs(self) -> "PolicyRetrievalPlanDraft":
        if not self.information_needs:
            fallback = str(self.policy_question or "").strip()
            self.information_needs = [fallback] if fallback else ["policy_question"]
        return self


def repair_json_object(raw_output: str | dict[str, Any]) -> dict[str, Any]:
    """Repair model JSON syntax and return a JSON object."""

    if isinstance(raw_output, dict):
        return dict(raw_output)
    text = str(raw_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        from json_repair import loads as repair_loads
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError(
            "PolicyFilterResolver requires the json-repair package"
        ) from exc
    payload = repair_loads(text)
    if not isinstance(payload, dict):
        raise ValueError("policy filter output must be a JSON object")
    return payload


def policy_retrieval_plan_tool_schema() -> dict[str, Any]:
    """Return the single function schema registered with the planning LLM."""

    return {
        "type": "function",
        "function": {
            "name": "submit_policy_retrieval_plan",
            "description": (
                "提交与 Policy RAG MCP 元数据字段完全兼容的医保政策检索计划。"
            ),
            "parameters": PolicyRetrievalPlanDraft.model_json_schema(),
        },
    }
