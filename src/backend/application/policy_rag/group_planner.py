"""Deterministic evidence-group planning for policy retrieval.

The Policy RAG MCP service deliberately does not run an LLM planner.  The
caller supplies the question and metadata filters; this module only splits
known policy domains into small evidence groups so that one source family
cannot crowd out another one in a cross-document query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalGroup:
    """One dense retrieval pass focused on a policy evidence family."""

    name: str
    question: str
    filters: dict[str, object]
    priority: int
    anchor_hints: tuple[str, ...] = ()
    preferred_doc_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """The deterministic plan used for one policy search."""

    groups: tuple[RetrievalGroup, ...]
    strategy: str
    profile: str = "strict_v3"


class PolicyRetrievalGroupPlanner:
    """Splits only explicitly supported policy domains into retrieval groups."""

    _GROUPS: tuple[
        tuple[str, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]],
        ...,
    ] = (
        (
            "remote_medical",
            ("remote_medical", "remote_medical_manual_reimbursement"),
            "异地就医备案、就医地目录与参保地待遇分工",
            ("备案", "就医地", "参保地", "直接结算", "急诊抢救"),
            ("policy_text", "policy_interpretation", "faq"),
        ),
        (
            "manual_reimbursement",
            ("manual_reimbursement", "remote_medical_manual_reimbursement"),
            "手工报销材料、票据和就诊事实链",
            ("手工报销", "费用清单", "处方", "诊疗证明", "费用收据", "门诊病历"),
            ("policy_text", "faq", "service_guide", "material_chain_table"),
        ),
        (
            "benefit",
            ("benefit",),
            "参保地门急诊待遇、起付线和支付比例",
            ("起付标准", "支付比例", "最高支付限额", "退休人员", "门诊"),
            ("service_guide", "table_row", "policy_text"),
        ),
        (
            "shanghai_payment_scope",
            ("shanghai_payment_scope",),
            "上海药品、诊疗项目和医用耗材支付范围",
            ("药品目录", "医用耗材", "支付范围", "支付办法", "CT"),
            ("policy_text", "table_row", "catalog_notice"),
        ),
        (
            "medical_service_price",
            ("medical_service_price",),
            "医疗服务价格、计价单位和项目内涵",
            ("CT", "计价单位", "按部位收费", "价格", "项目内涵"),
            ("price_table", "table_row", "policy_text"),
        ),
        (
            "drug_catalog",
            ("drug_catalog",),
            "医保药品目录和限定支付范围",
            ("药品目录", "甲类", "乙类", "限定支付", "支付范围"),
            ("table_row", "catalog_notice", "policy_text"),
        ),
        (
            "fund_supervision",
            ("fund_supervision",),
            "医保基金监管中的收费和用药核验",
            ("重复收费", "分解收费", "超标准收费", "超量开药", "重复开药", "串换"),
            ("policy_text", "case_study", "interpretation"),
        ),
        (
            "chronic_disease",
            (
                "chronic_disease_long_prescription",
                "special_disease_filing",
                "special_disease_scope",
            ),
            "慢病长期处方、续方、备案和病种范围",
            ("慢病", "长期处方", "续方", "备案", "病种"),
            ("policy_text", "faq", "service_guide"),
        ),
        (
            "designated_institution",
            ("designated_institution",),
            "定点医疗机构、药店和机构属性",
            ("定点", "药店", "机构", "编码", "状态"),
            ("iframe", "table_row", "catalog_notice", "policy_text"),
        ),
    )

    def plan(
        self,
        *,
        question: str,
        filters: dict[str, object],
    ) -> RetrievalPlan:
        question_text = str(question or "").strip()
        requested_domains = {
            str(value).strip()
            for value in filters.get("policy_domain") or []
            if str(value).strip()
        }
        matched: list[RetrievalGroup] = []
        for priority, (name, domains, focus, anchor_hints, preferred_doc_types) in enumerate(
            self._GROUPS
        ):
            selected_domains = [
                domain for domain in domains if domain in requested_domains
            ]
            if not selected_domains:
                continue
            group_filters = dict(filters)
            group_filters["policy_domain"] = selected_domains
            group_filters["jurisdiction"] = self._jurisdictions_for_group(
                question_text,
                name=name,
                requested=filters.get("jurisdiction"),
            )
            matched.append(
                RetrievalGroup(
                    name=name,
                    question=(
                        f"{question_text}\n检索重点：{focus}。"
                        f"关键词：{'、'.join(anchor_hints)}。"
                    ),
                    filters=group_filters,
                    priority=priority,
                    anchor_hints=anchor_hints,
                    preferred_doc_types=preferred_doc_types,
                )
            )

        if not self._should_split_strict_v3(
            question=question_text,
            matched=matched,
        ):
            return RetrievalPlan(
                groups=(
                    RetrievalGroup(
                        name="general_policy",
                        question=question_text,
                        filters=dict(filters),
                        priority=0,
                    ),
                ),
                strategy="single_group_strict_v3_dense",
            )

        groups = list(matched)
        clauses = _split_query_clauses(question_text)
        if clauses:
            groups.append(
                RetrievalGroup(
                    name="clause:1",
                    question=f"{clauses[0]}\n原始问题：{question_text}",
                    filters=dict(filters),
                    priority=100,
                )
            )
        groups.append(
            RetrievalGroup(
                name="base",
                question=question_text,
                filters=dict(filters),
                priority=999,
            )
        )
        return RetrievalPlan(
            groups=tuple(groups),
            strategy="multi_group_strict_v3_dense",
        )

    @staticmethod
    def _should_split_strict_v3(
        *,
        question: str,
        matched: list[RetrievalGroup],
    ) -> bool:
        if len(matched) < 2:
            return False
        if _is_case_like_complex(question, matched):
            return True
        if len(_split_query_clauses(question)) >= 2 and _has_multi_domain_signal(
            question,
            matched,
        ):
            return True
        return _is_cross_doc_query(question, matched)

    @staticmethod
    def _jurisdictions_for_group(
        question: str,
        *,
        name: str,
        requested: object,
    ) -> list[str]:
        requested_values = [
            str(value).strip()
            for value in requested or []
            if str(value).strip()
        ]
        requested_set = set(requested_values)
        if name == "shanghai_payment_scope" and "shanghai" in requested_set:
            return ["shanghai"]
        if name == "benefit" and "beijing" in requested_set and "北京" in question:
            return ["beijing"]
        if name == "remote_medical":
            remote_values = [
                value
                for value in ("national", "beijing", "shanghai")
                if value in requested_set
            ]
            if "北京" in question and "beijing" in requested_set:
                remote_values = [
                    value for value in remote_values if value != "shanghai"
                ]
            return remote_values or requested_values
        if "北京" in question and "beijing" in requested_set:
            return ["beijing"]
        if "上海" in question and "shanghai" in requested_set:
            return ["shanghai"]
        return requested_values


def plan_policy_retrieval(
    *,
    question: str,
    filters: dict[str, Any],
) -> RetrievalPlan:
    """Convenience entry point kept small for the application use case."""

    return PolicyRetrievalGroupPlanner().plan(question=question, filters=filters)


def _split_query_clauses(question: str) -> list[str]:
    raw_pieces = []
    buffer = ""
    for char in str(question or ""):
        buffer += char
        if char in "。；;？！?\n":
            raw_pieces.append(buffer.strip("。；;？！?\n "))
            buffer = ""
    if buffer.strip():
        raw_pieces.append(buffer.strip())

    clauses: list[str] = []
    for piece in raw_pieces:
        for sub_piece in piece.split("，"):
            clause = sub_piece.strip(" ,，")
            if len(clause) >= 10 and clause not in clauses:
                clauses.append(clause)
    return clauses


def _is_case_like_complex(
    question: str,
    matched: list[RetrievalGroup],
) -> bool:
    if len(matched) < 2:
        return False
    case_terms = (
        "本案",
        "案件",
        "审核",
        "审核员",
        "参保人",
        "患者",
        "申请",
        "报销",
        "就医",
        "费用",
        "票据",
        "处方",
        "病历",
    )
    case_term_count = sum(1 for term in case_terms if term in question)
    return case_term_count >= 2 and (
        len(_split_query_clauses(question)) >= 2
        or _has_multi_domain_signal(question, matched)
    )


def _is_cross_doc_query(
    question: str,
    matched: list[RetrievalGroup],
) -> bool:
    if len(matched) < 2:
        return False
    connectors = (
        "分别",
        "同时",
        "以及",
        "和",
        "与",
        "及",
        "并",
        "分工",
        "核验",
        "哪些",
        "如何处理",
        "如何判断",
        "how",
        "and",
    )
    if any(connector in question.lower() for connector in connectors):
        return True
    return _has_multi_domain_signal(question, matched)


def _has_multi_domain_signal(
    question: str,
    matched: list[RetrievalGroup],
) -> bool:
    signal_count = 0
    for group in matched:
        terms = [group.name, *group.anchor_hints]
        if any(str(term) and str(term) in question for term in terms):
            signal_count += 1
    return signal_count >= 2
