"""Generate a corpus-driven Policy RAG evaluation set.

The generated records are grounded in existing ``policy_nodes.jsonl`` nodes.
They are meant to evaluate retrieval coverage, not Case Agent reasoning or
audit decisions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_NODES_PATH,
)


SCHEMA_VERSION = "policy_rag_eval_v1_3"
RANDOM_SEED = 20260818
CORPUS_COUNTS = {
    "政策 / 待遇 / 办事规则": 70,
    "药品目录": 40,
    "定点医疗机构 / 药店": 30,
    "医疗服务价格": 25,
    "跨文档组合问题": 25,
    "FAQ / 办事指南细节": 10,
}
QUESTION_STYLE_COUNTS = {
    "standard": 70,
    "explanation": 40,
    "material_flow": 30,
    "condition": 30,
    "colloquial": 20,
    "boundary": 10,
}
BUSINESS_ALLOWED_STYLES = {
    "政策 / 待遇 / 办事规则": {
        "standard",
        "explanation",
        "material_flow",
        "condition",
        "colloquial",
        "boundary",
    },
    "药品目录": {
        "standard",
        "explanation",
        "material_flow",
        "condition",
        "colloquial",
        "boundary",
    },
    "定点医疗机构 / 药店": {
        "standard",
        "explanation",
        "material_flow",
        "condition",
        "colloquial",
        "boundary",
    },
    "医疗服务价格": {
        "standard",
        "explanation",
        "material_flow",
        "condition",
        "colloquial",
        "boundary",
    },
    "跨文档组合问题": {
        "standard",
        "explanation",
        "condition",
        "colloquial",
        "boundary",
    },
    "FAQ / 办事指南细节": {
        "standard",
        "material_flow",
        "condition",
        "colloquial",
    },
}

DEFAULT_CORPUS_EVAL_SET = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1_3_corpus_200.jsonl"
)
DEFAULT_COMBINED_EVAL_SET = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1_3_combined_224.jsonl"
)
DEFAULT_BASE_EVAL_SET = DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1_2.jsonl"
DEFAULT_AUDIT_REPORT = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_eval_standard_audit_v1_3_corpus_200.md"
)


@dataclass(frozen=True, slots=True)
class PolicyNode:
    node_id: str
    text: str
    metadata: dict[str, Any]
    fields: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 200 corpus-driven Policy RAG eval samples."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--base-eval-jsonl", type=Path, default=DEFAULT_BASE_EVAL_SET)
    parser.add_argument(
        "--corpus-eval-jsonl",
        type=Path,
        default=DEFAULT_CORPUS_EVAL_SET,
    )
    parser.add_argument(
        "--combined-eval-jsonl",
        type=Path,
        default=DEFAULT_COMBINED_EVAL_SET,
    )
    parser.add_argument("--audit-report-md", type=Path, default=DEFAULT_AUDIT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = load_nodes(args.nodes_jsonl)
    rows = build_corpus_eval(nodes)
    audit = audit_rows(rows, nodes)
    if audit["failure_count"]:
        raise RuntimeError(
            f"Generated eval set has {audit['failure_count']} audit failures; "
            f"see {args.audit_report_md}"
        )
    write_jsonl(args.corpus_eval_jsonl, rows)
    base_rows = load_jsonl(args.base_eval_jsonl)
    write_jsonl(args.combined_eval_jsonl, [*base_rows, *rows])
    write_audit_report(
        args.audit_report_md,
        rows=rows,
        base_rows=base_rows,
        audit=audit,
        nodes=nodes,
        output_paths={
            "corpus": args.corpus_eval_jsonl,
            "combined": args.combined_eval_jsonl,
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "corpus_eval_count": len(rows),
                "base_eval_count": len(base_rows),
                "combined_eval_count": len(base_rows) + len(rows),
                "corpus_eval_jsonl": str(args.corpus_eval_jsonl),
                "combined_eval_jsonl": str(args.combined_eval_jsonl),
                "audit_report_md": str(args.audit_report_md),
                "business_type_counts": dict(Counter(row["business_type"] for row in rows)),
                "question_style_counts": dict(Counter(row["question_style"] for row in rows)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            file_obj.write("\n")


def load_nodes(path: Path) -> dict[str, PolicyNode]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    nodes: dict[str, PolicyNode] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            node_id = str(payload.get("id_") or "").strip()
            text = str(payload.get("text") or "")
            metadata = payload.get("metadata") or {}
            if not node_id or not text.strip() or not isinstance(metadata, dict):
                raise ValueError(f"Invalid policy node at {path}:{line_no}")
            nodes[node_id] = PolicyNode(
                node_id=node_id,
                text=text,
                metadata=dict(metadata),
                fields=parse_text_fields(text),
            )
    return nodes


def parse_text_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([^:：]{1,80})[:：]\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value and key not in fields:
            fields[key] = value
    return fields


def build_corpus_eval(nodes: dict[str, PolicyNode]) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    style_plan = build_style_plan(rng)
    rows: list[dict[str, Any]] = []
    rows.extend(
        generate_policy_rule_rows(
            nodes,
            count=CORPUS_COUNTS["政策 / 待遇 / 办事规则"],
            style_picker=StylePicker(style_plan["政策 / 待遇 / 办事规则"]),
            rng=rng,
        )
    )
    rows.extend(
        generate_drug_rows(
            nodes,
            count=CORPUS_COUNTS["药品目录"],
            style_picker=StylePicker(style_plan["药品目录"]),
            rng=rng,
        )
    )
    rows.extend(
        generate_designated_rows(
            nodes,
            count=CORPUS_COUNTS["定点医疗机构 / 药店"],
            style_picker=StylePicker(style_plan["定点医疗机构 / 药店"]),
            rng=rng,
        )
    )
    rows.extend(
        generate_price_rows(
            nodes,
            count=CORPUS_COUNTS["医疗服务价格"],
            style_picker=StylePicker(style_plan["医疗服务价格"]),
            rng=rng,
        )
    )
    rows.extend(
        generate_cross_document_rows(
            nodes,
            count=CORPUS_COUNTS["跨文档组合问题"],
            style_picker=StylePicker(style_plan["跨文档组合问题"]),
            rng=rng,
        )
    )
    rows.extend(
        generate_faq_guide_rows(
            nodes,
            count=CORPUS_COUNTS["FAQ / 办事指南细节"],
            style_picker=StylePicker(style_plan["FAQ / 办事指南细节"]),
            rng=rng,
        )
    )
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 generated rows, got {len(rows)}")
    rows = uniquify_questions(rows)
    style_counts = Counter(row["question_style"] for row in rows)
    if style_counts != Counter(QUESTION_STYLE_COUNTS):
        raise RuntimeError(
            "Question style distribution drifted: "
            f"expected {dict(QUESTION_STYLE_COUNTS)}, got {dict(style_counts)}"
        )
    assigned = []
    seen_query_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        business_slug = business_slug_for(row["business_type"])
        query_id = f"corpus_{business_slug}_{index:03d}"
        if query_id in seen_query_ids:
            raise RuntimeError(f"Duplicate generated query_id {query_id}")
        seen_query_ids.add(query_id)
        output = dict(row)
        output["query_id"] = query_id
        output["case_id"] = f"corpus_{business_slug}"
        assigned.append(output)
    return assigned


class StylePicker:
    def __init__(self, styles: list[str]) -> None:
        self._remaining = Counter(styles)
        self._order = list(dict.fromkeys(styles))
        self._index = 0

    def next(self, allowed: set[str] | None = None) -> str:
        if not self._remaining:
            return "standard"
        candidates = [
            style
            for style in self._order
            if self._remaining.get(style, 0) > 0
            and (allowed is None or style in allowed)
        ]
        if not candidates:
            raise RuntimeError(
                f"No remaining question style for allowed={sorted(allowed or [])}"
            )
        max_remaining = max(self._remaining[style] for style in candidates)
        top_candidates = {
            style for style in candidates if self._remaining[style] == max_remaining
        }
        for offset in range(len(self._order)):
            style = self._order[(self._index + offset) % len(self._order)]
            if style in top_candidates:
                self._index = (self._index + offset + 1) % len(self._order)
                self._remaining[style] -= 1
                if self._remaining[style] <= 0:
                    del self._remaining[style]
                return style
        raise RuntimeError("Style picker could not select a candidate")


def uniquify_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    outputs: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        question = str(output["question"])
        if question in seen:
            suffix = question_suffix(output)
            base = question.rstrip("？?")
            question = f"{base}，在{suffix}下怎么查？"
            repeat_index = 2
            while question in seen:
                question = f"{base}，在{suffix}场景{repeat_index}下怎么查？"
                repeat_index += 1
            output["question"] = question
        seen.add(question)
        outputs.append(output)
    return outputs


def question_suffix(row: dict[str, Any]) -> str:
    filters = row.get("filters") or {}
    jurisdictions = filters.get("jurisdiction") or []
    domains = filters.get("policy_domain") or []
    jurisdiction_label = "、".join(area_name(str(item)) for item in jurisdictions) or "对应地区"
    domain_label = "、".join(domain_name(str(item)) for item in domains) or "对应政策"
    role = ""
    groups = row.get("expected_source_groups") or []
    if groups:
        roles = groups[0].get("document_roles") or []
        if roles:
            role = role_name(str(roles[0]))
    return " / ".join(part for part in [jurisdiction_label, domain_label, role] if part)


def _expanded_pool(counts: dict[str, int]) -> list[str]:
    return [style for style, count in counts.items() for _ in range(count)]


def build_style_plan(rng: random.Random) -> dict[str, list[str]]:
    remaining = Counter(QUESTION_STYLE_COUNTS)
    plan: dict[str, list[str]] = {}
    # Allocate restricted categories first so their allowed styles are reserved.
    business_order = [
        "FAQ / 办事指南细节",
        "跨文档组合问题",
        "政策 / 待遇 / 办事规则",
        "药品目录",
        "定点医疗机构 / 药店",
        "医疗服务价格",
    ]
    for business_type in business_order:
        allowed = BUSINESS_ALLOWED_STYLES[business_type]
        styles: list[str] = []
        for _ in range(CORPUS_COUNTS[business_type]):
            candidates = [
                style for style in sorted(allowed) if remaining.get(style, 0) > 0
            ]
            if not candidates:
                raise RuntimeError(f"No style quota left for {business_type}")
            style = max(
                candidates,
                key=lambda item: (
                    remaining[item] / QUESTION_STYLE_COUNTS[item],
                    remaining[item],
                    rng.random(),
                ),
            )
            remaining[style] -= 1
            if remaining[style] <= 0:
                del remaining[style]
            styles.append(style)
        rng.shuffle(styles)
        plan[business_type] = styles
    if remaining:
        raise RuntimeError(f"Unassigned style quota remains: {dict(remaining)}")
    return plan


def source_id(node: PolicyNode) -> str:
    return str(node.metadata.get("source_id") or "").strip()


def doc_id(node: PolicyNode) -> str:
    return str(node.metadata.get("doc_id") or "").strip()


def jurisdiction(node: PolicyNode) -> str:
    return str(node.metadata.get("jurisdiction") or "").strip()


def policy_domain(node: PolicyNode) -> str:
    return str(node.metadata.get("policy_domain") or "").strip()


def doc_type(node: PolicyNode) -> str:
    return str(node.metadata.get("doc_type") or "").strip()


def content_type(node: PolicyNode) -> str:
    return str(node.metadata.get("content_type") or "").strip()


def field(node: PolicyNode, *keys: str) -> str:
    for key in keys:
        value = node.fields.get(key)
        if value:
            return value.strip()
    return ""


def is_citeable(node: PolicyNode) -> bool:
    return (
        bool(source_id(node))
        and bool(node.metadata.get("source_url"))
        and node.metadata.get("can_cite_as_policy_basis") is True
    )


def group_for_node(
    node: PolicyNode,
    *,
    group: str,
    anchors: list[str],
    document_roles: list[str],
) -> dict[str, Any]:
    return {
        "group": group,
        "acceptable_sources": [source_id(node)],
        "content_anchor_sets": [anchors],
        "document_roles": document_roles,
    }


def evidence_ref_for_node(
    node: PolicyNode,
    *,
    group: str,
    anchors: list[str],
    match_scope: str,
) -> dict[str, Any]:
    return {
        "group": group,
        "node_id": node.node_id,
        "source_id": source_id(node),
        "doc_id": doc_id(node),
        "anchor": anchors,
        "match_scope": match_scope,
    }


def make_row(
    *,
    business_type: str,
    question_style: str,
    difficulty: str,
    query_intent: str,
    question: str,
    filters: dict[str, Any],
    expected_groups: list[dict[str, Any]],
    gold_refs: list[dict[str, Any]],
    evidence_level: str,
    support_scope: str,
    standard_basis: str,
    known_limitations: list[str] | None = None,
    expected_status: str = "supported",
    evidence_status: str = "full",
) -> dict[str, Any]:
    canonical_sources = sorted({str(ref["source_id"]) for ref in gold_refs})
    evidence_groups: list[dict[str, Any]] = []
    for group in expected_groups:
        group_name = str(group["group"])
        group_refs = [ref for ref in gold_refs if ref["group"] == group_name]
        evidence_groups.append(
            {
                "group": group_name,
                "status": "available",
                "matched_node_ids": sorted({str(ref["node_id"]) for ref in group_refs}),
                "matched_source_ids": sorted({str(ref["source_id"]) for ref in group_refs}),
                "matched_anchor_sets": [
                    list(anchors) for anchors in group.get("content_anchor_sets") or []
                ],
                "gold_node_ids": sorted({str(ref["node_id"]) for ref in group_refs}),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_type": "corpus_coverage",
        "case_id": "",
        "query_id": "",
        "business_type": business_type,
        "question_style": question_style,
        "difficulty": difficulty,
        "query_intent": query_intent,
        "question": question,
        "filters": filters,
        "expected_status": expected_status,
        "evidence_level": evidence_level,
        "support_scope": support_scope,
        "standard_basis": standard_basis,
        "expected_source_groups": expected_groups,
        "evidence_status": evidence_status,
        "gap_type": "none",
        "metric_eligibility": {
            "strict_retrieval": evidence_status == "full",
            "partial_retrieval": evidence_status == "partial",
            "gap_detection": False,
        },
        "evidence_groups": evidence_groups,
        "gold_evidence_refs": gold_refs,
        "canonical_source_ids": canonical_sources,
        "known_limitations": known_limitations or [],
        "standard_evidence_audit": {
            "verified": True,
            "query_evidence_status": evidence_status,
            "matched_group_count": len(expected_groups),
            "group_count": len(expected_groups),
        },
    }


def filters_for_nodes(nodes: list[PolicyNode]) -> dict[str, list[str]]:
    return {
        "jurisdiction": sorted({jurisdiction(node) for node in nodes if jurisdiction(node)}),
        "policy_domain": sorted({policy_domain(node) for node in nodes if policy_domain(node)}),
    }


POLICY_ANCHORS: dict[str, list[tuple[list[str], str]]] = {
    "remote_medical": [
        (["就医地规定的支付范围", "参保地规定的基本医疗保险基金起付标准"], "remote_policy_split"),
        (["就医地支付范围", "参保地规定的支付比例"], "remote_policy_split"),
        (["异地就医备案", "直接结算"], "remote_filing_settlement"),
        (["补办异地就医备案", "手工报销"], "remote_retroactive_filing"),
        (["急诊抢救", "备案"], "remote_emergency_filing"),
        (["普通门诊", "备案"], "remote_outpatient_filing"),
    ],
    "manual_reimbursement": [
        (["诊疗证明", "费用清单"], "manual_materials"),
        (["处方底方", "费用收据"], "manual_materials"),
        (["手工报销", "费用清单"], "manual_materials"),
        (["收据", "处方"], "manual_materials"),
    ],
    "remote_medical_manual_reimbursement": [
        (["急诊留观", "不能实现异地直接结算"], "remote_emergency_manual"),
        (["手工报销", "异地就医票据"], "remote_manual_materials"),
        (["补办备案", "手工报销"], "remote_retroactive_filing"),
    ],
    "benefit": [
        (["退休人员", "85%以上"], "beijing_employee_benefit"),
        (["退休人员", "报销80%"], "beijing_employee_benefit"),
        (["起付标准", "支付比例"], "benefit_parameters"),
        (["封顶线", "住院"], "benefit_parameters"),
    ],
    "fund_supervision": [
        (["合理、必要", "支付范围"], "fund_supervision_scope"),
        (["支付具体项目", "支付标准"], "fund_supervision_price"),
        (["已结算的医药费用", "再次纳入"], "duplicate_claim_supervision"),
        (["转卖药品"], "drug_resale_supervision"),
    ],
    "chronic_disease_long_prescription": [
        (["长处方", "慢性病患者"], "long_prescription"),
        (["处方延续服务", "高血压", "糖尿病"], "chronic_refill"),
        (["开药服务", "慢性病患者"], "chronic_refill"),
    ],
    "special_disease_filing": [
        (["备案申报表", "定点医疗机构"], "special_disease_filing"),
        (["符合我市特殊病备案条件", "备案手续"], "special_disease_filing"),
    ],
    "special_disease_scope": [
        (["门诊特殊疾病范围", "备案审核"], "special_disease_scope"),
        (["特殊病种定点医疗机构", "不纳入"], "special_disease_scope"),
    ],
    "shanghai_payment_scope": [
        (["药品目录", "基金支付范围"], "shanghai_drug_scope"),
        (["医疗服务设施", "支付范围"], "shanghai_service_scope"),
        (["医用耗材", "支付范围"], "shanghai_consumable_scope"),
        (["药品目录", "支付办法"], "shanghai_drug_scope"),
    ],
    "medical_service_price": [
        (["CT", "按部位收费"], "ct_price_rule"),
        (["出具诊断报告", "不得分解"], "ct_charge_scope"),
        (["医疗服务项目医保支付范围"], "medical_service_payment_scope"),
    ],
}


def generate_policy_rule_rows(
    nodes: dict[str, PolicyNode],
    *,
    count: int,
    style_picker: StylePicker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[tuple[PolicyNode, list[str], str]] = []
    domains = {
        "remote_medical",
        "manual_reimbursement",
        "remote_medical_manual_reimbursement",
        "benefit",
        "fund_supervision",
        "chronic_disease_long_prescription",
        "special_disease_filing",
        "special_disease_scope",
        "shanghai_payment_scope",
        "medical_service_price",
    }
    for node in nodes.values():
        if not is_citeable(node) or policy_domain(node) not in domains:
            continue
        if content_type(node) != "policy_text" and policy_domain(node) != "benefit":
            continue
        for anchors, intent in POLICY_ANCHORS.get(policy_domain(node), []):
            if anchors_hit(node.text, anchors):
                candidates.append((node, anchors, intent))
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} policy/rule candidates for {count}")
    rows: list[dict[str, Any]] = []
    for node, anchors, intent in candidates[:count]:
        style = style_picker.next()
        question = policy_question(node, anchors, style, intent)
        group = "policy_rule"
        document_roles = [role_for_node(node)]
        rows.append(
            make_row(
                business_type="政策 / 待遇 / 办事规则",
                question_style=style,
                difficulty="medium" if style in {"explanation", "condition"} else "easy",
                query_intent=intent,
                question=question,
                filters=filters_for_nodes([node]),
                expected_groups=[
                    group_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        document_roles=document_roles,
                    )
                ],
                gold_refs=[
                    evidence_ref_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        match_scope="same_clause",
                    )
                ],
                evidence_level="clause",
                support_scope="can_answer_rule",
                standard_basis=f"节点正文包含 {'、'.join(anchors)}，可作为{intent}类政策证据。",
            )
        )
    return rows


def policy_question(
    node: PolicyNode,
    anchors: list[str],
    style: str,
    intent: str,
) -> str:
    area = area_name(jurisdiction(node))
    first = anchors[0]
    second = anchors[1] if len(anchors) > 1 else "相关规则"
    domain = policy_domain(node)
    if style == "colloquial":
        if domain == "remote_medical":
            return f"异地看病时，{first}和{second}到底按哪边政策查？"
        if domain == "manual_reimbursement":
            return f"材料里提到{first}，手工报销还要一起看什么？"
        return f"{area}这条关于{first}的规则，审核时应该怎么查？"
    if style == "explanation":
        return f"为什么核验{first}时还要同时看{second}？"
    if style == "condition":
        return f"出现{first}情形时，政策上需要核验哪些适用条件？"
    if style == "material_flow":
        return f"{area}{first}相关材料或流程应如何和{second}一起核验？"
    if style == "boundary":
        return f"只看到{first}，能否不核验{second}就形成政策判断？"
    return f"{area}{first}和{second}的政策依据如何核验？"


def generate_drug_rows(
    nodes: dict[str, PolicyNode],
    *,
    count: int,
    style_picker: StylePicker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and policy_domain(node) == "drug_catalog"
        and content_type(node) == "table_row"
        and usable_name(field(node, "drug_name"))
    ]
    selected = balanced_sample(records, count, rng=rng, key=source_id)
    rows: list[dict[str, Any]] = []
    for node in selected:
        drug = field(node, "drug_name")
        insurance_class = field(node, "insurance_class")
        local_policy = field(node, "local_payment_policy")
        dosage = field(node, "dosage_form")
        anchors = [drug]
        if insurance_class and insurance_class in node.text:
            anchors.append(insurance_class)
        style = style_picker.next()
        question = drug_question(drug, insurance_class, local_policy, dosage, style)
        group = "drug_catalog_row"
        rows.append(
            make_row(
                business_type="药品目录",
                question_style=style,
                difficulty="medium" if style in {"condition", "boundary"} else "easy",
                query_intent="drug_catalog_row",
                question=question,
                filters=filters_for_nodes([node]),
                expected_groups=[
                    group_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        document_roles=["catalog_table"],
                    )
                ],
                gold_refs=[
                    evidence_ref_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        match_scope="same_row",
                    )
                ],
                evidence_level="catalog_row",
                support_scope="can_answer_catalog_row",
                standard_basis=f"目录行包含药品 {drug} 及其目录类别/剂型等字段。",
            )
        )
    return rows


def drug_question(
    drug: str,
    insurance_class: str,
    local_policy: str,
    dosage: str,
    style: str,
) -> str:
    category = local_policy or insurance_class or "目录类别"
    if style == "colloquial":
        return f"{drug}到底是不是医保目录里的药，应该查哪条目录？"
    if style == "explanation":
        return f"审核{drug}时，为什么要看医保目录类别和备注信息？"
    if style == "condition":
        return f"{drug}申报医保支付时，目录行里哪些字段需要核对？"
    if style == "boundary":
        return f"只看到处方里有{drug}，能不能不查目录就判断可报？"
    if style == "material_flow":
        return f"处方出现{drug}时，如何把药品名称和目录{category}信息对应起来？"
    dosage_hint = f"、{dosage}" if dosage else ""
    return f"{drug}在医保药品目录中的甲乙类{dosage_hint}和备注如何核验？"


def generate_designated_rows(
    nodes: dict[str, PolicyNode],
    *,
    count: int,
    style_picker: StylePicker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and policy_domain(node) == "designated_institution"
        and content_type(node) == "table_row"
        and usable_name(institution_name(node))
    ]
    selected = balanced_sample(records, count, rng=rng, key=source_id)
    rows: list[dict[str, Any]] = []
    for node in selected:
        name = institution_name(node)
        district = field(node, "district")
        code = field(node, "institution_code", "code", "机构编码")
        anchors = [name]
        if district and district in node.text:
            anchors.append(district)
        style = style_picker.next()
        group = "designated_institution_row"
        rows.append(
            make_row(
                business_type="定点医疗机构 / 药店",
                question_style=style,
                difficulty="easy",
                query_intent="designated_institution_lookup",
                question=designated_question(name, district, code, style),
                filters=filters_for_nodes([node]),
                expected_groups=[
                    group_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        document_roles=["institution_catalog"],
                    )
                ],
                gold_refs=[
                    evidence_ref_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        match_scope="same_row",
                    )
                ],
                evidence_level="catalog_row",
                support_scope="can_answer_catalog_row",
                standard_basis=f"定点机构/药店目录行包含 {name} 及地区、编码或地址字段。",
            )
        )
    return rows


def institution_name(node: PolicyNode) -> str:
    return field(
        node,
        "institution_name",
        "pharmacy_name",
        "机构名称",
        "药店名称",
        "name",
    )


def designated_question(name: str, district: str, code: str, style: str) -> str:
    district_hint = f"（{district}）" if district else ""
    if style == "colloquial":
        return f"{name}是不是医保定点，去哪里查？"
    if style == "condition":
        return f"票据机构显示{name}{district_hint}时，目录中应核对哪些定点信息？"
    if style == "explanation":
        return f"为什么审核机构{name}时要查定点机构或定点药店目录？"
    if style == "material_flow":
        return f"费用票据、处方机构和{name}目录信息如何对应核验？"
    if style == "boundary":
        return f"只看到机构名称{name}，能不能不查定点状态就继续审核？"
    code_hint = "编码、" if code else ""
    return f"{name}{district_hint}的{code_hint}地区和定点属性如何核验？"


def generate_price_rows(
    nodes: dict[str, PolicyNode],
    *,
    count: int,
    style_picker: StylePicker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and policy_domain(node) == "medical_service_price"
        and content_type(node) == "table_row"
        and usable_price_item(price_item_name(node))
    ]
    preferred = [
        node
        for node in records
        if any(token in price_item_name(node) for token in ["CT", "磁共振", "扫描", "检查", "治疗"])
    ]
    selected = balanced_sample(preferred or records, count, rng=rng, key=source_id)
    rows: list[dict[str, Any]] = []
    for node in selected:
        item = price_item_name(node)
        unit = field(node, "billing_unit", "计价单位", "单位")
        price = field(node, "price_yuan", "收费标准", "价格")
        anchors = [item]
        if unit and unit in node.text:
            anchors.append(unit)
        style = style_picker.next()
        group = "medical_service_price_row"
        rows.append(
            make_row(
                business_type="医疗服务价格",
                question_style=style,
                difficulty="medium" if style in {"condition", "boundary"} else "easy",
                query_intent="medical_service_price_lookup",
                question=price_question(item, unit, price, style),
                filters=filters_for_nodes([node]),
                expected_groups=[
                    group_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        document_roles=["price_table"],
                    )
                ],
                gold_refs=[
                    evidence_ref_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        match_scope="same_row",
                    )
                ],
                evidence_level="price_table",
                support_scope="can_answer_catalog_row",
                standard_basis=f"医疗服务价格行包含 {item} 的计价单位、价格或医保类别字段。",
            )
        )
    return rows


def price_item_name(node: PolicyNode) -> str:
    return field(node, "item_name", "项目名称", "service_name")


def price_question(item: str, unit: str, price: str, style: str) -> str:
    if style == "colloquial":
        return f"{item}这个检查收费，应该查哪个价格目录？"
    if style == "explanation":
        return f"为什么核验{item}时要看计价单位和价格项目内涵？"
    if style == "condition":
        return f"费用明细出现{item}时，应核对哪些价格目录字段？"
    if style == "material_flow":
        return f"{item}的项目名称、数量和计价单位如何和费用明细对应？"
    if style == "boundary":
        return f"只有费用里写着{item}，能不能不查价格项目就判断收费合规？"
    unit_hint = f"、计价单位{unit}" if unit else ""
    price_hint = f"、价格{price}" if price else ""
    return f"{item}的项目编码{unit_hint}{price_hint}如何核验？"


def generate_cross_document_rows(
    nodes: dict[str, PolicyNode],
    *,
    count: int,
    style_picker: StylePicker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    policy_units = [
        (node, anchors, intent)
        for node in nodes.values()
        if is_citeable(node)
        for anchors, intent in POLICY_ANCHORS.get(policy_domain(node), [])
        if anchors_hit(node.text, anchors)
    ]
    drug_nodes = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and policy_domain(node) == "drug_catalog"
        and content_type(node) == "table_row"
        and usable_name(field(node, "drug_name"))
    ]
    price_nodes = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and policy_domain(node) == "medical_service_price"
        and content_type(node) == "table_row"
        and usable_price_item(price_item_name(node))
    ]
    designated_nodes = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and policy_domain(node) == "designated_institution"
        and content_type(node) == "table_row"
        and usable_name(institution_name(node))
    ]
    fund_units = [
        unit for unit in policy_units if policy_domain(unit[0]) == "fund_supervision"
    ]
    remote_units = [
        unit for unit in policy_units if policy_domain(unit[0]) == "remote_medical"
    ]
    manual_units = [
        unit
        for unit in policy_units
        if policy_domain(unit[0]) in {"manual_reimbursement", "remote_medical_manual_reimbursement"}
    ]
    benefit_units = [unit for unit in policy_units if policy_domain(unit[0]) == "benefit"]
    chronic_units = [
        unit
        for unit in policy_units
        if policy_domain(unit[0]) in {"chronic_disease_long_prescription", "special_disease_filing"}
    ]
    pairs: list[tuple[str, PolicyNode, list[str], PolicyNode, list[str]]] = []
    pairs.extend(build_pairs("remote_benefit", remote_units, benefit_units, 6))
    pairs.extend(build_pairs("remote_manual", remote_units, manual_units, 5))
    pairs.extend(build_pairs("drug_policy", [(node, [field(node, "drug_name")], "drug") for node in drug_nodes], fund_units, 5))
    pairs.extend(build_pairs("price_supervision", [(node, [price_item_name(node)], "price") for node in price_nodes], fund_units, 4))
    pairs.extend(build_pairs("designated_supervision", [(node, [institution_name(node)], "designated") for node in designated_nodes], fund_units, 3))
    pairs.extend(build_pairs("chronic_designated", chronic_units, [(node, [institution_name(node)], "designated") for node in designated_nodes], 2))
    rng.shuffle(pairs)
    if len(pairs) < count:
        raise RuntimeError(f"Only {len(pairs)} cross-document candidates for {count}")
    rows: list[dict[str, Any]] = []
    for intent, left, left_anchors, right, right_anchors in pairs[:count]:
        style = style_picker.next({"standard", "explanation", "condition", "colloquial", "boundary"})
        question = cross_question(intent, left, left_anchors, right, right_anchors, style)
        groups = [
            group_for_node(
                left,
                group=f"{intent}_left",
                anchors=left_anchors,
                document_roles=[role_for_node(left)],
            ),
            group_for_node(
                right,
                group=f"{intent}_right",
                anchors=right_anchors,
                document_roles=[role_for_node(right)],
            ),
        ]
        refs = [
            evidence_ref_for_node(
                left,
                group=f"{intent}_left",
                anchors=left_anchors,
                match_scope=match_scope_for_node(left),
            ),
            evidence_ref_for_node(
                right,
                group=f"{intent}_right",
                anchors=right_anchors,
                match_scope=match_scope_for_node(right),
            ),
        ]
        rows.append(
            make_row(
                business_type="跨文档组合问题",
                question_style=style,
                difficulty="hard",
                query_intent=intent,
                question=question,
                filters=filters_for_nodes([left, right]),
                expected_groups=groups,
                gold_refs=refs,
                evidence_level="multi_source",
                support_scope="can_answer_with_multiple_evidence_groups",
                standard_basis="该问题需要同时召回两个以上政策或目录证据组。",
            )
        )
    return rows


def build_pairs(
    intent: str,
    left_units: list[tuple[PolicyNode, list[str], str]],
    right_units: list[tuple[PolicyNode, list[str], str]],
    limit: int,
) -> list[tuple[str, PolicyNode, list[str], PolicyNode, list[str]]]:
    pairs: list[tuple[str, PolicyNode, list[str], PolicyNode, list[str]]] = []
    if not left_units or not right_units:
        return pairs
    for index in range(limit):
        left, left_anchors, _ = left_units[index % len(left_units)]
        right, right_anchors, _ = right_units[(index * 3) % len(right_units)]
        if source_id(left) == source_id(right):
            continue
        pairs.append((intent, left, left_anchors, right, right_anchors))
    return pairs


def cross_question(
    intent: str,
    left: PolicyNode,
    left_anchors: list[str],
    right: PolicyNode,
    right_anchors: list[str],
    style: str,
) -> str:
    left_topic = left_anchors[0]
    right_topic = right_anchors[0]
    if intent == "remote_benefit":
        if style == "colloquial":
            return "人在外地看病，费用范围和报销比例到底要分别查哪类政策？"
        return f"异地就医审核中，{left_topic}和{right_topic}应如何联合核验？"
    if intent == "remote_manual":
        return f"异地就医无法直接结算时，{left_topic}和{right_topic}需要分别查哪些资料？"
    if intent == "drug_policy":
        return f"{left_topic}目录信息和基金监管中的{right_topic}要求如何一起核验？"
    if intent == "price_supervision":
        return f"{left_topic}收费项目和基金监管{right_topic}规则如何联合核验？"
    if intent == "designated_supervision":
        return f"{left_topic}定点状态和基金监管{right_topic}规则需要如何分别查？"
    return f"{left_topic}和{right_topic}两个证据组如何一起支持慢病取药核验？"


def generate_faq_guide_rows(
    nodes: dict[str, PolicyNode],
    *,
    count: int,
    style_picker: StylePicker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records = [
        node
        for node in nodes.values()
        if is_citeable(node)
        and doc_type(node) in {"faq", "service_guide", "policy_interpretation"}
        and content_type(node) == "policy_text"
    ]
    units: list[tuple[PolicyNode, list[str], str]] = []
    for node in records:
        for anchors, intent in POLICY_ANCHORS.get(policy_domain(node), []):
            if anchors_hit(node.text, anchors):
                units.append((node, anchors, intent))
        if "材料" in node.text and "报销" in node.text:
            units.append((node, ["材料", "报销"], "service_materials"))
        if "备案" in node.text and "手续" in node.text:
            units.append((node, ["备案", "手续"], "service_filing"))
    rng.shuffle(units)
    if len(units) < count:
        raise RuntimeError(f"Only {len(units)} FAQ/guide candidates for {count}")
    rows: list[dict[str, Any]] = []
    for node, anchors, intent in units[:count]:
        style = style_picker.next({"standard", "material_flow", "condition", "colloquial"})
        group = "faq_guide_detail"
        rows.append(
            make_row(
                business_type="FAQ / 办事指南细节",
                question_style=style,
                difficulty="easy",
                query_intent=intent,
                question=faq_question(node, anchors, style),
                filters=filters_for_nodes([node]),
                expected_groups=[
                    group_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        document_roles=[role_for_node(node)],
                    )
                ],
                gold_refs=[
                    evidence_ref_for_node(
                        node,
                        group=group,
                        anchors=anchors,
                        match_scope="same_clause",
                    )
                ],
                evidence_level="guide_detail",
                support_scope="can_answer_service_detail",
                standard_basis=f"FAQ/办事指南节点包含 {'、'.join(anchors)} 等办理细节。",
            )
        )
    return rows


def faq_question(node: PolicyNode, anchors: list[str], style: str) -> str:
    first = anchors[0]
    second = anchors[1] if len(anchors) > 1 else "办理要求"
    area = area_name(jurisdiction(node))
    if style == "colloquial":
        return f"{area}{first}这件事，办的时候到底要看哪些要求？"
    if style == "condition":
        return f"符合{first}情形时，办事指南里对{second}有什么要求？"
    if style == "material_flow":
        return f"{area}{first}相关材料和{second}流程如何核验？"
    return f"{area}{first}和{second}的办事指南依据在哪里？"


def anchors_hit(text: str, anchors: list[str]) -> bool:
    return bool(anchors) and all(str(anchor) in text for anchor in anchors)


def usable_name(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 2 or len(stripped) > 45:
        return False
    bad_tokens = {"标题", "合计", "nan", "None", "无"}
    return stripped not in bad_tokens


def usable_price_item(value: str) -> bool:
    stripped = value.strip()
    if not usable_name(stripped):
        return False
    return not any(token in stripped for token in ["标题", "收费", "合计", "说明"])


def balanced_sample(
    records: list[PolicyNode],
    count: int,
    *,
    rng: random.Random,
    key,
) -> list[PolicyNode]:
    by_key: dict[str, list[PolicyNode]] = defaultdict(list)
    for record in records:
        by_key[str(key(record))].append(record)
    for bucket in by_key.values():
        bucket.sort(key=lambda item: item.node_id)
    selected: list[PolicyNode] = []
    keys = sorted(by_key)
    start_positions: dict[str, list[PolicyNode]] = {}
    per_bucket = max(1, count // max(1, len(keys)) + 2)
    for bucket_key in keys:
        start_positions[bucket_key] = spread(by_key[bucket_key], per_bucket)
    round_index = 0
    while len(selected) < count:
        added = False
        for bucket_key in keys:
            bucket = start_positions[bucket_key]
            if round_index < len(bucket):
                selected.append(bucket[round_index])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        round_index += 1
    if len(selected) < count:
        leftovers = [record for record in records if record not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} records available for {count}")
    return selected[:count]


def spread(records: list[PolicyNode], count: int) -> list[PolicyNode]:
    if count >= len(records):
        return records
    if count <= 1:
        return [records[0]]
    return [records[round(index * (len(records) - 1) / (count - 1))] for index in range(count)]


def role_for_node(node: PolicyNode) -> str:
    if content_type(node) == "table_row":
        if policy_domain(node) == "drug_catalog":
            return "catalog_table"
        if policy_domain(node) == "designated_institution":
            return "institution_catalog"
        if policy_domain(node) == "medical_service_price":
            return "price_table"
        return "table_row"
    if doc_type(node) == "faq":
        return "faq"
    if doc_type(node) == "service_guide":
        return "service_guide"
    if doc_type(node) == "policy_interpretation":
        return "policy_interpretation"
    return "policy"


def match_scope_for_node(node: PolicyNode) -> str:
    return "same_row" if content_type(node) == "table_row" else "same_clause"


def area_name(value: str) -> str:
    return {
        "national": "国家",
        "beijing": "北京",
        "shanghai": "上海",
        "multi": "跨地区",
    }.get(value, value or "相关地区")


def domain_name(value: str) -> str:
    return {
        "benefit": "待遇标准",
        "chronic_disease_long_prescription": "慢病长处方",
        "designated_institution": "定点机构目录",
        "drug_catalog": "药品目录",
        "fund_supervision": "基金监管",
        "manual_reimbursement": "手工报销",
        "medical_service_price": "医疗服务价格",
        "remote_medical": "异地就医",
        "remote_medical_manual_reimbursement": "异地手工报销",
        "shanghai_payment_scope": "上海支付范围",
        "special_disease_filing": "特殊病备案",
        "special_disease_scope": "特殊病范围",
    }.get(value, value or "政策资料")


def role_name(value: str) -> str:
    return {
        "catalog_table": "目录表行",
        "faq": "FAQ",
        "institution_catalog": "机构目录行",
        "policy": "政策正文",
        "policy_interpretation": "政策解读",
        "price_table": "价格表行",
        "service_guide": "办事指南",
        "table_row": "表格行",
    }.get(value, value or "资料")


def business_slug_for(value: str) -> str:
    return {
        "政策 / 待遇 / 办事规则": "policy_rules",
        "药品目录": "drug_catalog",
        "定点医疗机构 / 药店": "designated",
        "医疗服务价格": "medical_price",
        "跨文档组合问题": "cross_document",
        "FAQ / 办事指南细节": "faq_guide",
    }[value]


def audit_rows(rows: list[dict[str, Any]], nodes: dict[str, PolicyNode]) -> dict[str, Any]:
    failures: list[str] = []
    query_ids = [row["query_id"] for row in rows]
    duplicates = [item for item, count in Counter(query_ids).items() if count > 1]
    for duplicate in duplicates:
        failures.append(f"duplicate query_id: {duplicate}")
    questions = [row["question"] for row in rows]
    duplicate_questions = [
        item for item, count in Counter(questions).items() if count > 1
    ]
    for duplicate in duplicate_questions:
        failures.append(f"duplicate question: {duplicate}")
    for row in rows:
        if not row.get("expected_source_groups"):
            failures.append(f"{row.get('query_id')}: missing expected_source_groups")
        if not row.get("gold_evidence_refs"):
            failures.append(f"{row.get('query_id')}: missing gold_evidence_refs")
        filters = row.get("filters") or {}
        for ref in row.get("gold_evidence_refs") or []:
            node = nodes.get(str(ref.get("node_id") or ""))
            if node is None:
                failures.append(f"{row.get('query_id')}: missing node {ref.get('node_id')}")
                continue
            if source_id(node) != str(ref.get("source_id") or ""):
                failures.append(f"{row.get('query_id')}: source mismatch for {node.node_id}")
            anchors = [str(anchor) for anchor in ref.get("anchor") or []]
            if not anchors_hit(node.text, anchors):
                failures.append(f"{row.get('query_id')}: anchor miss {anchors} in {node.node_id}")
            if not node.metadata.get("source_url"):
                failures.append(f"{row.get('query_id')}: missing source_url for {node.node_id}")
            if not filter_match(node, filters):
                failures.append(f"{row.get('query_id')}: filters do not match {node.node_id}")
            if row.get("evidence_status") == "full" and not is_citeable(node):
                failures.append(f"{row.get('query_id')}: full evidence node not citeable {node.node_id}")
    return {
        "failure_count": len(failures),
        "failures": failures,
        "business_type_counts": Counter(row["business_type"] for row in rows),
        "question_style_counts": Counter(row["question_style"] for row in rows),
        "difficulty_counts": Counter(row["difficulty"] for row in rows),
        "source_counts": Counter(
            ref["source_id"]
            for row in rows
            for ref in row.get("gold_evidence_refs") or []
        ),
    }


def filter_match(node: PolicyNode, filters: dict[str, Any]) -> bool:
    jurisdictions = [str(item) for item in filters.get("jurisdiction") or []]
    domains = [str(item) for item in filters.get("policy_domain") or []]
    return (
        (not jurisdictions or jurisdiction(node) in jurisdictions)
        and (not domains or policy_domain(node) in domains)
    )


def write_audit_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    nodes: dict[str, PolicyNode],
    output_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    node_domain_counts = Counter(policy_domain(node) for node in nodes.values())
    node_source_counts = Counter(source_id(node) for node in nodes.values())
    lines = [
        "# Policy RAG Corpus Eval v1.3 Standard Audit",
        "",
        "This audit verifies generated corpus-driven eval records against actual policy nodes.",
        "",
        "## Outputs",
        "",
        f"- Corpus eval set: `{output_paths['corpus']}`",
        f"- Combined eval set: `{output_paths['combined']}`",
        "",
        "## Counts",
        "",
        f"- Existing closed-loop eval rows: {len(base_rows)}",
        f"- New corpus eval rows: {len(rows)}",
        f"- Combined eval rows: {len(base_rows) + len(rows)}",
        f"- Audit failures: {audit['failure_count']}",
        "",
        "## Business Type Distribution",
        "",
        "| Business type | Count |",
        "|---|---:|",
    ]
    for name, expected in CORPUS_COUNTS.items():
        lines.append(f"| {name} | {audit['business_type_counts'].get(name, 0)} |")
    lines.extend(
        [
            "",
            "## Question Style Distribution",
            "",
            "| Question style | Count |",
            "|---|---:|",
        ]
    )
    for name, expected in QUESTION_STYLE_COUNTS.items():
        lines.append(f"| {name} | {audit['question_style_counts'].get(name, 0)} |")
    lines.extend(
        [
            "",
            "## Top Evidence Sources",
            "",
            "| Source ID | Gold ref count | Corpus node count |",
            "|---|---:|---:|",
        ]
    )
    for source, count in audit["source_counts"].most_common(30):
        lines.append(f"| `{source}` | {count} | {node_source_counts.get(source, 0)} |")
    lines.extend(
        [
            "",
            "## Corpus Node Domain Distribution",
            "",
            "| Policy domain | Node count |",
            "|---|---:|",
        ]
    )
    for domain, count in node_domain_counts.most_common():
        lines.append(f"| `{domain}` | {count} |")
    if audit["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in audit["failures"])
    lines.extend(
        [
            "",
            "## Sample Rows",
            "",
            "| Query ID | Business type | Style | Question |",
            "|---|---|---|---|",
        ]
    )
    for row in rows[:20]:
        question = str(row["question"]).replace("|", "\\|")
        lines.append(
            f"| `{row['query_id']}` | {row['business_type']} | "
            f"`{row['question_style']}` | {question} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
