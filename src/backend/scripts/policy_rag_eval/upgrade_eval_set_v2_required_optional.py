"""Upgrade a policy RAG v2.1 audited goldset to required/optional v2.2.

The v2.1 goldset treated every answer point as required. That is too strict
for generation metrics when a generated reference answer contains useful but
unasked process details. This migration preserves retrieval evidence groups
and adds a generation-specific answer contract:

- required_answer_points: facts the question must answer
- optional_answer_points: allowed facts that should not create FN when omitted
- reference_answer_required: concise reference answer for generation scoring
- reference_answer_full: the original full reference answer for review
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_EVAL_ROOT,
    DEFAULT_REPORT_ROOT,
    SCHEMA_EVAL_V2_2,
    read_jsonl,
    write_jsonl,
)


DEFAULT_SOURCE_JSONL = (
    DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_1_entity_fixed_from300_audited.jsonl"
)
DEFAULT_OUTPUT_JSONL = (
    DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_2_required_optional_audited.jsonl"
)
DEFAULT_REPORT_MD = (
    DEFAULT_REPORT_ROOT / "policy_rag_eval_set_v2_2_required_optional_report.md"
)


TABLE_FIELD_SPECS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "drug_or_project_category",
        ("医保类别", "报销类别", "药品类别", "医保项目", "医保报销类别"),
        ("医保类别", "报销类别", "乙类", "甲类", "非医保项目", "医保项目"),
    ),
    (
        "catalog_no",
        ("目录编号", "目录号"),
        ("目录编号", "目录号"),
    ),
    (
        "local_payment_ratio",
        (
            "本地支付比例",
            "支付比例",
            "先行支付",
            "自付比例",
            "个人先自付",
            "个人自负比例",
            "个人自付比例",
        ),
        (
            "本地支付比例",
            "支付比例",
            "先行支付",
            "自付比例",
            "个人先行支付",
            "个人自负比例",
            "个人自付比例",
        ),
    ),
    (
        "classification",
        ("药品分类", "分类代码", "分类名称", "分类"),
        ("分类代码", "分类名称", "分类为", "分类"),
    ),
    (
        "institution_code",
        ("医保编码", "机构编码", "机构代码"),
        ("医保机构编码", "机构编码", "机构代码", "编码为"),
    ),
    (
        "district",
        ("所在区", "所属区", "哪个区", "位于哪个区"),
        ("所属区", "位于"),
    ),
    (
        "address",
        ("地址",),
        ("地址",),
    ),
    (
        "institution_type",
        ("机构类型", "类型"),
        ("机构类型", "类型为"),
    ),
    (
        "institution_level",
        ("级别", "等级"),
        ("级别", "等级"),
    ),
    (
        "designated_status",
        ("是否为", "是否均为", "是否在列", "是否在医保定点", "定点"),
        ("定点", "名单", "名录", "出现在", "在列"),
    ),
    (
        "unit",
        ("计价单位",),
        ("计价单位",),
    ),
    (
        "price",
        ("收费标准", "单价", "价格", "医保支付标准"),
        ("收费标准", "单价", "价格", "医保支付标准", "元"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade policy_rag_eval v2.1 audited goldset to v2.2 required/optional schema."
    )
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source_jsonl)
    upgraded_rows: list[dict[str, Any]] = []
    migration_results: list[dict[str, Any]] = []

    for row in source_rows:
        upgraded, result = upgrade_row(row)
        upgraded_rows.append(upgraded)
        migration_results.append(result)

    write_jsonl(args.output_jsonl, upgraded_rows)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        build_report(
            source_path=args.source_jsonl,
            output_path=args.output_jsonl,
            rows=upgraded_rows,
            migration_results=migration_results,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "source_count": len(source_rows),
                "output_count": len(upgraded_rows),
                "needs_review_count": sum(
                    1
                    for row in upgraded_rows
                    if row.get("audit_status") == "needs_review"
                ),
                "avg_required_points": average_count(
                    row.get("required_answer_points") for row in upgraded_rows
                ),
                "avg_optional_points": average_count(
                    row.get("optional_answer_points") for row in upgraded_rows
                ),
                "output_jsonl": str(args.output_jsonl),
                "report_md": str(args.report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def upgrade_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original_answer_points = [
        point for point in row.get("answer_points") or [] if isinstance(point, dict)
    ]
    question = str(row.get("question") or "")
    groups_by_point = evidence_group_ids_by_answer_point(row)

    required_items: list[dict[str, Any]] = []
    optional_items: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for point in original_answer_points:
        source_point_id = str(point.get("point_id") or "")
        group_ids = groups_by_point.get(source_point_id, [])
        for item in classify_answer_point(row, point, group_ids=group_ids):
            if item["required"]:
                required_items.append(item)
            else:
                optional_items.append(item)
            reasons[str(item.get("classification_reason") or "unknown")] += 1

    required_items, optional_items = apply_post_classification_rules(
        question,
        required_items=required_items,
        optional_items=optional_items,
    )

    required_points = finalize_metric_points(required_items, prefix="rp")
    optional_points = finalize_metric_points(optional_items, prefix="op")
    needs_review = not required_points
    if needs_review:
        required_points = finalize_metric_points(
            [
                build_metric_point(
                    source_point=point,
                    statement=str(point.get("statement") or ""),
                    evidence_group_ids=groups_by_point.get(str(point.get("point_id") or ""), []),
                    required=True,
                    reason="fallback_no_required_points",
                )
                for point in original_answer_points
            ],
            prefix="rp",
        )
        reasons["fallback_no_required_points"] += len(required_points)

    upgraded = dict(row)
    upgraded["schema_version"] = SCHEMA_EVAL_V2_2
    upgraded["reference_answer_full"] = str(row.get("reference_answer") or "")
    upgraded["required_answer_points"] = required_points
    upgraded["optional_answer_points"] = optional_points
    upgraded["reference_answer_required"] = compose_reference_answer(required_points)
    if needs_review:
        upgraded["audit_status"] = "needs_review"
    upgraded["required_optional_audit"] = {
        "status": "needs_review" if needs_review else "accepted",
        "source_schema_version": row.get("schema_version"),
        "source_answer_point_count": len(original_answer_points),
        "required_answer_point_count": len(required_points),
        "optional_answer_point_count": len(optional_points),
        "classification_reason_counts": dict(sorted(reasons.items())),
        "migration_rule_version": "required_optional_heuristic_v1",
    }
    return upgraded, upgraded["required_optional_audit"]


def classify_answer_point(
    row: dict[str, Any],
    point: dict[str, Any],
    *,
    group_ids: list[str],
) -> list[dict[str, Any]]:
    question = str(row.get("question") or "")
    statement = str(point.get("statement") or "").strip()
    if not statement:
        return []

    processing_parts = split_processing_time_statement(
        question,
        point,
        statement=statement,
        group_ids=group_ids,
    )
    if processing_parts is not None:
        return processing_parts

    if asks_required_materials(question) and contains_any(statement, ("非必要",)):
        return [
            build_metric_point(
                source_point=point,
                statement=statement,
                evidence_group_ids=group_ids,
                required=False,
                reason="unasked_non_required_materials",
            )
        ]

    if contains_onsite_count(statement) and not asks_onsite_count(question):
        return [
            build_metric_point(
                source_point=point,
                statement=statement,
                evidence_group_ids=group_ids,
                required=False,
                reason="unasked_onsite_count",
            )
        ]

    table_result = classify_table_answer_point(row, point, group_ids=group_ids)
    if table_result is not None:
        return [table_result]

    return [
        build_metric_point(
            source_point=point,
            statement=statement,
            evidence_group_ids=group_ids,
            required=True,
            reason="direct_question_scope",
        )
    ]


def split_processing_time_statement(
    question: str,
    point: dict[str, Any],
    *,
    statement: str,
    group_ids: list[str],
) -> list[dict[str, Any]] | None:
    has_deadline = contains_deadline(statement)
    has_process_step_time = contains_process_step_time(statement)
    if not has_deadline and not has_process_step_time:
        return None

    parts: list[dict[str, Any]] = []
    if has_deadline:
        deadline_statement = extract_deadline_statement(statement)
        parts.append(
            build_metric_point(
                source_point=point,
                statement=deadline_statement,
                evidence_group_ids=group_ids,
                required=asks_deadline(question),
                reason=(
                    "asked_deadline"
                    if asks_deadline(question)
                    else "unasked_deadline_detail"
                ),
            )
        )
    if has_process_step_time:
        process_statement = extract_process_step_time_statement(statement)
        parts.append(
            build_metric_point(
                source_point=point,
                statement=process_statement,
                evidence_group_ids=group_ids,
                required=asks_process_steps(question),
                reason=(
                    "asked_process_step_time"
                    if asks_process_steps(question)
                    else "unasked_process_step_time"
                ),
            )
        )
    return parts or None


def classify_table_answer_point(
    row: dict[str, Any],
    point: dict[str, Any],
    *,
    group_ids: list[str],
) -> dict[str, Any] | None:
    question = str(row.get("question") or "")
    statement = str(point.get("statement") or "").strip()
    if not is_table_like_row(row):
        return None
    requested_fields = requested_table_fields(question)
    if not requested_fields:
        return None
    matching_fields = [
        field
        for field in requested_fields
        if table_statement_matches_field(field, statement)
    ]
    required = bool(matching_fields)
    return build_metric_point(
        source_point=point,
        statement=statement,
        evidence_group_ids=group_ids,
        required=required,
        reason=(
            "asked_table_field:" + ",".join(matching_fields)
            if required
            else "unasked_table_field"
        ),
    )


def apply_post_classification_rules(
    question: str,
    *,
    required_items: list[dict[str, Any]],
    optional_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not asks_single_deadline_value(question):
        return required_items, optional_items

    deadline_items = [
        item
        for item in required_items
        if contains_deadline(str(item.get("statement") or ""))
    ]
    if len(deadline_items) <= 1:
        return required_items, optional_items

    preferred = sorted(
        deadline_items,
        key=lambda item: (
            0 if "法定" in str(item.get("statement") or "") else 1,
            required_items.index(item),
        ),
    )[0]
    new_required: list[dict[str, Any]] = []
    for item in required_items:
        if item is preferred or item not in deadline_items:
            new_required.append(item)
            continue
        demoted = dict(item)
        demoted["required"] = False
        demoted["classification_reason"] = "duplicate_or_supporting_deadline_detail"
        optional_items.append(demoted)
    return new_required, optional_items


def build_metric_point(
    *,
    source_point: dict[str, Any],
    statement: str,
    evidence_group_ids: list[str],
    required: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "statement": clean_sentence(statement),
        "required": required,
        "evidence_group_ids": list(dict.fromkeys(evidence_group_ids)),
        "source_answer_point_ids": [
            str(source_point.get("point_id") or "")
        ]
        if source_point.get("point_id")
        else [],
        "classification_reason": reason,
    }


def finalize_metric_points(
    items: list[dict[str, Any]],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        statement = clean_sentence(item.get("statement"))
        if not statement or statement in seen:
            continue
        seen.add(statement)
        point = dict(item)
        point["point_id"] = f"{prefix}{len(points) + 1}"
        point["statement"] = statement
        points.append(point)
    return points


def evidence_group_ids_by_answer_point(row: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for group in row.get("gold_evidence_groups") or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "")
        if not group_id:
            continue
        for point_id in group.get("answer_point_ids") or []:
            if str(point_id):
                mapping[str(point_id)].append(group_id)
    return {key: list(dict.fromkeys(value)) for key, value in mapping.items()}


def requested_table_fields(question: str) -> list[str]:
    requested: list[str] = []
    for field, question_terms, _statement_terms in TABLE_FIELD_SPECS:
        if contains_any(question, question_terms):
            requested.append(field)
    return list(dict.fromkeys(requested))


def table_statement_matches_field(field: str, statement: str) -> bool:
    for spec_field, _question_terms, statement_terms in TABLE_FIELD_SPECS:
        if field == spec_field:
            return contains_any(statement, statement_terms)
    return False


def is_table_like_row(row: dict[str, Any]) -> bool:
    return str(row.get("question_type") or "") == "table_lookup" or str(
        row.get("bundle_type") or ""
    ) == "table_lookup"


def asks_required_materials(question: str) -> bool:
    return contains_any(question, ("必要材料", "哪些必要", "应要求提供哪些"))


def asks_deadline(question: str) -> bool:
    return contains_any(
        question,
        (
            "办结时限",
            "办理时限",
            "审批时限",
            "时限是多少",
            "时限是多久",
            "审批多久",
            "多少个工作日",
            "办理期限",
        ),
    )


def asks_single_deadline_value(question: str) -> bool:
    if not asks_deadline(question):
        return False
    return not contains_any(
        question,
        ("审查", "决定", "环节", "流程", "到现场", "现场次数", "起算", "从何时"),
    )


def asks_process_steps(question: str) -> bool:
    return contains_any(question, ("审查", "决定", "办理流程", "流程", "环节"))


def asks_onsite_count(question: str) -> bool:
    return contains_any(question, ("到现场", "现场次数", "到场", "次数"))


def contains_deadline(statement: str) -> bool:
    return contains_any(statement, ("法定办结时限", "承诺办结时限", "办结时限"))


def contains_process_step_time(statement: str) -> bool:
    return contains_any(statement, ("审查", "决定")) and contains_any(
        statement,
        ("15个工作日", "15 个工作日", "15工作日", "环节时限"),
    )


def contains_onsite_count(statement: str) -> bool:
    return contains_any(statement, ("到现场", "现场次数", "需到现场"))


def extract_deadline_statement(statement: str) -> str:
    legal_match = re.search(r"法定办结时限[为是]?\s*\d+\s*个?\(?工作日\)?", statement)
    if legal_match:
        return legal_match.group(0)
    promise_match = re.search(r"承诺办结时限[为是]?\s*\d+\s*个?\(?工作日\)?", statement)
    if promise_match:
        return promise_match.group(0)
    return statement


def extract_process_step_time_statement(statement: str) -> str:
    combined_match = re.search(r"审查和决定环节时限各为\s*\d+\s*个?工作日", statement)
    if combined_match:
        return combined_match.group(0)
    return statement


def clean_sentence(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("；;。 ")
    return text


def compose_reference_answer(required_points: list[dict[str, Any]]) -> str:
    statements = [clean_sentence(point.get("statement")) for point in required_points]
    statements = [statement for statement in statements if statement]
    if not statements:
        return ""
    return "；".join(statements) + "。"


def contains_any(value: str, terms: tuple[str, ...]) -> bool:
    text = str(value or "")
    return any(term in text for term in terms)


def average_count(values: Any) -> float | None:
    counts = [len(value or []) for value in values]
    return sum(counts) / len(counts) if counts else None


def build_report(
    *,
    source_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    migration_results: list[dict[str, Any]],
) -> str:
    reason_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for result in migration_results:
        status_counts[str(result.get("status") or "unknown")] += 1
        reason_counts.update(result.get("classification_reason_counts") or {})

    changed_rows = [row for row in rows if row.get("optional_answer_points")]
    lines = [
        "# Policy RAG Eval Set v2.2 Required/Optional Migration",
        "",
        f"- Source: `{source_path}`",
        f"- Output: `{output_path}`",
        f"- Row count: {len(rows)}",
        f"- Status counts: `{dict(sorted(status_counts.items()))}`",
        f"- Average required points: {fmt(average_count(row.get('required_answer_points') for row in rows))}",
        f"- Average optional points: {fmt(average_count(row.get('optional_answer_points') for row in rows))}",
        f"- Rows with optional points: {sum(1 for row in rows if row.get('optional_answer_points'))}",
        "",
        "## Classification Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## Rows With Optional Points", ""])
    for row in changed_rows[:20]:
        lines.extend(
            [
                f"### {row.get('query_id')}",
                "",
                f"- Question: {row.get('question')}",
                f"- Required: {len(row.get('required_answer_points') or [])}",
                f"- Optional: {len(row.get('optional_answer_points') or [])}",
                f"- Reference required: {row.get('reference_answer_required')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
