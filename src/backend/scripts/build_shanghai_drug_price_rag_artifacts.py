"""Build row-level RAG chunks and TextNodes for Shanghai drug price rows.

This script stops at reference-data RAG artifacts. It does not embed, build
FAISS, switch the MCP runtime, integrate Case Agent, or make audit decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"
DEFAULT_REFERENCE_DIR = (
    DEFAULT_RAG_READY_ROOT / "reference" / "drug_product_price_reference"
)
DEFAULT_SNAPSHOT = "20260820_full"
DEFAULT_CLEAN_INPUT = (
    DEFAULT_REFERENCE_DIR / f"shanghai_drug_price_clean_{DEFAULT_SNAPSHOT}.jsonl"
)
DEFAULT_CHUNKS_OUTPUT = (
    DEFAULT_RAG_READY_ROOT
    / "chunks"
    / f"shanghai_drug_price_chunks_{DEFAULT_SNAPSHOT}.jsonl"
)
DEFAULT_NODES_OUTPUT = (
    DEFAULT_RAG_READY_ROOT
    / "nodes"
    / f"shanghai_drug_price_nodes_{DEFAULT_SNAPSHOT}.jsonl"
)
DEFAULT_BASE_NODES = DEFAULT_RAG_READY_ROOT / "nodes" / "policy_nodes.jsonl"
DEFAULT_COMBINED_NODES = (
    DEFAULT_RAG_READY_ROOT
    / "nodes"
    / f"policy_nodes_with_shanghai_drug_price_{DEFAULT_SNAPSHOT}.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / f"shanghai_drug_price_rag_artifacts_{DEFAULT_SNAPSHOT}.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / f"shanghai_drug_price_rag_artifacts_{DEFAULT_SNAPSHOT}.md"
)

CHUNK_SCHEMA_VERSION = "shanghai_drug_price_chunk_v0.1"
NODE_SCHEMA_VERSION = "policy_text_node_v0.1"
SOURCE_ID_PREFIX = "reference_shanghai_drug_product_price"

EXCLUDED_EMBED_METADATA_KEYS = {
    "avg_price",
    "chunk_id",
    "chunk_index",
    "fetched_at",
    "package_quantity",
    "pharmacy_count",
    "price_period_end",
    "price_period_start",
    "price_range_max",
    "price_range_min",
    "sales_count",
    "source_path",
    "source_row_number",
    "source_sequence",
    "source_url",
    "text_chars",
}
EXCLUDED_LLM_METADATA_KEYS = {
    "chunk_strategy",
    "fetched_at",
    "source_path",
    "source_row_number",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-input", type=Path, default=DEFAULT_CLEAN_INPUT)
    parser.add_argument("--chunks-output", type=Path, default=DEFAULT_CHUNKS_OUTPUT)
    parser.add_argument("--nodes-output", type=Path, default=DEFAULT_NODES_OUTPUT)
    parser.add_argument("--base-nodes", type=Path, default=DEFAULT_BASE_NODES)
    parser.add_argument("--combined-nodes", type=Path, default=DEFAULT_COMBINED_NODES)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--skip-combined-nodes",
        action="store_true",
        help="Only write drug chunks/nodes; do not append them to base policy nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_rows = load_jsonl(args.clean_input)
    chunks, chunk_issues = build_chunks(
        clean_rows,
        clean_input=args.clean_input,
        snapshot=args.snapshot,
    )
    nodes = [chunk_to_text_node(chunk) for chunk in chunks]
    write_jsonl(args.chunks_output, chunks)
    write_nodes(args.nodes_output, nodes)

    combined_report: dict[str, Any] = {"written": False}
    if not args.skip_combined_nodes:
        combined_report = write_combined_nodes(
            base_nodes=args.base_nodes,
            drug_nodes=args.nodes_output,
            combined_nodes=args.combined_nodes,
        )

    report = build_report(
        args=args,
        clean_rows=clean_rows,
        chunks=chunks,
        nodes=nodes,
        chunk_issues=chunk_issues,
        combined_report=combined_report,
    )
    write_report(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": "ok" if report["validation"]["is_valid"] else "invalid",
                "clean_count": len(clean_rows),
                "chunk_count": len(chunks),
                "node_count": len(nodes),
                "combined_node_count": combined_report.get("combined_node_count"),
                "chunks_output": str(args.chunks_output),
                "nodes_output": str(args.nodes_output),
                "combined_nodes": str(args.combined_nodes)
                if not args.skip_combined_nodes
                else None,
                "report_json": str(args.report_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def write_nodes(path: Path, nodes: list[TextNode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for node in nodes:
            file_obj.write(json.dumps(node.to_dict(), ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def build_chunks(
    rows: list[dict[str, Any]],
    *,
    clean_input: Path,
    snapshot: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    chunks: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    seen_ids: set[str] = set()
    source_id = f"{SOURCE_ID_PREFIX}_{snapshot}"

    for row_number, row in enumerate(rows, start=1):
        chunk_id = str(row.get("record_id") or "").strip()
        if not chunk_id:
            issues["missing_record_id"] += 1
            continue
        if chunk_id in seen_ids:
            issues["duplicate_record_id"] += 1
            continue
        seen_ids.add(chunk_id)
        if not text_value(row, "drug_name"):
            issues["missing_drug_name"] += 1
            continue
        if not any(row.get(key) is not None for key in ("avg_price", "price_range_min", "price_range_max")):
            issues["missing_price"] += 1
            continue

        text = build_chunk_text(row)
        chunk = {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "source_id": source_id,
            "title": build_title(row),
            "jurisdiction": "shanghai",
            "policy_domain": "drug_product_price_reference",
            "legal_weight": "reference",
            "rag_ingestion_role": "drug_price_reference",
            "content_type": "table_row",
            "resource_type": "jsonl_table",
            "evidence_role": "reference_evidence",
            "can_cite_as_policy_basis": False,
            "source_platform": row.get("source_platform") or "国家医保开放平台上海专区",
            "source_url": row.get("source_url"),
            "source_service_code": row.get("source_service_code") or "YBDRUG001",
            "source_path": str(clean_input),
            "source_row_number": row_number,
            "chunk_strategy": "drug_product_price_row_v0.1",
            "chunk_profile": "drug_product_price_row",
            "chunk_size": 1,
            "chunk_overlap": 0,
            "chunk_hard_max": 1,
            "chunk_unit": "row",
            "split_method": "one_clean_row_one_chunk",
            "doc_id": source_id,
            "doc_type": "reference",
            "chunk_id": chunk_id,
            "chunk_index": row_number,
            "source_sequence": row_number,
            "section_heading": "上海医保药店药品产品价格参考",
            "page_refs": [],
            "query_keyword": row.get("query_keyword"),
            "fetched_at": row.get("fetched_at"),
            "price_period_label": row.get("price_period_label") or "上周",
            "price_period_start": row.get("price_period_start"),
            "price_period_end": row.get("price_period_end"),
            "price_region": row.get("price_region") or "上海",
            "drug_code": row.get("drug_code"),
            "drug_name": row.get("drug_name"),
            "registered_name": row.get("registered_name"),
            "trade_name": row.get("trade_name"),
            "manufacturer": row.get("manufacturer"),
            "specification": row.get("specification"),
            "dosage_form": row.get("dosage_form"),
            "package_unit": row.get("package_unit"),
            "package_quantity": row.get("package_quantity"),
            "drug_type": row.get("drug_type"),
            "avg_price": row.get("avg_price"),
            "price_range_min": row.get("price_range_min"),
            "price_range_max": row.get("price_range_max"),
            "price_unit": row.get("price_unit") or "元",
            "sales_count": row.get("sales_count"),
            "pharmacy_count": row.get("pharmacy_count"),
            "text": text,
            "text_chars": len(text),
        }
        chunks.append(chunk)

    return chunks, issues


def build_chunk_text(row: dict[str, Any]) -> str:
    period = format_period(row)
    price_text = format_price(row)
    product_parts = [
        f"药品名称：{text_value(row, 'drug_name')}",
        f"注册名称：{text_value(row, 'registered_name')}",
    ]
    if text_value(row, "trade_name"):
        product_parts.append(f"商品名：{text_value(row, 'trade_name')}")
    if text_value(row, "drug_code"):
        product_parts.append(f"药品编码：{text_value(row, 'drug_code')}")
    product_parts.extend(
        [
            f"生产企业：{text_value(row, 'manufacturer')}",
            f"规格：{text_value(row, 'specification')}",
            f"剂型：{text_value(row, 'dosage_form')}",
            f"最小包装单位：{text_value(row, 'package_unit')}",
            f"最小包装数量：{text_value(row, 'package_quantity')}",
            f"药品类型：{text_value(row, 'drug_type')}",
        ]
    )
    stats_parts = [
        f"价格地区：{text_value(row, 'price_region') or '上海'}",
        f"价格周期：{period}",
        price_text,
    ]
    if row.get("sales_count") is not None:
        stats_parts.append(f"结算销量参考：{format_number(row.get('sales_count'))}")
    if row.get("pharmacy_count") is not None:
        stats_parts.append(f"涉及医保药店数量：{format_number(row.get('pharmacy_count'))}")
    stats_parts.append("数据用途：药品产品价格参考，不作为医保报销政策依据")
    return "上海药品产品价格参考。 " + "；".join(product_parts + stats_parts) + "。"


def build_title(row: dict[str, Any]) -> str:
    name = text_value(row, "drug_name")
    manufacturer = text_value(row, "manufacturer")
    spec = text_value(row, "specification")
    period_end = text_value(row, "price_period_end")
    return f"上海药品价格参考：{name} {manufacturer} {spec}（截至{period_end}）"


def format_period(row: dict[str, Any]) -> str:
    start = text_value(row, "price_period_start")
    end = text_value(row, "price_period_end")
    label = text_value(row, "price_period_label") or "上周"
    if start and end:
        return f"{label}，{start}至{end}"
    return label


def format_price(row: dict[str, Any]) -> str:
    unit = text_value(row, "price_unit") or "元"
    avg = row.get("avg_price")
    min_price = row.get("price_range_min")
    max_price = row.get("price_range_max")
    parts: list[str] = []
    if avg is not None:
        parts.append(f"医保药店周均价：{format_number(avg)}{unit}")
    if min_price is not None or max_price is not None:
        parts.append(
            f"医保药店周价格区间：{format_number(min_price)}至{format_number(max_price)}{unit}"
        )
    return "，".join(parts) if parts else "未提供价格"


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def text_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def chunk_to_text_node(chunk: dict[str, Any]) -> TextNode:
    metadata: dict[str, Any] = {
        "node_schema_version": NODE_SCHEMA_VERSION,
        "evidence_role": "reference_evidence",
        "can_cite_as_policy_basis": False,
    }
    for key, value in chunk.items():
        if key == "text" or value in (None, ""):
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return TextNode(
        id_=str(chunk["chunk_id"]),
        text=str(chunk["text"]),
        metadata=metadata,
        excluded_embed_metadata_keys=sorted(
            key for key in metadata if key in EXCLUDED_EMBED_METADATA_KEYS
        ),
        excluded_llm_metadata_keys=sorted(
            key for key in metadata if key in EXCLUDED_LLM_METADATA_KEYS
        ),
    )


def write_combined_nodes(
    *,
    base_nodes: Path,
    drug_nodes: Path,
    combined_nodes: Path,
) -> dict[str, Any]:
    base_records = read_raw_jsonl(base_nodes)
    drug_records = read_raw_jsonl(drug_nodes)
    base_ids = [str(record.get("id_") or "") for record in base_records]
    drug_ids = [str(record.get("id_") or "") for record in drug_records]
    duplicate_ids = sorted(set(base_ids).intersection(drug_ids))
    if duplicate_ids:
        raise RuntimeError(f"Drug node IDs already exist in base nodes: {duplicate_ids[:5]}")
    combined_nodes.parent.mkdir(parents=True, exist_ok=True)
    with combined_nodes.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in base_records + drug_records:
            file_obj.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")
    return {
        "written": True,
        "base_nodes": str(base_nodes),
        "drug_nodes": str(drug_nodes),
        "combined_nodes": str(combined_nodes),
        "base_node_count": len(base_records),
        "drug_node_count": len(drug_records),
        "combined_node_count": len(base_records) + len(drug_records),
        "duplicate_ids": duplicate_ids,
    }


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    return records


def build_report(
    *,
    args: argparse.Namespace,
    clean_rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    nodes: list[TextNode],
    chunk_issues: Counter[str],
    combined_report: dict[str, Any],
) -> dict[str, Any]:
    duplicate_chunk_ids = [
        item for item, count in Counter(chunk["chunk_id"] for chunk in chunks).items() if count > 1
    ]
    missing_source_url = [
        chunk["chunk_id"] for chunk in chunks if not str(chunk.get("source_url") or "").startswith("http")
    ]
    can_cite_errors = [
        chunk["chunk_id"] for chunk in chunks if chunk.get("can_cite_as_policy_basis") is not False
    ]
    validation = {
        "is_valid": (
            len(clean_rows) == len(chunks) == len(nodes)
            and not duplicate_chunk_ids
            and not missing_source_url
            and not can_cite_errors
            and not chunk_issues
        ),
        "count_matches_clean": len(clean_rows) == len(chunks) == len(nodes),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "missing_source_url": missing_source_url,
        "can_cite_as_policy_basis_errors": can_cite_errors,
        "chunk_issues": dict(chunk_issues),
    }
    prices = [
        float(chunk["avg_price"])
        for chunk in chunks
        if isinstance(chunk.get("avg_price"), (int, float))
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "ok" if validation["is_valid"] else "invalid",
        "clean_input": str(args.clean_input),
        "chunks_output": str(args.chunks_output),
        "nodes_output": str(args.nodes_output),
        "combined_nodes": str(args.combined_nodes)
        if not args.skip_combined_nodes
        else None,
        "clean_count": len(clean_rows),
        "chunk_count": len(chunks),
        "node_count": len(nodes),
        "by_drug_name_count": len(set(str(chunk.get("drug_name") or "") for chunk in chunks)),
        "by_registered_name_count": len(
            set(str(chunk.get("registered_name") or "") for chunk in chunks)
        ),
        "by_manufacturer_count": len(
            set(str(chunk.get("manufacturer") or "") for chunk in chunks)
        ),
        "by_policy_domain": dict(Counter(str(chunk.get("policy_domain") or "") for chunk in chunks)),
        "by_jurisdiction": dict(Counter(str(chunk.get("jurisdiction") or "") for chunk in chunks)),
        "by_evidence_role": dict(Counter(str(chunk.get("evidence_role") or "") for chunk in chunks)),
        "top_query_keywords": Counter(str(chunk.get("query_keyword") or "") for chunk in chunks).most_common(20),
        "price_periods": sorted(
            {
                f"{chunk.get('price_period_start')}..{chunk.get('price_period_end')}"
                for chunk in chunks
            }
        ),
        "avg_price_stats": numeric_stats(prices),
        "text_length_stats": numeric_stats([len(node.text) for node in nodes]),
        "combined_nodes_report": combined_report,
        "validation": validation,
        "boundary": {
            "builds_chunks": True,
            "builds_nodes": True,
            "embeds": False,
            "builds_faiss": False,
            "switches_mcp_default_index": False,
            "can_cite_as_policy_basis": False,
        },
    }


def numeric_stats(values: list[float | int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
    }


def write_report(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    validation = report["validation"]
    combined = report.get("combined_nodes_report") or {}
    return "\n".join(
        [
            "# Shanghai Drug Price RAG Artifact Report",
            "",
            f"Updated at: {report['generated_at']}",
            "",
            "## Summary",
            "",
            f"- Status: {report['status']}",
            f"- Clean input: `{report['clean_input']}`",
            f"- Chunks output: `{report['chunks_output']}`",
            f"- Nodes output: `{report['nodes_output']}`",
            f"- Combined nodes: `{report['combined_nodes']}`",
            f"- Clean rows: {report['clean_count']}",
            f"- Chunks: {report['chunk_count']}",
            f"- Nodes: {report['node_count']}",
            f"- Distinct drug names: {report['by_drug_name_count']}",
            f"- Distinct manufacturers: {report['by_manufacturer_count']}",
            f"- Price periods: {', '.join(report['price_periods'])}",
            f"- Combined node count: {combined.get('combined_node_count')}",
            "",
            "## Validation",
            "",
            f"- Valid: {validation['is_valid']}",
            f"- Count matches clean: {validation['count_matches_clean']}",
            f"- Duplicate chunk IDs: {len(validation['duplicate_chunk_ids'])}",
            f"- Missing source URLs: {len(validation['missing_source_url'])}",
            f"- Incorrect policy-basis flags: {len(validation['can_cite_as_policy_basis_errors'])}",
            f"- Chunk issue counts: `{json.dumps(validation['chunk_issues'], ensure_ascii=False)}`",
            "",
            "## Boundary",
            "",
            "- This data is drug product price reference only.",
            "- It is not a formal reimbursement policy basis.",
            "- No embedding, FAISS build, MCP switch, or Agent decision was performed in this step.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
