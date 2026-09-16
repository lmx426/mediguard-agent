"""Convert policy RAG chunks into LlamaIndex TextNode records.

This script is intentionally limited to node preparation. It does not create
embeddings, vector indexes, MCP tools, Case Agent integration, or audit
decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"
DEFAULT_INPUT_CHUNKS = DEFAULT_RAG_READY_ROOT / "chunks" / "policy_chunks.jsonl"
DEFAULT_OUTPUT_NODES = DEFAULT_RAG_READY_ROOT / "nodes" / "policy_nodes.jsonl"
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_nodes_prepare_report.md"
DEFAULT_REPORT_JSON = DEFAULT_CORPUS_ROOT / "reports" / "policy_nodes_prepare_report.json"

NODE_SCHEMA_VERSION = "policy_text_node_v0.1"

SKIP_METADATA_KEYS = {"text"}
EXCLUDED_EMBED_METADATA_KEYS = {
    "chunk_id",
    "chunk_index",
    "source_sequence",
    "source_row_number",
    "source_line_number",
    "source_path",
    "source_url",
    "sha256",
    "page_refs",
    "row_scope",
    "original_chunk_id",
    "chunk_strategy",
    "chunk_size",
    "chunk_overlap",
    "chunk_hard_max",
    "chunk_unit",
    "text_chars",
}
EXCLUDED_LLM_METADATA_KEYS = {
    "source_path",
    "sha256",
    "row_scope",
    "source_line_number",
    "source_row_number",
    "original_chunk_id",
    "chunk_strategy",
}
REQUIRED_METADATA_KEYS = {
    "chunk_id",
    "source_id",
    "title",
    "jurisdiction",
    "policy_domain",
    "content_type",
    "legal_weight",
    "rag_ingestion_role",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LlamaIndex TextNode JSONL from policy_chunks.jsonl."
    )
    parser.add_argument("--input-chunks", type=Path, default=DEFAULT_INPUT_CHUNKS)
    parser.add_argument("--output-nodes", type=Path, default=DEFAULT_OUTPUT_NODES)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


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


def normalize_metadata_value(value: Any) -> str | int | float | bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def evidence_role_for_chunk(chunk: dict[str, Any]) -> str:
    if chunk.get("content_type") == "graph_hint":
        return "retrieval_hint"
    if chunk.get("rag_ingestion_role") == "graph_hint":
        return "retrieval_hint"
    if chunk.get("legal_weight") == "derived_lookup_hint":
        return "retrieval_hint"
    return "policy_evidence"


def can_cite_as_policy_basis(chunk: dict[str, Any]) -> bool:
    return evidence_role_for_chunk(chunk) == "policy_evidence"


def build_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "node_schema_version": NODE_SCHEMA_VERSION,
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "evidence_role": evidence_role_for_chunk(chunk),
        "can_cite_as_policy_basis": can_cite_as_policy_basis(chunk),
    }
    for key, value in chunk.items():
        if key in SKIP_METADATA_KEYS:
            continue
        normalized = normalize_metadata_value(value)
        if normalized is not None:
            metadata[key] = normalized
    return metadata


def chunk_to_text_node(chunk: dict[str, Any]) -> TextNode:
    chunk_id = str(chunk.get("chunk_id") or "").strip()
    text = str(chunk.get("text") or "").strip()
    if not chunk_id:
        raise ValueError("Chunk is missing chunk_id")
    if not text:
        raise ValueError(f"Chunk {chunk_id} is missing text")

    metadata = build_metadata(chunk)
    embed_excluded = sorted(key for key in metadata if key in EXCLUDED_EMBED_METADATA_KEYS)
    llm_excluded = sorted(key for key in metadata if key in EXCLUDED_LLM_METADATA_KEYS)
    return TextNode(
        id_=chunk_id,
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=embed_excluded,
        excluded_llm_metadata_keys=llm_excluded,
    )


def prepare_nodes(chunks: list[dict[str, Any]]) -> list[TextNode]:
    return [chunk_to_text_node(chunk) for chunk in chunks]


def validate_nodes(chunks: list[dict[str, Any]], nodes: list[TextNode]) -> dict[str, Any]:
    node_ids = [node.id_ for node in nodes]
    duplicate_ids = [
        node_id
        for node_id, count in Counter(node_ids).items()
        if node_id and count > 1
    ]
    empty_text = [node.id_ for node in nodes if not node.text.strip()]
    missing_required_metadata: list[dict[str, Any]] = []
    source_url_missing: list[str] = []
    graph_hint_policy_basis: list[str] = []

    for node in nodes:
        missing = sorted(key for key in REQUIRED_METADATA_KEYS if key not in node.metadata)
        if missing:
            missing_required_metadata.append({"node_id": node.id_, "missing": missing})
        if (
            node.metadata.get("content_type") != "graph_hint"
            and not str(node.metadata.get("source_url") or "").startswith(("http://", "https://"))
        ):
            source_url_missing.append(node.id_)
        if (
            node.metadata.get("content_type") == "graph_hint"
            and node.metadata.get("can_cite_as_policy_basis") is not False
        ):
            graph_hint_policy_basis.append(node.id_)

    return {
        "input_chunk_count": len(chunks),
        "node_count": len(nodes),
        "count_matches_input": len(chunks) == len(nodes),
        "duplicate_node_ids": duplicate_ids,
        "empty_text_nodes": empty_text,
        "missing_required_metadata": missing_required_metadata,
        "missing_source_url_non_graph": source_url_missing,
        "graph_hints_marked_as_policy_basis": graph_hint_policy_basis,
        "is_valid": (
            len(chunks) == len(nodes)
            and not duplicate_ids
            and not empty_text
            and not missing_required_metadata
            and not source_url_missing
            and not graph_hint_policy_basis
        ),
    }


def write_nodes(path: Path, nodes: list[TextNode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for node in nodes:
            file_obj.write(json.dumps(node.to_dict(), ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def build_report(
    *,
    chunks: list[dict[str, Any]],
    nodes: list[TextNode],
    validation: dict[str, Any],
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    by_content_type = Counter(str(node.metadata.get("content_type") or "") for node in nodes)
    by_jurisdiction = Counter(str(node.metadata.get("jurisdiction") or "") for node in nodes)
    by_domain = Counter(str(node.metadata.get("policy_domain") or "") for node in nodes)
    by_evidence_role = Counter(str(node.metadata.get("evidence_role") or "") for node in nodes)
    text_lengths = [len(node.text) for node in nodes]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "node_schema_version": NODE_SCHEMA_VERSION,
        "input_chunks": str(input_path),
        "output_nodes": str(output_path),
        "input_chunk_count": len(chunks),
        "node_count": len(nodes),
        "by_content_type": dict(by_content_type),
        "by_jurisdiction": dict(by_jurisdiction),
        "by_policy_domain": dict(by_domain),
        "by_evidence_role": dict(by_evidence_role),
        "text_length_stats": length_stats(text_lengths),
        "excluded_embed_metadata_keys": sorted(EXCLUDED_EMBED_METADATA_KEYS),
        "excluded_llm_metadata_keys": sorted(EXCLUDED_LLM_METADATA_KEYS),
        "validation": validation,
    }


def length_stats(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "avg": round(sum(values) / len(values), 2),
        "p50": percentile(sorted_values, 0.5),
        "p90": percentile(sorted_values, 0.9),
    }


def percentile(values: list[int], ratio: float) -> int:
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * ratio))))
    return values[index]


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy TextNode Prepare Report",
        "",
        f"Updated at: {report['generated_at']}",
        "",
        "This report covers only conversion from `policy_chunks.jsonl` to LlamaIndex `TextNode` records. It does not create embeddings, vector indexes, MCP tools, Case Agent integration, or audit decisions.",
        "",
        "## Summary",
        "",
        f"- Input chunks: `{report['input_chunks']}`",
        f"- Output nodes: `{report['output_nodes']}`",
        f"- Input chunk count: {report['input_chunk_count']}",
        f"- TextNode count: {report['node_count']}",
        f"- Schema version: `{report['node_schema_version']}`",
        f"- Output valid: {'yes' if report['validation']['is_valid'] else 'no'}",
        "",
        "## Distribution",
        "",
        "### Content Type",
        "",
    ]
    append_distribution(lines, report["by_content_type"])
    lines.extend(["", "### Jurisdiction", ""])
    append_distribution(lines, report["by_jurisdiction"])
    lines.extend(["", "### Policy Domain", ""])
    append_distribution(lines, report["by_policy_domain"])
    lines.extend(["", "### Evidence Role", ""])
    append_distribution(lines, report["by_evidence_role"])
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Count matches input: {report['validation']['count_matches_input']}",
            f"- Duplicate node IDs: {len(report['validation']['duplicate_node_ids'])}",
            f"- Empty text nodes: {len(report['validation']['empty_text_nodes'])}",
            f"- Missing required metadata: {len(report['validation']['missing_required_metadata'])}",
            f"- Missing source URL for non-graph nodes: {len(report['validation']['missing_source_url_non_graph'])}",
            f"- Graph hints incorrectly marked as policy basis: {len(report['validation']['graph_hints_marked_as_policy_basis'])}",
            "",
            "## Metadata Handling",
            "",
            "The node text is the only required embedding payload. Source, jurisdiction, policy type, version, evidence role, and URL are preserved as metadata for later filtering and citation display.",
            "",
            "Metadata excluded from embedding text:",
        ]
    )
    for key in report["excluded_embed_metadata_keys"]:
        lines.append(f"- `{key}`")
    lines.extend(["", "Metadata excluded from LLM display by default:", ""])
    for key in report["excluded_llm_metadata_keys"]:
        lines.append(f"- `{key}`")
    lines.append("")
    return "\n".join(lines)


def append_distribution(lines: list[str], distribution: dict[str, int]) -> None:
    for key, count in sorted(distribution.items()):
        lines.append(f"- `{key}`: {count}")


def main() -> None:
    args = parse_args()
    chunks = load_jsonl(args.input_chunks)
    nodes = prepare_nodes(chunks)
    validation = validate_nodes(chunks, nodes)
    if not validation["is_valid"]:
        raise RuntimeError(f"TextNode validation failed: {validation}")
    write_nodes(args.output_nodes, nodes)
    report = build_report(
        chunks=chunks,
        nodes=nodes,
        validation=validation,
        input_path=args.input_chunks,
        output_path=args.output_nodes,
    )
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "input_chunk_count": report["input_chunk_count"],
                "node_count": report["node_count"],
                "by_content_type": report["by_content_type"],
                "by_evidence_role": report["by_evidence_role"],
                "output_nodes": report["output_nodes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
