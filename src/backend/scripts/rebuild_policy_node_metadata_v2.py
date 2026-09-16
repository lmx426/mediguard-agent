"""Build a versioned Policy RAG node file with registered metadata v2.

The script never overwrites the input nodes. It classifies policy evidence
through DeepSeek Function Calling, validates every response with Pydantic, and
routes low-confidence or retrieval-hint records to a review JSONL.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.backend.application.agent.expert_agent.service.policy_filter_contract import (
    ContentType,
    Jurisdiction,
    PolicyDomain,
    repair_json_object,
)
from src.backend.application.agent.runtime.gateways.deepseek import (
    DeepSeekModelGateway,
)
from src.backend.infrastructure.policy_rag.paths import DEFAULT_NODES_PATH
from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    DEFAULT_REPORT_ROOT,
    first_env,
    load_env_file,
)


DEFAULT_OUTPUT = DEFAULT_NODES_PATH.with_name("policy_nodes_metadata_v2.jsonl")
DEFAULT_REVIEW = DEFAULT_NODES_PATH.with_name(
    "policy_nodes_metadata_v2_review.jsonl"
)
DEFAULT_REPORT = DEFAULT_REPORT_ROOT / "policy_node_metadata_v2_report.json"
METADATA_VERSION = "policy_metadata_v2"
REVIEW_CONFIDENCE_THRESHOLD = 0.80


class PolicyNodeMetadataClassification(BaseModel):
    """Registered filter metadata returned by the offline classifier."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    jurisdiction: Jurisdiction
    policy_domain: PolicyDomain
    content_type: ContentType
    can_cite_as_policy_basis: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reclassify Policy RAG node metadata with DeepSeek."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-jsonl", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_distinct_paths(args.nodes_jsonl, args.output_jsonl)
    load_env_file(args.env_file)
    gateway, model = _build_gateway(args)
    rows = read_jsonl(args.nodes_jsonl)
    selected = rows[max(args.offset, 0) :]
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]

    completed = _existing_node_ids(args.output_jsonl) if args.resume else set()
    mode = "a" if args.resume and args.output_jsonl.exists() else "w"
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.review_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()

    with (
        args.output_jsonl.open(mode, encoding="utf-8", newline="\n") as output_file,
        args.review_jsonl.open(mode, encoding="utf-8", newline="\n") as review_file,
    ):
        for index, node in enumerate(selected, start=1):
            node_id = node_id_of(node)
            if node_id in completed:
                counters["resumed"] += 1
                continue
            metadata = metadata_of(node)
            if is_retrieval_hint(metadata):
                counters["retrieval_hint_excluded"] += 1
                _write_jsonl_line(
                    review_file,
                    review_record(
                        node=node,
                        status="excluded_retrieval_hint",
                        old_metadata=metadata,
                    ),
                )
                continue
            try:
                classification = classify_node(
                    node=node,
                    gateway=gateway,
                )
                rebuilt = apply_classification(
                    node=node,
                    classification=classification,
                    model=model,
                )
                _write_jsonl_line(output_file, rebuilt)
                counters["classified"] += 1
                counters[f"jurisdiction:{classification.jurisdiction}"] += 1
                counters[f"policy_domain:{classification.policy_domain}"] += 1
                counters[f"content_type:{classification.content_type}"] += 1
                if classification.confidence < REVIEW_CONFIDENCE_THRESHOLD:
                    counters["low_confidence"] += 1
                    _write_jsonl_line(
                        review_file,
                        review_record(
                            node=rebuilt,
                            status="low_confidence",
                            old_metadata=metadata,
                            classification=classification,
                        ),
                    )
            except Exception as exc:
                counters["classification_error"] += 1
                _write_jsonl_line(
                    review_file,
                    review_record(
                        node=node,
                        status="classification_error",
                        old_metadata=metadata,
                        error=exc,
                    ),
                )
            if args.progress_every > 0 and (
                index % args.progress_every == 0 or index == len(selected)
            ):
                print(
                    f"[metadata-v2] {index}/{len(selected)} "
                    f"classified={counters['classified']} "
                    f"review={counters['low_confidence']} "
                    f"errors={counters['classification_error']}",
                    flush=True,
                )

    report = {
        "status": "complete",
        "metadata_version": METADATA_VERSION,
        "model": model,
        "source_nodes": str(args.nodes_jsonl),
        "output_nodes": str(args.output_jsonl),
        "review_jsonl": str(args.review_jsonl),
        "selected_count": len(selected),
        "counts": dict(sorted(counters.items())),
        "boundary": {
            "input_overwritten": False,
            "node_text_changed": False,
            "goldset_used_online": False,
            "retrieval_hints_in_evidence_index": False,
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _build_gateway(args: argparse.Namespace) -> tuple[DeepSeekModelGateway, str]:
    api_key = first_env(
        "MEDIGUARD_EXPERT_ANALYSIS_DEEPSEEK_API_KEY",
        "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY",
        "MEDIGUARD_DEEPSEEK_API_KEY",
    )
    if not api_key:
        raise RuntimeError("DeepSeek API key for metadata v2 rebuild is missing")
    base_url = args.base_url or first_env(
        "MEDIGUARD_EXPERT_ANALYSIS_BASE_URL",
        "MEDIGUARD_POLICY_RAG_EVAL_BASE_URL",
        "MEDIGUARD_LLM_BASE_URL",
    ) or "https://api.deepseek.com"
    model = args.model or first_env(
        "MEDIGUARD_EXPERT_ANALYSIS_MODEL",
        "MEDIGUARD_POLICY_RAG_EVAL_MODEL",
        "MEDIGUARD_LLM_MODEL",
    ) or "deepseek-chat"
    return (
        DeepSeekModelGateway(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=args.timeout_seconds,
            thinking_enabled=False,
            reasoning_effort="low",
        ),
        model,
    )


def classify_node(
    *,
    node: dict[str, Any],
    gateway: DeepSeekModelGateway,
) -> PolicyNodeMetadataClassification:
    response = gateway.complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是医保政策证据 metadata 分类器，只分类当前节点，不回答政策问题。"
                    "旧 jurisdiction、policy_domain、content_type 不可信，不会提供给你。"
                    "content_type 只能是 policy_text 或 table_row；办事指南和 FAQ 是 doc_type。"
                    "can_cite_as_policy_basis 必须依据发布主体、证据角色和正文性质判断。"
                    "必须调用 submit_policy_node_metadata。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    classification_input(node),
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ],
        tools=[metadata_tool_schema()],
        temperature=0.0,
    )
    if not response.tool_calls:
        raise ValueError("metadata classifier did not call the registered tool")
    payload = repair_json_object(response.tool_calls[0].arguments)
    return PolicyNodeMetadataClassification.model_validate(payload)


def metadata_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_policy_node_metadata",
            "description": "提交当前政策节点的四字段 metadata 分类。",
            "parameters": PolicyNodeMetadataClassification.model_json_schema(),
        },
    }


def classification_input(node: dict[str, Any]) -> dict[str, Any]:
    metadata = metadata_of(node)
    return {
        "node_id": node_id_of(node),
        "title": metadata.get("title"),
        "source_id": metadata.get("source_id"),
        "publishing_organization": metadata.get("publishing_organization")
        or metadata.get("issuer"),
        "doc_type": metadata.get("doc_type"),
        "evidence_role": metadata.get("evidence_role"),
        "resource_type": metadata.get("resource_type"),
        "section_heading": metadata.get("section_heading"),
        "text": str(node.get("text") or "")[:5000],
    }


def apply_classification(
    *,
    node: dict[str, Any],
    classification: PolicyNodeMetadataClassification,
    model: str,
) -> dict[str, Any]:
    rebuilt = dict(node)
    metadata = metadata_of(node)
    metadata.update(
        {
            "jurisdiction": classification.jurisdiction,
            "policy_domain": classification.policy_domain,
            "content_type": classification.content_type,
            "can_cite_as_policy_basis": classification.can_cite_as_policy_basis,
            "metadata_version": METADATA_VERSION,
            "metadata_classifier": "deepseek_function_calling",
            "metadata_classifier_model": model,
            "metadata_confidence": classification.confidence,
            "metadata_review_status": (
                "needs_review"
                if classification.confidence < REVIEW_CONFIDENCE_THRESHOLD
                else "accepted"
            ),
        }
    )
    rebuilt["metadata"] = metadata
    return rebuilt


def review_record(
    *,
    node: dict[str, Any],
    status: str,
    old_metadata: dict[str, Any],
    classification: PolicyNodeMetadataClassification | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id_of(node),
        "status": status,
        "title": old_metadata.get("title"),
        "source_id": old_metadata.get("source_id"),
        "legacy_filters": {
            key: old_metadata.get(key)
            for key in (
                "jurisdiction",
                "policy_domain",
                "content_type",
                "can_cite_as_policy_basis",
            )
        },
        "classification": (
            classification.model_dump(mode="json") if classification else None
        ),
        "error_type": error.__class__.__name__ if error else None,
        "error_message": str(error)[:500] if error else None,
    }


def is_retrieval_hint(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("evidence_role") == "retrieval_hint"
        or metadata.get("content_type") == "graph_hint"
        or metadata.get("policy_domain") == "graph_hints"
    )


def node_id_of(node: dict[str, Any]) -> str:
    return str(node.get("id_") or node.get("node_id") or "").strip()


def metadata_of(node: dict[str, Any]) -> dict[str, Any]:
    value = node.get("metadata")
    return dict(value) if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _existing_node_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {node_id_of(row) for row in read_jsonl(path) if node_id_of(row)}


def _write_jsonl_line(file_obj: Any, payload: dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _validate_distinct_paths(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("metadata v2 output must not overwrite the source nodes")


if __name__ == "__main__":
    main()
