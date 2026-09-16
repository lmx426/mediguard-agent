"""Embed prepared policy TextNode records with BAAI/bge-m3.

This script reads serialized LlamaIndex TextNode records and writes an
embedding matrix plus node-id mapping. It intentionally does not build a FAISS
index, expose MCP tools, integrate Case Agent, or produce audit decisions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_rag.device import resolve_policy_model_device


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"
DEFAULT_INPUT_NODES = DEFAULT_RAG_READY_ROOT / "nodes" / "policy_nodes.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_RAG_READY_ROOT / "embeddings" / "policy_bge_m3"
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_embedding_report.md"
DEFAULT_REPORT_JSON = DEFAULT_CORPUS_ROOT / "reports" / "policy_embedding_report.json"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "runtime" / "hf_cache"

EMBEDDING_ARTIFACT_VERSION = "policy_embedding_bge_m3_v0.1"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed policy TextNode records with BAAI/bge-m3."
    )
    parser.add_argument("--input-nodes", type=Path, default=DEFAULT_INPUT_NODES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu/cuda.")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--limit", type=int, default=0, help="Embed only first N nodes for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embedding artifacts.")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.limit and args.limit > 0:
        return DEFAULT_RAG_READY_ROOT / "embeddings" / f"policy_bge_m3_sample_{args.limit}"
    return DEFAULT_OUTPUT_DIR


def load_nodes(path: Path, *, limit: int = 0) -> list[TextNode]:
    nodes: list[TextNode] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            node = TextNode.from_dict(payload)
            if not node.id_:
                raise ValueError(f"TextNode at line {line_no} is missing id_")
            if not node.text.strip():
                raise ValueError(f"TextNode {node.id_} is missing text")
            nodes.append(node)
            if limit and len(nodes) >= limit:
                break
    return nodes


def cleanup_existing(output_dir: Path) -> None:
    for name in (
        "embeddings.npy",
        "node_ids.jsonl",
        "embedding_manifest.json",
        "embedding_progress.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_node_ids(path: Path, nodes: list[TextNode]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for index, node in enumerate(nodes):
            payload = {
                "embedding_index": index,
                "node_id": node.id_,
                "source_id": node.metadata.get("source_id"),
                "title": node.metadata.get("title"),
                "jurisdiction": node.metadata.get("jurisdiction"),
                "policy_domain": node.metadata.get("policy_domain"),
                "content_type": node.metadata.get("content_type"),
                "evidence_role": node.metadata.get("evidence_role"),
                "can_cite_as_policy_basis": node.metadata.get("can_cite_as_policy_basis"),
                "source_url": node.metadata.get("source_url"),
            }
            file_obj.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def build_embed_model(args: argparse.Namespace) -> HuggingFaceEmbedding:
    if args.cache_folder:
        args.cache_folder.mkdir(parents=True, exist_ok=True)
    device = resolve_policy_model_device(args.device)
    return HuggingFaceEmbedding(
        model_name=args.model_name,
        max_length=args.max_length,
        normalize=True,
        embed_batch_size=args.batch_size,
        cache_folder=str(args.cache_folder) if args.cache_folder else None,
        device=device,
        trust_remote_code=False,
        show_progress_bar=False,
    )


def embed_batch(embed_model: HuggingFaceEmbedding, texts: list[str]) -> np.ndarray:
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")
    return matrix


def completed_count_from_progress(progress_path: Path, *, node_count: int) -> int:
    progress = read_json(progress_path)
    completed = int(progress.get("completed_count") or 0)
    if completed < 0 or completed > node_count:
        return 0
    return completed


def load_existing_matrix(path: Path, *, expected_count: int) -> np.memmap | None:
    if not path.exists():
        return None
    matrix = np.load(path, mmap_mode="r+")
    if matrix.ndim != 2 or matrix.shape[0] != expected_count:
        raise ValueError(
            f"Existing embedding matrix shape {matrix.shape} does not match node count {expected_count}"
        )
    return matrix


def create_matrix(path: Path, *, node_count: int, dim: int) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(node_count, dim),
    )


def content_distribution(nodes: list[TextNode], key: str) -> dict[str, int]:
    return dict(Counter(str(node.metadata.get(key) or "") for node in nodes))


def embed_nodes(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    node_ids_path = output_dir / "node_ids.jsonl"
    manifest_path = output_dir / "embedding_manifest.json"
    progress_path = output_dir / "embedding_progress.json"

    if args.force:
        cleanup_existing(output_dir)

    nodes = load_nodes(args.input_nodes, limit=max(0, args.limit))
    if not nodes:
        raise RuntimeError("No TextNode records found to embed.")

    existing_manifest = read_json(manifest_path)
    if (
        existing_manifest.get("status") == "complete"
        and embeddings_path.exists()
        and not args.force
    ):
        return {
            **existing_manifest,
            "skipped": True,
            "skip_reason": "embedding artifacts already complete",
        }

    if not node_ids_path.exists() or args.force:
        write_node_ids(node_ids_path, nodes)

    embed_model = build_embed_model(args)
    node_count = len(nodes)
    matrix = load_existing_matrix(embeddings_path, expected_count=node_count)
    completed_count = completed_count_from_progress(progress_path, node_count=node_count)

    if matrix is None:
        completed_count = 0
        first_batch_end = min(args.batch_size, node_count)
        first_texts = [node.text for node in nodes[:first_batch_end]]
        first_matrix = embed_batch(embed_model, first_texts)
        matrix = create_matrix(
            embeddings_path,
            node_count=node_count,
            dim=first_matrix.shape[1],
        )
        matrix[:first_batch_end] = first_matrix
        matrix.flush()
        completed_count = first_batch_end
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_count": completed_count,
                "node_count": node_count,
                "embedding_dim": int(first_matrix.shape[1]),
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )

    embedding_dim = int(matrix.shape[1])
    for start in range(completed_count, node_count, args.batch_size):
        end = min(start + args.batch_size, node_count)
        texts = [node.text for node in nodes[start:end]]
        batch_matrix = embed_batch(embed_model, texts)
        if batch_matrix.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dim changed from {embedding_dim} to {batch_matrix.shape[1]}"
            )
        matrix[start:end] = batch_matrix
        matrix.flush()
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_count": end,
                "node_count": node_count,
                "embedding_dim": embedding_dim,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        print(
            json.dumps(
                {
                    "status": "running",
                    "completed_count": end,
                    "node_count": node_count,
                    "embedding_dim": embedding_dim,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    manifest = {
        "artifact_version": EMBEDDING_ARTIFACT_VERSION,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_nodes": str(args.input_nodes),
        "output_dir": str(output_dir),
        "embeddings_path": str(embeddings_path),
        "node_ids_path": str(node_ids_path),
        "model_name": args.model_name,
        "normalize_embeddings": True,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": resolve_policy_model_device(args.device),
        "cache_folder": str(args.cache_folder) if args.cache_folder else None,
        "node_count": node_count,
        "embedding_dim": embedding_dim,
        "dtype": "float32",
        "shape": [node_count, embedding_dim],
        "limited_run": bool(args.limit),
        "by_content_type": content_distribution(nodes, "content_type"),
        "by_jurisdiction": content_distribution(nodes, "jurisdiction"),
        "by_evidence_role": content_distribution(nodes, "evidence_role"),
        "skipped": False,
    }
    write_json(manifest_path, manifest)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_count": node_count,
            "node_count": node_count,
            "embedding_dim": embedding_dim,
            "updated_at": manifest["generated_at"],
        },
    )
    return manifest


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy Embedding Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report covers only TextNode embedding generation. It does not build vector indexes, MCP tools, Case Agent integration, or audit decisions.",
        "",
        "## Summary",
        "",
        f"- Status: {report.get('status')}",
        f"- Model: `{report.get('model_name')}`",
        f"- Input nodes: `{report.get('input_nodes')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Embeddings: `{report.get('embeddings_path')}`",
        f"- Node ID map: `{report.get('node_ids_path')}`",
        f"- Node count: {report.get('node_count')}",
        f"- Embedding dimension: {report.get('embedding_dim')}",
        f"- Shape: {report.get('shape')}",
        f"- Dtype: `{report.get('dtype')}`",
        f"- Normalize embeddings: {report.get('normalize_embeddings')}",
        f"- Max length: {report.get('max_length')}",
        f"- Batch size: {report.get('batch_size')}",
        f"- Device: `{report.get('device')}`",
        f"- Cache folder: `{report.get('cache_folder')}`",
        f"- Limited run: {report.get('limited_run')}",
        "",
        "## Distribution",
        "",
        "### Content Type",
        "",
    ]
    append_distribution(lines, report.get("by_content_type") or {})
    lines.extend(["", "### Jurisdiction", ""])
    append_distribution(lines, report.get("by_jurisdiction") or {})
    lines.extend(["", "### Evidence Role", ""])
    append_distribution(lines, report.get("by_evidence_role") or {})
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- No index was built in this step.",
            "- No retrieval or reranking was run in this step.",
            "- No Agent or audit workflow was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def append_distribution(lines: list[str], distribution: dict[str, int]) -> None:
    for key, count in sorted(distribution.items()):
        lines.append(f"- `{key}`: {count}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    report = embed_nodes(args)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "skipped": report.get("skipped"),
                "node_count": report.get("node_count"),
                "embedding_dim": report.get("embedding_dim"),
                "shape": report.get("shape"),
                "output_dir": report.get("output_dir"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
