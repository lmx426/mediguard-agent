"""Build a local FAISS index for the policy RAG corpus.

This script consumes the completed bge-m3 embedding artifacts and writes a
local exact-search FAISS index. It intentionally does not run retrieval,
reranking, MCP tools, Case Agent integration, or audit decisions.
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

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"
DEFAULT_EMBEDDING_DIR = DEFAULT_RAG_READY_ROOT / "embeddings" / "policy_bge_m3"
DEFAULT_EMBEDDINGS = DEFAULT_EMBEDDING_DIR / "embeddings.npy"
DEFAULT_EMBEDDING_MANIFEST = DEFAULT_EMBEDDING_DIR / "embedding_manifest.json"
DEFAULT_NODE_IDS = DEFAULT_EMBEDDING_DIR / "node_ids.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_RAG_READY_ROOT / "indexes" / "faiss_bge_m3_flat_ip"
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_faiss_index_report.md"
DEFAULT_REPORT_JSON = DEFAULT_CORPUS_ROOT / "reports" / "policy_faiss_index_report.json"

INDEX_ARTIFACT_VERSION = "policy_faiss_bge_m3_flat_ip_v0.1"
INDEX_FILENAME = "policy.index"
NODE_ID_MAP_FILENAME = "index_node_ids.jsonl"
MANIFEST_FILENAME = "index_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS IndexFlatIP index from policy bge-m3 embeddings."
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--node-ids", type=Path, default=DEFAULT_NODE_IDS)
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def load_node_id_records(path: Path) -> list[dict[str, Any]]:
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


def validate_node_id_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = [str(record.get("node_id") or "") for record in records]
    embedding_indexes = [record.get("embedding_index") for record in records]
    duplicate_node_ids = [
        node_id for node_id, count in Counter(node_ids).items() if node_id and count > 1
    ]
    missing_node_ids = [index for index, node_id in enumerate(node_ids) if not node_id]
    non_sequential_indexes = [
        index
        for index, embedding_index in enumerate(embedding_indexes)
        if embedding_index != index
    ]
    return {
        "node_id_count": len(records),
        "duplicate_node_ids": duplicate_node_ids,
        "missing_node_id_positions": missing_node_ids,
        "non_sequential_embedding_indexes": non_sequential_indexes,
        "is_valid": (
            bool(records)
            and not duplicate_node_ids
            and not missing_node_ids
            and not non_sequential_indexes
        ),
    }


def load_embedding_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings, got {matrix.dtype}")
    return matrix


def validate_embeddings(matrix: np.ndarray, *, expected_rows: int) -> dict[str, Any]:
    finite = bool(np.isfinite(matrix).all())
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": str(matrix.dtype),
        "row_count_matches_node_ids": int(matrix.shape[0]) == expected_rows,
        "all_finite": finite,
        "norm_min": float(norms.min()) if norms.size else 0.0,
        "norm_mean": float(norms.mean()) if norms.size else 0.0,
        "norm_max": float(norms.max()) if norms.size else 0.0,
        "norm_close_to_1_all": bool(np.allclose(norms, 1.0, atol=1e-3)),
    }


def build_faiss_index(matrix: np.ndarray) -> faiss.Index:
    vectors = np.ascontiguousarray(matrix, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def validate_index(index: faiss.Index, matrix: np.ndarray) -> dict[str, Any]:
    sample_count = min(5, int(matrix.shape[0]))
    sample_scores: list[float] = []
    sample_indexes: list[int] = []
    if sample_count:
        scores, indexes = index.search(np.ascontiguousarray(matrix[:sample_count]), 1)
        sample_scores = [float(score) for score in scores[:, 0]]
        sample_indexes = [int(item) for item in indexes[:, 0]]
    return {
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "cosine_equivalent": True,
        "index_dimension": int(index.d),
        "index_ntotal": int(index.ntotal),
        "sample_self_search_count": sample_count,
        "sample_top_indexes": sample_indexes,
        "sample_top_score_min": min(sample_scores) if sample_scores else None,
        "sample_top_score_max": max(sample_scores) if sample_scores else None,
        "sample_scores_all_finite": bool(np.isfinite(sample_scores).all())
        if sample_scores
        else True,
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    embedding_manifest: dict[str, Any],
    embedding_validation: dict[str, Any],
    node_id_validation: dict[str, Any],
    index_validation: dict[str, Any],
    skipped: bool,
) -> dict[str, Any]:
    index_path = args.output_dir / INDEX_FILENAME
    index_node_ids_path = args.output_dir / NODE_ID_MAP_FILENAME
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    validation_is_valid = (
        node_id_validation["is_valid"]
        and embedding_validation["row_count_matches_node_ids"]
        and embedding_validation["all_finite"]
        and embedding_validation["norm_close_to_1_all"]
        and index_validation["index_ntotal"] == node_id_validation["node_id_count"]
        and index_validation["index_dimension"] == embedding_validation["shape"][1]
        and index_validation["sample_scores_all_finite"]
    )
    return {
        "artifact_version": INDEX_ARTIFACT_VERSION,
        "status": "complete" if validation_is_valid else "invalid",
        "generated_at": generated_at,
        "skipped": skipped,
        "input_embeddings": str(args.embeddings),
        "input_node_ids": str(args.node_ids),
        "input_embedding_manifest": str(args.embedding_manifest),
        "embedding_manifest_status": embedding_manifest.get("status"),
        "embedding_model_name": embedding_manifest.get("model_name"),
        "embedding_normalize": embedding_manifest.get("normalize_embeddings"),
        "output_dir": str(args.output_dir),
        "index_path": str(index_path),
        "index_node_ids_path": str(index_node_ids_path),
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "cosine_equivalent": True,
        "node_count": node_id_validation["node_id_count"],
        "embedding_dim": embedding_validation["shape"][1],
        "validation": {
            "is_valid": validation_is_valid,
            "embedding": embedding_validation,
            "node_ids": node_id_validation,
            "index": index_validation,
        },
        "boundary": {
            "builds_vector_index_only": True,
            "runs_retrieval": False,
            "runs_reranker": False,
            "integrates_mcp": False,
            "integrates_case_agent": False,
            "makes_audit_decisions": False,
        },
    }


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / INDEX_FILENAME
    index_node_ids_path = args.output_dir / NODE_ID_MAP_FILENAME
    manifest_path = args.output_dir / MANIFEST_FILENAME

    existing_manifest = read_json(manifest_path)
    if (
        existing_manifest.get("status") == "complete"
        and index_path.exists()
        and index_node_ids_path.exists()
        and not args.force
    ):
        return {**existing_manifest, "skipped": True, "skip_reason": "index already complete"}

    embedding_manifest = read_json(args.embedding_manifest)
    records = load_node_id_records(args.node_ids)
    node_id_validation = validate_node_id_records(records)
    if not node_id_validation["is_valid"]:
        raise RuntimeError(f"Invalid node-id map: {node_id_validation}")

    matrix = load_embedding_matrix(args.embeddings)
    embedding_validation = validate_embeddings(matrix, expected_rows=len(records))
    if not (
        embedding_validation["row_count_matches_node_ids"]
        and embedding_validation["all_finite"]
        and embedding_validation["norm_close_to_1_all"]
    ):
        raise RuntimeError(f"Invalid embedding matrix: {embedding_validation}")

    index = build_faiss_index(matrix)
    index_validation = validate_index(index, matrix)
    faiss.write_index(index, str(index_path))
    shutil.copyfile(args.node_ids, index_node_ids_path)

    manifest = build_manifest(
        args=args,
        embedding_manifest=embedding_manifest,
        embedding_validation=embedding_validation,
        node_id_validation=node_id_validation,
        index_validation=index_validation,
        skipped=False,
    )
    if manifest["status"] != "complete":
        raise RuntimeError(f"FAISS index validation failed: {manifest['validation']}")
    write_json(manifest_path, manifest)
    return manifest


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    validation = report.get("validation") or {}
    embedding = validation.get("embedding") or {}
    index = validation.get("index") or {}
    node_ids = validation.get("node_ids") or {}
    lines = [
        "# Policy FAISS Index Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report covers only local FAISS index construction. It does not run retrieval, reranking, MCP tools, Case Agent integration, or audit decisions.",
        "",
        "## Summary",
        "",
        f"- Status: {report.get('status')}",
        f"- Index type: `{report.get('index_type')}`",
        f"- Metric: `{report.get('metric')}`",
        f"- Cosine equivalent: {report.get('cosine_equivalent')}",
        f"- Embedding model: `{report.get('embedding_model_name')}`",
        f"- Input embeddings: `{report.get('input_embeddings')}`",
        f"- Input node IDs: `{report.get('input_node_ids')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Index path: `{report.get('index_path')}`",
        f"- Index node IDs: `{report.get('index_node_ids_path')}`",
        f"- Node count: {report.get('node_count')}",
        f"- Embedding dimension: {report.get('embedding_dim')}",
        "",
        "## Validation",
        "",
        f"- Overall valid: {validation.get('is_valid')}",
        f"- Node ID records: {node_ids.get('node_id_count')}",
        f"- Duplicate node IDs: {len(node_ids.get('duplicate_node_ids') or [])}",
        f"- Non-sequential embedding indexes: {len(node_ids.get('non_sequential_embedding_indexes') or [])}",
        f"- Embedding shape: {embedding.get('shape')}",
        f"- Embedding dtype: `{embedding.get('dtype')}`",
        f"- Embedding row count matches node IDs: {embedding.get('row_count_matches_node_ids')}",
        f"- All embedding values finite: {embedding.get('all_finite')}",
        f"- Row norms close to 1 within 1e-3: {embedding.get('norm_close_to_1_all')}",
        f"- Index dimension: {index.get('index_dimension')}",
        f"- Index ntotal: {index.get('index_ntotal')}",
        f"- Self-search sample top score min: {index.get('sample_top_score_min')}",
        f"- Self-search sample top score max: {index.get('sample_top_score_max')}",
        "",
        "## Boundary",
        "",
        "- No query retrieval was run in this step.",
        "- No reranker was used in this step.",
        "- No Agent or audit workflow was changed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_index(args)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "skipped": report.get("skipped"),
                "node_count": report.get("node_count"),
                "embedding_dim": report.get("embedding_dim"),
                "index_path": report.get("index_path"),
                "index_type": report.get("index_type"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
