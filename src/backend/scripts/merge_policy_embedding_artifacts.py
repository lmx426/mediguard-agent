"""Merge completed policy embedding artifacts into a combined artifact.

This is intended for small reference-data additions where re-embedding the
whole policy corpus would be wasteful. It does not build FAISS, run retrieval,
switch MCP runtime paths, or make audit decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RAG_READY_ROOT = (
    PROJECT_ROOT / "src" / "backend" / "policy_corpus" / "rag_ready"
)
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus" / "reports"
DEFAULT_BASE_DIR = DEFAULT_RAG_READY_ROOT / "embeddings" / "policy_bge_m3"
DEFAULT_EXTRA_DIR = (
    DEFAULT_RAG_READY_ROOT / "embeddings" / "shanghai_drug_price_bge_m3_20260820_full"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_RAG_READY_ROOT
    / "embeddings"
    / "policy_bge_m3_with_shanghai_drug_price_20260820_full"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_REPORT_ROOT / "policy_embedding_merge_shanghai_drug_price_20260820_full.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_REPORT_ROOT / "policy_embedding_merge_shanghai_drug_price_20260820_full.md"
)

MERGE_ARTIFACT_VERSION = "policy_embedding_merge_v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--extra-dir", type=Path, default=DEFAULT_EXTRA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge_artifacts(args)
    write_report(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": report["status"],
                "base_node_count": report["base_node_count"],
                "extra_node_count": report["extra_node_count"],
                "combined_node_count": report["combined_node_count"],
                "embedding_dim": report["embedding_dim"],
                "output_dir": report["output_dir"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def merge_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_output = output_dir / "embeddings.npy"
    node_ids_output = output_dir / "node_ids.jsonl"
    manifest_output = output_dir / "embedding_manifest.json"
    progress_output = output_dir / "embedding_progress.json"

    if manifest_output.exists() and not args.force:
        existing = read_json(manifest_output)
        if existing.get("status") == "complete":
            return {**existing, "skipped": True}

    base_manifest = read_json(args.base_dir / "embedding_manifest.json")
    extra_manifest = read_json(args.extra_dir / "embedding_manifest.json")
    base_records = load_node_id_records(args.base_dir / "node_ids.jsonl")
    extra_records = load_node_id_records(args.extra_dir / "node_ids.jsonl")
    base_matrix = load_matrix(args.base_dir / "embeddings.npy")
    extra_matrix = load_matrix(args.extra_dir / "embeddings.npy")

    validation = validate_inputs(
        base_records=base_records,
        extra_records=extra_records,
        base_matrix=base_matrix,
        extra_matrix=extra_matrix,
        base_manifest=base_manifest,
        extra_manifest=extra_manifest,
    )
    if not validation["is_valid"]:
        raise RuntimeError(f"Embedding merge validation failed: {validation}")

    combined_count = len(base_records) + len(extra_records)
    embedding_dim = int(base_matrix.shape[1])
    combined_matrix = np.lib.format.open_memmap(
        embeddings_output,
        mode="w+",
        dtype=np.float32,
        shape=(combined_count, embedding_dim),
    )
    combined_matrix[: len(base_records)] = base_matrix
    combined_matrix[len(base_records) :] = extra_matrix
    combined_matrix.flush()

    combined_records = reindex_records(base_records, extra_records)
    write_jsonl(node_ids_output, combined_records)

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        "artifact_version": MERGE_ARTIFACT_VERSION,
        "status": "complete",
        "generated_at": generated_at,
        "skipped": False,
        "base_dir": str(args.base_dir),
        "extra_dir": str(args.extra_dir),
        "output_dir": str(output_dir),
        "embeddings_path": str(embeddings_output),
        "node_ids_path": str(node_ids_output),
        "model_name": base_manifest.get("model_name"),
        "normalize_embeddings": base_manifest.get("normalize_embeddings"),
        "max_length": base_manifest.get("max_length"),
        "dtype": "float32",
        "shape": [combined_count, embedding_dim],
        "node_count": combined_count,
        "base_node_count": len(base_records),
        "extra_node_count": len(extra_records),
        "combined_node_count": combined_count,
        "embedding_dim": embedding_dim,
        "by_content_type": distribution(combined_records, "content_type"),
        "by_jurisdiction": distribution(combined_records, "jurisdiction"),
        "by_policy_domain": distribution(combined_records, "policy_domain"),
        "by_evidence_role": distribution(combined_records, "evidence_role"),
        "validation": validation,
        "boundary": {
            "merges_embeddings": True,
            "embeds": False,
            "builds_faiss": False,
            "runs_retrieval": False,
            "switches_mcp_default_index": False,
        },
    }
    write_json(manifest_output, manifest)
    write_json(
        progress_output,
        {
            "status": "complete",
            "completed_count": combined_count,
            "node_count": combined_count,
            "embedding_dim": embedding_dim,
            "updated_at": generated_at,
        },
    )
    return manifest


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


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"Expected float32 matrix at {path}, got {matrix.dtype}")
    return matrix


def validate_inputs(
    *,
    base_records: list[dict[str, Any]],
    extra_records: list[dict[str, Any]],
    base_matrix: np.ndarray,
    extra_matrix: np.ndarray,
    base_manifest: dict[str, Any],
    extra_manifest: dict[str, Any],
) -> dict[str, Any]:
    base_ids = [str(record.get("node_id") or "") for record in base_records]
    extra_ids = [str(record.get("node_id") or "") for record in extra_records]
    duplicate_between = sorted(set(base_ids).intersection(extra_ids))
    duplicate_all = [
        node_id
        for node_id, count in Counter(base_ids + extra_ids).items()
        if node_id and count > 1
    ]
    base_norms = np.linalg.norm(base_matrix, axis=1)
    extra_norms = np.linalg.norm(extra_matrix, axis=1)
    model_matches = base_manifest.get("model_name") == extra_manifest.get("model_name")
    normalize_matches = (
        base_manifest.get("normalize_embeddings")
        == extra_manifest.get("normalize_embeddings")
    )
    return {
        "is_valid": (
            bool(base_records)
            and bool(extra_records)
            and base_matrix.shape[0] == len(base_records)
            and extra_matrix.shape[0] == len(extra_records)
            and base_matrix.shape[1] == extra_matrix.shape[1]
            and not duplicate_between
            and not duplicate_all
            and model_matches
            and normalize_matches
            and bool(np.isfinite(base_matrix).all())
            and bool(np.isfinite(extra_matrix).all())
            and bool(np.allclose(base_norms, 1.0, atol=1e-3))
            and bool(np.allclose(extra_norms, 1.0, atol=1e-3))
        ),
        "base_matrix_shape": [int(base_matrix.shape[0]), int(base_matrix.shape[1])],
        "extra_matrix_shape": [int(extra_matrix.shape[0]), int(extra_matrix.shape[1])],
        "base_record_count": len(base_records),
        "extra_record_count": len(extra_records),
        "base_row_count_matches_records": base_matrix.shape[0] == len(base_records),
        "extra_row_count_matches_records": extra_matrix.shape[0] == len(extra_records),
        "embedding_dim_matches": base_matrix.shape[1] == extra_matrix.shape[1],
        "model_matches": model_matches,
        "normalize_matches": normalize_matches,
        "duplicate_node_ids_between_inputs": duplicate_between,
        "duplicate_node_ids_all": duplicate_all,
        "base_norm_min": float(base_norms.min()) if base_norms.size else None,
        "base_norm_max": float(base_norms.max()) if base_norms.size else None,
        "extra_norm_min": float(extra_norms.min()) if extra_norms.size else None,
        "extra_norm_max": float(extra_norms.max()) if extra_norms.size else None,
    }


def reindex_records(
    base_records: list[dict[str, Any]],
    extra_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, record in enumerate(base_records + extra_records):
        new_record = dict(record)
        new_record["embedding_index"] = index
        output.append(new_record)
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) or "") for record in records))


def write_report(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    validation = report.get("validation") or {}
    return "\n".join(
        [
            "# Policy Embedding Merge Report",
            "",
            f"Updated at: {report.get('generated_at')}",
            "",
            "## Summary",
            "",
            f"- Status: {report.get('status')}",
            f"- Base dir: `{report.get('base_dir')}`",
            f"- Extra dir: `{report.get('extra_dir')}`",
            f"- Output dir: `{report.get('output_dir')}`",
            f"- Base nodes: {report.get('base_node_count')}",
            f"- Extra nodes: {report.get('extra_node_count')}",
            f"- Combined nodes: {report.get('combined_node_count')}",
            f"- Embedding dimension: {report.get('embedding_dim')}",
            f"- Model: `{report.get('model_name')}`",
            "",
            "## Validation",
            "",
            f"- Valid: {validation.get('is_valid')}",
            f"- Base matrix shape: {validation.get('base_matrix_shape')}",
            f"- Extra matrix shape: {validation.get('extra_matrix_shape')}",
            f"- Model matches: {validation.get('model_matches')}",
            f"- Normalize matches: {validation.get('normalize_matches')}",
            f"- Duplicate node IDs between inputs: {len(validation.get('duplicate_node_ids_between_inputs') or [])}",
            "",
            "## Boundary",
            "",
            "- No FAISS index was built by this merge script.",
            "- No retrieval, reranking, MCP switch, or Agent workflow was changed.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
