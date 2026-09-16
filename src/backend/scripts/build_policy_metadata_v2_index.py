"""Build a versioned FAISS index for metadata v2 without re-embedding text."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_METADATA_V2_INDEX_DIR,
    DEFAULT_METADATA_V2_NODES_PATH,
    DEFAULT_RAG_READY_ROOT,
)


DEFAULT_SOURCE_EMBEDDING_DIR = (
    DEFAULT_RAG_READY_ROOT
    / "embeddings"
    / "policy_bge_m3_with_shanghai_drug_price_20260820_full"
)
DEFAULT_SOURCE_EMBEDDINGS = DEFAULT_SOURCE_EMBEDDING_DIR / "embeddings.npy"
DEFAULT_SOURCE_NODE_IDS = DEFAULT_SOURCE_EMBEDDING_DIR / "node_ids.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse BGE-M3 vectors to build the metadata v2 FAISS index."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_METADATA_V2_NODES_PATH)
    parser.add_argument("--source-embeddings", type=Path, default=DEFAULT_SOURCE_EMBEDDINGS)
    parser.add_argument("--source-node-ids", type=Path, default=DEFAULT_SOURCE_NODE_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METADATA_V2_INDEX_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.output_dir / "policy.index"
    node_ids_path = args.output_dir / "index_node_ids.jsonl"
    manifest_path = args.output_dir / "index_manifest.json"
    if index_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {index_path}; pass --force for this v2 target."
        )

    nodes = read_jsonl(args.nodes_jsonl)
    source_ids = read_jsonl(args.source_node_ids)
    vectors = np.load(args.source_embeddings, mmap_mode="r")
    if vectors.ndim != 2 or len(source_ids) != vectors.shape[0]:
        raise ValueError("source embeddings and node-id mapping are inconsistent")
    source_positions = {
        str(record.get("node_id") or ""): int(record.get("embedding_index"))
        for record in source_ids
    }
    selected_positions: list[int] = []
    index_records: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id_") or node.get("node_id") or "")
        if node_id not in source_positions:
            raise KeyError(f"metadata v2 node has no reusable embedding: {node_id}")
        selected_positions.append(source_positions[node_id])
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        index_records.append(
            {
                "embedding_index": len(index_records),
                "node_id": node_id,
                "source_id": metadata.get("source_id"),
                "title": metadata.get("title"),
                "jurisdiction": metadata.get("jurisdiction"),
                "policy_domain": metadata.get("policy_domain"),
                "content_type": metadata.get("content_type"),
                "evidence_role": metadata.get("evidence_role"),
                "can_cite_as_policy_basis": metadata.get("can_cite_as_policy_basis"),
                "source_url": metadata.get("source_url"),
                "metadata_version": metadata.get("metadata_version"),
            }
        )

    selected = np.asarray(vectors[selected_positions], dtype=np.float32)
    index = faiss.IndexFlatIP(int(selected.shape[1]))
    index.add(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    write_jsonl(node_ids_path, index_records)
    manifest = {
        "artifact_version": "policy_faiss_bge_m3_metadata_v2",
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(),
        "metadata_version": "policy_metadata_v2",
        "nodes_jsonl": str(args.nodes_jsonl),
        "source_embeddings": str(args.source_embeddings),
        "source_node_ids": str(args.source_node_ids),
        "embedding_recomputed": False,
        "embedding_model_name": "BAAI/bge-m3",
        "embedding_normalize": True,
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "node_count": int(index.ntotal),
        "embedding_dim": int(selected.shape[1]),
        "index_path": str(index_path),
        "index_node_ids_path": str(node_ids_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
