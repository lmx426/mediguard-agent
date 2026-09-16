"""Build the optional Qdrant index for Policy RAG.

This script imports the existing bge-m3 embeddings into Qdrant. It does not
recompute embeddings and does not run retrieval evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_rag.node_store import PolicyNodeStore
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_EMBEDDING_MANIFEST_PATH,
    DEFAULT_EMBEDDING_NODE_IDS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_NODES_PATH,
    DEFAULT_QDRANT_MANIFEST_PATH,
)
from src.backend.infrastructure.policy_rag.qdrant_retriever import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_URL,
)


DEFAULT_BATCH_SIZE = 128
DEFAULT_VECTOR_SIZE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import policy bge-m3 vectors into Qdrant.")
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path(os.environ["MEDIGUARD_POLICY_RAG_QDRANT_PATH"])
        if os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_PATH")
        else None,
        help="Use embedded local Qdrant storage at this path instead of a Qdrant HTTP service.",
    )
    parser.add_argument("--qdrant-url", default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_URL") or DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION") or DEFAULT_QDRANT_COLLECTION)
    parser.add_argument("--api-key", default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_API_KEY") or None)
    parser.add_argument("--embeddings-npy", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--embedding-node-ids", type=Path, default=DEFAULT_EMBEDDING_NODE_IDS_PATH)
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST_PATH)
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_QDRANT_MANIFEST_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-payload-indexes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    QdrantClient, models = load_qdrant()
    client = build_client(
        QdrantClient=QdrantClient,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        api_key=args.api_key,
    )
    embeddings = np.load(args.embeddings_npy, mmap_mode="r")
    node_ids = load_embedding_node_ids(args.embedding_node_ids)
    node_store = PolicyNodeStore(args.nodes_jsonl)
    validate_inputs(embeddings=embeddings, node_ids=node_ids, vector_size=args.vector_size)

    ensure_collection(
        client=client,
        models=models,
        collection=args.collection,
        vector_size=args.vector_size,
        recreate=args.recreate,
    )
    if not args.skip_payload_indexes:
        create_payload_indexes(client=client, models=models, collection=args.collection)

    total = len(node_ids)
    for start in range(0, total, args.batch_size):
        stop = min(start + args.batch_size, total)
        points = []
        for embedding_index in range(start, stop):
            node_id = node_ids[embedding_index]
            record = node_store.get(node_id)
            if record is None:
                raise RuntimeError(f"Node not found for embedding index {embedding_index}: {node_id}")
            points.append(
                models.PointStruct(
                    id=embedding_index,
                    vector=embeddings[embedding_index].astype("float32").tolist(),
                    payload={
                        "node_id": record.node_id,
                        "text": record.text,
                        "metadata": dict(record.metadata),
                    },
                )
            )
        client.upsert(collection_name=args.collection, points=points, wait=True)
        if stop == total or stop % max(args.batch_size * 10, 1) == 0:
            print(f"[qdrant-policy-index] upserted={stop}/{total}", flush=True)

    manifest = build_manifest(
        args=args,
        embeddings=embeddings,
        node_count=total,
        embedding_manifest=read_optional_json(args.embedding_manifest),
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "local_path" if args.qdrant_path else "url",
                "qdrant_path": str(args.qdrant_path) if args.qdrant_path else None,
                "qdrant_url": None if args.qdrant_path else args.qdrant_url,
                "collection": args.collection,
                "node_count": total,
                "vector_size": int(embeddings.shape[1]),
                "manifest": str(args.manifest_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required. Install it with: "
            "python -m pip install qdrant-client"
        ) from exc
    return QdrantClient, models


def build_client(
    *,
    QdrantClient: Any,
    qdrant_path: Path | None,
    qdrant_url: str,
    api_key: str | None,
) -> Any:
    if qdrant_path is not None:
        qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(qdrant_path))
    return QdrantClient(url=qdrant_url, api_key=api_key)


def ensure_collection(
    *,
    client: Any,
    models: Any,
    collection: str,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = collection_exists(client, collection)
    if exists and recreate:
        client.delete_collection(collection_name=collection)
        exists = False
    if exists:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def collection_exists(client: Any, collection: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name=collection))
    try:
        client.get_collection(collection_name=collection)
        return True
    except Exception:
        return False


def create_payload_indexes(*, client: Any, models: Any, collection: str) -> None:
    index_specs = [
        ("node_id", models.PayloadSchemaType.KEYWORD),
        ("metadata.source_id", models.PayloadSchemaType.KEYWORD),
        ("metadata.jurisdiction", models.PayloadSchemaType.KEYWORD),
        ("metadata.policy_domain", models.PayloadSchemaType.KEYWORD),
        ("metadata.content_type", models.PayloadSchemaType.KEYWORD),
        ("metadata.doc_type", models.PayloadSchemaType.KEYWORD),
        ("metadata.can_cite_as_policy_basis", models.PayloadSchemaType.BOOL),
    ]
    for field_name, schema in index_specs:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:
            # Existing indexes are fine; Qdrant versions vary in duplicate-index behavior.
            continue


def load_embedding_node_ids(path: Path) -> list[str]:
    node_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            embedding_index = int(payload.get("embedding_index"))
            if embedding_index != len(node_ids):
                raise ValueError(f"Non-sequential embedding_index at {path}:{line_no}")
            node_id = str(payload.get("node_id") or "").strip()
            if not node_id:
                raise ValueError(f"Missing node_id at {path}:{line_no}")
            node_ids.append(node_id)
    return node_ids


def validate_inputs(*, embeddings: Any, node_ids: list[str], vector_size: int) -> None:
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}")
    if embeddings.shape[0] != len(node_ids):
        raise ValueError(
            f"Embedding rows {embeddings.shape[0]} do not match node IDs {len(node_ids)}"
        )
    if embeddings.shape[1] != vector_size:
        raise ValueError(
            f"Embedding dim {embeddings.shape[1]} does not match vector size {vector_size}"
        )


def build_manifest(
    *,
    args: argparse.Namespace,
    embeddings: Any,
    node_count: int,
    embedding_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_version": "policy_qdrant_bge_m3_v0.1",
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "local_path" if args.qdrant_path else "url",
        "qdrant_path": str(args.qdrant_path) if args.qdrant_path else None,
        "qdrant_url": None if args.qdrant_path else args.qdrant_url,
        "collection": args.collection,
        "node_count": node_count,
        "embedding_dim": int(embeddings.shape[1]),
        "distance": "Cosine",
        "input_embeddings": str(args.embeddings_npy),
        "input_node_ids": str(args.embedding_node_ids),
        "input_nodes": str(args.nodes_jsonl),
        "input_embedding_manifest": str(args.embedding_manifest),
        "embedding_model_name": embedding_manifest.get("embedding_model_name")
        or embedding_manifest.get("model_name")
        or "BAAI/bge-m3",
        "embedding_normalize": embedding_manifest.get("embedding_normalize", True),
        "boundary": {
            "builds_vector_index_only": True,
            "runs_retrieval": False,
            "runs_evaluation": False,
            "runs_reranker": False,
            "makes_audit_decisions": False,
        },
    }


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
