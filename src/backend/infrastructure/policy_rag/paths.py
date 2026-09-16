"""Default artifact paths for the local policy RAG runtime."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value and value.strip():
        return Path(value)
    return default


DEFAULT_RUNTIME_NODES_PATH = (
    DEFAULT_RAG_READY_ROOT
    / "nodes"
    / "policy_nodes_with_shanghai_drug_price_20260820_full.jsonl"
)
DEFAULT_RUNTIME_INDEX_DIR = (
    DEFAULT_RAG_READY_ROOT
    / "indexes"
    / "faiss_bge_m3_flat_ip_with_shanghai_drug_price_20260820_full"
)
DEFAULT_METADATA_V2_NODES_PATH = (
    DEFAULT_RAG_READY_ROOT / "nodes" / "policy_nodes_metadata_v2.jsonl"
)
DEFAULT_METADATA_V2_INDEX_DIR = (
    DEFAULT_RAG_READY_ROOT / "indexes" / "faiss_bge_m3_flat_ip_metadata_v2"
)
POLICY_RAG_INDEX_VERSION = str(
    os.environ.get("MEDIGUARD_POLICY_RAG_INDEX_VERSION") or "v1"
).strip().lower()
_VERSIONED_NODES_PATH = (
    DEFAULT_METADATA_V2_NODES_PATH
    if POLICY_RAG_INDEX_VERSION == "v2"
    else DEFAULT_RUNTIME_NODES_PATH
)
_VERSIONED_INDEX_DIR = (
    DEFAULT_METADATA_V2_INDEX_DIR
    if POLICY_RAG_INDEX_VERSION == "v2"
    else DEFAULT_RUNTIME_INDEX_DIR
)
DEFAULT_NODES_PATH = _path_from_env(
    "MEDIGUARD_POLICY_RAG_NODES_PATH",
    _VERSIONED_NODES_PATH,
)
DEFAULT_INDEX_DIR = _path_from_env(
    "MEDIGUARD_POLICY_RAG_INDEX_DIR",
    _VERSIONED_INDEX_DIR,
)
DEFAULT_INDEX_PATH = _path_from_env(
    "MEDIGUARD_POLICY_RAG_INDEX_PATH",
    DEFAULT_INDEX_DIR / "policy.index",
)
DEFAULT_INDEX_NODE_IDS_PATH = _path_from_env(
    "MEDIGUARD_POLICY_RAG_INDEX_NODE_IDS_PATH",
    DEFAULT_INDEX_DIR / "index_node_ids.jsonl",
)
DEFAULT_INDEX_MANIFEST_PATH = _path_from_env(
    "MEDIGUARD_POLICY_RAG_INDEX_MANIFEST_PATH",
    DEFAULT_INDEX_DIR / "index_manifest.json",
)
DEFAULT_EMBEDDING_DIR = DEFAULT_RAG_READY_ROOT / "embeddings" / "policy_bge_m3"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_EMBEDDING_DIR / "embeddings.npy"
DEFAULT_EMBEDDING_NODE_IDS_PATH = DEFAULT_EMBEDDING_DIR / "node_ids.jsonl"
DEFAULT_EMBEDDING_MANIFEST_PATH = DEFAULT_EMBEDDING_DIR / "embedding_manifest.json"
DEFAULT_QDRANT_MANIFEST_PATH = (
    DEFAULT_RAG_READY_ROOT / "indexes" / "qdrant_bge_m3" / "qdrant_manifest.json"
)
DEFAULT_VERSION_INDEX_PATH = (
    DEFAULT_RAG_READY_ROOT / "metadata" / "policy_version_index.jsonl"
)
DEFAULT_VERSION_REVIEW_INDEX_PATH = (
    DEFAULT_RAG_READY_ROOT / "metadata" / "policy_version_review_index.jsonl"
)
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "runtime" / "hf_cache"
