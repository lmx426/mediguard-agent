"""Map v2 candidate contexts back to local policy node IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_MODEL_CACHE,
    DEFAULT_NODES_PATH,
    DEFAULT_V2_CANDIDATES_PATH,
    DEFAULT_V2_MAPPED_PATH,
    SCHEMA_MAPPED_V2,
    best_evidence_text,
    derive_filters_from_refs,
    extract_anchor_terms,
    load_policy_nodes,
    read_jsonl,
    text_contains,
    versioned_eval_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map RAGAS/DeepSeek candidate contexts to local policy nodes."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--version-suffix", default=None)
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--mapped-jsonl", type=Path, default=None)
    parser.add_argument("--enable-semantic-fallback", action="store_true")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fetch-k", type=int, default=10)
    args = parser.parse_args()
    if args.candidates_jsonl is None:
        args.candidates_jsonl = (
            versioned_eval_path("candidates", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_CANDIDATES_PATH
        )
    if args.mapped_jsonl is None:
        args.mapped_jsonl = (
            versioned_eval_path("mapped", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_MAPPED_PATH
        )
    return args


def main() -> None:
    args = parse_args()
    nodes = load_policy_nodes(args.nodes_jsonl)
    candidates = read_jsonl(args.candidates_jsonl)
    mapper = CandidateNodeMapper(
        nodes,
        enable_semantic_fallback=args.enable_semantic_fallback,
        cache_folder=args.cache_folder,
        device=args.device,
        fetch_k=args.fetch_k,
    )
    mapped = [mapper.map_candidate(candidate) for candidate in candidates]
    write_jsonl(args.mapped_jsonl, mapped)
    mapped_context_count = sum(len(row.get("mapped_contexts") or []) for row in mapped)
    unmapped_context_count = sum(
        len(row.get("unmapped_contexts") or []) for row in mapped
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_count": len(mapped),
                "mapped_context_count": mapped_context_count,
                "unmapped_context_count": unmapped_context_count,
                "mapped_path": str(args.mapped_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


class CandidateNodeMapper:
    def __init__(
        self,
        nodes: dict[str, Any],
        *,
        enable_semantic_fallback: bool,
        cache_folder: Path,
        device: str | None,
        fetch_k: int,
    ) -> None:
        self._nodes = nodes
        self._by_source: dict[str, list[Any]] = {}
        self._by_doc: dict[str, list[Any]] = {}
        for node in nodes.values():
            source_id = str(node.metadata.get("source_id") or "")
            doc_id = str(node.metadata.get("doc_id") or "")
            if source_id:
                self._by_source.setdefault(source_id, []).append(node)
            if doc_id:
                self._by_doc.setdefault(doc_id, []).append(node)
        self._enable_semantic_fallback = enable_semantic_fallback
        self._cache_folder = cache_folder
        self._device = device
        self._fetch_k = fetch_k
        self._semantic_retriever = None

    def map_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        mapped_contexts: list[dict[str, Any]] = []
        unmapped_contexts: list[dict[str, Any]] = []
        for context in candidate.get("reference_contexts") or []:
            mapped = self.map_context(context)
            if mapped is None:
                unmapped_contexts.append(context)
            else:
                mapped_contexts.append(mapped)
        refs = [{"node_id": item["node_id"]} for item in mapped_contexts]
        output = dict(candidate)
        output["schema_version"] = SCHEMA_MAPPED_V2
        output["mapped_contexts"] = mapped_contexts
        output["unmapped_contexts"] = unmapped_contexts
        output["derived_filters"] = derive_filters_from_refs(refs, self._nodes)
        output["mapping_summary"] = {
            "reference_context_count": len(candidate.get("reference_contexts") or []),
            "mapped_context_count": len(mapped_contexts),
            "unmapped_context_count": len(unmapped_contexts),
            "semantic_fallback_enabled": self._enable_semantic_fallback,
        }
        return output

    def map_context(self, context: dict[str, Any]) -> dict[str, Any] | None:
        metadata = context.get("metadata") or {}
        context_text = str(context.get("text") or context.get("evidence_text") or "")
        node_id = str(metadata.get("node_id") or "").strip()
        if node_id and node_id in self._nodes:
            node = self._nodes[node_id]
            evidence_text = best_evidence_text(context_text, node.text)
            return self._mapped_ref(
                context=context,
                node=node,
                evidence_text=evidence_text,
                method="metadata_direct",
                confidence=1.0 if text_contains(node.text, context_text) else 0.88,
            )

        exact_node = self._exact_text_match(context_text, metadata)
        if exact_node is not None:
            evidence_text = best_evidence_text(context_text, exact_node.text)
            return self._mapped_ref(
                context=context,
                node=exact_node,
                evidence_text=evidence_text,
                method="exact_text",
                confidence=0.95,
            )

        if self._enable_semantic_fallback:
            semantic_node = self._semantic_match(context_text, metadata)
            if semantic_node is not None:
                evidence_text = best_evidence_text(context_text, semantic_node.text)
                return self._mapped_ref(
                    context=context,
                    node=semantic_node,
                    evidence_text=evidence_text,
                    method="semantic_match",
                    confidence=0.75,
                )
        return None

    def _exact_text_match(
        self,
        context_text: str,
        metadata: dict[str, Any],
    ) -> Any | None:
        candidates = self._candidate_nodes_for_metadata(metadata)
        for node in candidates:
            if text_contains(node.text, context_text):
                return node
        if candidates:
            return None
        for node in self._nodes.values():
            if text_contains(node.text, context_text):
                return node
        return None

    def _candidate_nodes_for_metadata(self, metadata: dict[str, Any]) -> list[Any]:
        source_id = str(metadata.get("source_id") or "").strip()
        doc_id = str(metadata.get("doc_id") or "").strip()
        candidates: list[Any] = []
        if source_id:
            candidates.extend(self._by_source.get(source_id, []))
        if doc_id:
            candidates.extend(self._by_doc.get(doc_id, []))
        unique: dict[str, Any] = {}
        for node in candidates:
            unique[node.node_id] = node
        return list(unique.values())

    def _semantic_match(self, context_text: str, metadata: dict[str, Any]) -> Any | None:
        if not context_text.strip():
            return None
        if self._semantic_retriever is None:
            from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
            from src.backend.infrastructure.policy_rag.faiss_retriever import (
                FaissPolicyRetriever,
            )

            embedder = BgeQueryEmbedder(
                cache_folder=self._cache_folder,
                device=self._device,
            )
            self._semantic_retriever = FaissPolicyRetriever(embedder=embedder)
        filters = {
            key: [metadata[key]]
            for key in ("jurisdiction", "policy_domain", "content_type")
            if metadata.get(key)
        }
        candidates = self._semantic_retriever.retrieve(
            question=context_text[:600],
            filters=filters,
            fetch_k=self._fetch_k,
        )
        expected_source = str(metadata.get("source_id") or "")
        expected_url = str(metadata.get("source_url") or "")
        for candidate in candidates:
            node = self._nodes.get(candidate.node_id)
            if node is None:
                continue
            if expected_source and node.metadata.get("source_id") == expected_source:
                return node
            if expected_url and node.metadata.get("source_url") == expected_url:
                return node
        return self._nodes.get(candidates[0].node_id) if candidates else None

    def _mapped_ref(
        self,
        *,
        context: dict[str, Any],
        node: Any,
        evidence_text: str,
        method: str,
        confidence: float,
    ) -> dict[str, Any]:
        metadata = node.metadata
        return {
            "context_id": str(context.get("context_id") or ""),
            "node_id": node.node_id,
            "source_id": str(metadata.get("source_id") or ""),
            "doc_id": str(metadata.get("doc_id") or ""),
            "source_url": str(metadata.get("source_url") or ""),
            "title": str(metadata.get("title") or ""),
            "jurisdiction": str(metadata.get("jurisdiction") or ""),
            "policy_domain": str(metadata.get("policy_domain") or ""),
            "content_type": str(metadata.get("content_type") or ""),
            "document_role": str(metadata.get("doc_type") or metadata.get("content_type") or ""),
            "evidence_text": evidence_text,
            "anchor_terms": extract_anchor_terms(
                str(context.get("text") or ""),
                evidence_text,
            ),
            "node_mapping_method": method,
            "node_mapping_confidence": round(confidence, 3),
        }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v2-map] failed: {exc}", file=sys.stderr)
        raise
