"""Node store for serialized policy RAG TextNode records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode


@dataclass(frozen=True, slots=True)
class PolicyNodeRecord:
    node_id: str
    text: str
    metadata: dict[str, Any]


class PolicyNodeStore:
    """Loads and filters policy TextNode records."""

    def __init__(self, nodes_path: Path) -> None:
        self._nodes_path = nodes_path
        self._nodes = self._load_nodes(nodes_path)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def nodes_path(self) -> Path:
        return self._nodes_path

    def records(self) -> tuple[PolicyNodeRecord, ...]:
        """Return loaded records in file order for secondary local indexes."""

        return tuple(self._nodes.values())

    def get(self, node_id: str) -> PolicyNodeRecord | None:
        return self._nodes.get(node_id)

    def require_all(self, node_ids: list[str]) -> None:
        missing = [node_id for node_id in node_ids if node_id not in self._nodes]
        if missing:
            raise RuntimeError(
                f"Policy node store is missing {len(missing)} indexed node IDs"
            )

    def matches_filters(
        self,
        record: PolicyNodeRecord,
        filters: dict[str, object],
    ) -> bool:
        metadata = record.metadata
        if not _matches_list_filter(metadata, "jurisdiction", filters.get("jurisdiction")):
            return False
        if not _matches_list_filter(metadata, "policy_domain", filters.get("policy_domain")):
            return False
        cite_filter = filters.get("can_cite_as_policy_basis")
        if (
            cite_filter is not None
            and metadata.get("can_cite_as_policy_basis") != cite_filter
        ):
            return False
        return True

    @staticmethod
    def _load_nodes(path: Path) -> dict[str, PolicyNodeRecord]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        nodes: dict[str, PolicyNodeRecord] = {}
        with path.open("r", encoding="utf-8") as file_obj:
            for line_no, line in enumerate(file_obj, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                node = TextNode.from_dict(payload)
                if not node.id_:
                    raise ValueError(f"TextNode at {path}:{line_no} is missing id_")
                if not node.text.strip():
                    raise ValueError(f"TextNode {node.id_} is missing text")
                nodes[node.id_] = PolicyNodeRecord(
                    node_id=node.id_,
                    text=node.text,
                    metadata=dict(node.metadata or {}),
                )
        if not nodes:
            raise RuntimeError(f"No policy nodes loaded from {path}")
        return nodes


def _matches_list_filter(
    metadata: dict[str, Any],
    key: str,
    filter_value: object,
) -> bool:
    values = [str(item) for item in filter_value or []]
    if not values:
        return True
    return str(metadata.get(key) or "") in values
