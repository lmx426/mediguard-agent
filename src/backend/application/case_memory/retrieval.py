"""Deterministic, type-specific retrieval projections for case memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ...domain.case_memory.entities import (
    MemoryLevel,
    MemoryQueryProfile,
    MemoryRecallRequest,
    MemoryRecord,
    MemoryType,
    MemoryVectorProjection,
)


RetrievalChannel = Literal["sql", "vector", "graph", "bm25"]


@dataclass(frozen=True)
class MemoryTypeRetrievalPolicy:
    channels: tuple[RetrievalChannel, ...]
    levels: tuple[MemoryLevel, ...]
    max_vector_candidates: int
    max_graph_candidates: int
    min_signal_required: bool = True


class MemoryTypeRetrievalPolicyRegistry:
    """Select retrieval lanes by purpose instead of forcing one universal path."""

    _policies = {
        MemoryType.INTENT_ROUTE: MemoryTypeRetrievalPolicy(
            channels=("sql", "vector", "graph", "bm25"),
            levels=(MemoryLevel.L2, MemoryLevel.L1),
            max_vector_candidates=12,
            max_graph_candidates=8,
        ),
        MemoryType.POLICY_SEARCH: MemoryTypeRetrievalPolicy(
            channels=("sql", "vector", "graph", "bm25"),
            levels=(MemoryLevel.L2, MemoryLevel.L1, MemoryLevel.L3),
            max_vector_candidates=16,
            max_graph_candidates=12,
        ),
        MemoryType.FAILURE: MemoryTypeRetrievalPolicy(
            channels=("sql", "graph", "vector", "bm25"),
            levels=(MemoryLevel.L2, MemoryLevel.L1),
            max_vector_candidates=8,
            max_graph_candidates=8,
        ),
        MemoryType.ANSWER_STYLE: MemoryTypeRetrievalPolicy(
            channels=("sql",),
            levels=(MemoryLevel.L3, MemoryLevel.L1),
            max_vector_candidates=0,
            max_graph_candidates=0,
            min_signal_required=False,
        ),
        MemoryType.DECISION_PLAN: MemoryTypeRetrievalPolicy(
            channels=("sql", "vector", "graph", "bm25"),
            levels=(MemoryLevel.L3, MemoryLevel.L2, MemoryLevel.L1),
            max_vector_candidates=12,
            max_graph_candidates=10,
        ),
    }

    def get(self, memory_type: MemoryType) -> MemoryTypeRetrievalPolicy:
        return self._policies[memory_type]


@dataclass(frozen=True)
class MemoryGraphRelation:
    node_type: str
    node_key: str
    label: str
    edge_type: str

    @property
    def anchor_key(self) -> str:
        return f"{self.node_type}:{self.node_key}"


def _unique(values: list[Any], *, limit: int = 24) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            continue
        text = str(value or "").strip()
        if not text:
            continue
        normalized = text.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text[:240])
        if len(result) >= limit:
            break
    return result


def _values(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique(value)
    if value in (None, ""):
        return []
    return _unique([value])


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sections(memory: MemoryRecord) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = memory.payload
    structured = _mapping(payload.get("structured_content"))
    recommended = _mapping(payload.get("recommended_action"))
    stable = _mapping(payload.get("stable_preferences"))
    return structured, recommended, stable


def memory_match_profile(memory: MemoryRecord) -> dict[str, Any]:
    """Read a v2 match profile, with safe legacy fallbacks for old records."""

    structured, recommended, stable = _sections(memory)
    profile = (
        _mapping(structured.get("match_profile"))
        or _mapping(recommended.get("match_profile"))
        or _mapping(stable.get("match_profile"))
    )
    result = dict(profile)
    sources = (structured, recommended, stable, memory.payload)
    aliases = {
        "task_type": ("task_type", "business_intent", "intent"),
        "task_pattern": ("task_pattern",),
        "action": ("action",),
        "target_objects": ("target_objects",),
        "evidence_need": ("evidence_need",),
        "jurisdiction": ("jurisdiction",),
        "policy_domain": ("policy_domain",),
        "information_needs": ("information_needs",),
        "required_capabilities": ("required_capabilities", "capabilities", "planning_steps"),
        "validation_codes": ("validation_codes", "validation_code"),
        "node": ("node",),
        "error_category": ("error_category", "error_type"),
        "scenario_type": ("scenario_type", "answer_shape", "granularity"),
        "component_version": ("component_version", "validation_contract_version"),
        "trigger_conditions": ("trigger_conditions", "applicable_conditions"),
    }
    filters = {}
    for source in sources:
        filters.update(_mapping(source.get("filters")))
    for target, keys in aliases.items():
        if result.get(target) not in (None, "", []):
            continue
        for source in sources:
            value = next((source.get(key) for key in keys if source.get(key) not in (None, "", [])), None)
            if value not in (None, "", []):
                result[target] = value
                break
    for key in ("jurisdiction", "policy_domain"):
        if result.get(key) in (None, "", []) and filters.get(key) not in (None, "", []):
            result[key] = filters[key]
    if memory.memory_level == MemoryLevel.L2:
        result.setdefault("trigger_conditions", memory.payload.get("applicable_conditions", []))
    return result


def memory_action_payload(memory: MemoryRecord) -> dict[str, Any]:
    structured, recommended, stable = _sections(memory)
    return (
        _mapping(structured.get("action_payload"))
        or _mapping(recommended.get("action_payload"))
        or _mapping(stable.get("action_payload"))
    )


class MemoryVectorProjectionBuilder:
    """Build retrieval text from applicability, not from the execution payload."""

    _labels = {
        "task_type": "任务类型",
        "task_pattern": "任务模式",
        "task_complexity": "任务复杂度",
        "action": "动作",
        "target_objects": "目标对象",
        "evidence_need": "证据需求",
        "jurisdiction": "地域",
        "policy_domain": "政策领域",
        "information_needs": "信息需求",
        "required_capabilities": "所需能力",
        "validation_codes": "校验码",
        "node": "节点",
        "error_category": "错误类别",
        "component_version": "组件版本",
        "trigger_conditions": "触发条件",
    }
    _fields = {
        MemoryType.INTENT_ROUTE: (
            "task_type", "task_pattern", "action", "target_objects", "evidence_need", "trigger_conditions"
        ),
        MemoryType.POLICY_SEARCH: (
            "task_type", "task_pattern", "target_objects", "jurisdiction", "policy_domain", "information_needs", "trigger_conditions"
        ),
        MemoryType.DECISION_PLAN: (
            "task_type", "task_pattern", "task_complexity", "action", "target_objects", "required_capabilities", "evidence_need", "trigger_conditions"
        ),
        MemoryType.FAILURE: (
            "node", "validation_codes", "error_category", "component_version", "trigger_conditions"
        ),
    }

    def build(self, memory: MemoryRecord) -> MemoryVectorProjection | None:
        if memory.memory_type == MemoryType.ANSWER_STYLE or memory.memory_level == MemoryLevel.L0:
            return None
        profile = memory_match_profile(memory)
        parts: list[str] = []
        for key in self._fields[memory.memory_type]:
            values = _values(profile.get(key))
            if values:
                parts.append(f"{self._labels[key]}：{'、'.join(values)}")
        if not parts:
            parts.append(f"适用经验：{memory.summary[:500]}")
        metadata = {
            "projection_version": "v2",
            "jurisdiction": _values(profile.get("jurisdiction")),
            "policy_domain": _values(profile.get("policy_domain")),
            "validation_codes": _values(profile.get("validation_codes")),
            "node": _values(profile.get("node")),
            "task_type": _values(profile.get("task_type")),
        }
        return MemoryVectorProjection(
            memory_id=memory.memory_id,
            embedding_text="；".join(parts)[:2000],
            metadata={key: value for key, value in metadata.items() if value not in (None, "", [])},
        )


class MemoryQueryBuilder:
    """Create a transient query that mirrors the stored vector projection."""

    _aliases = {
        "task_type": {"task_type", "intent", "business_intent"},
        "task_pattern": {"task_pattern"},
        "task_complexity": {"task_complexity", "complexity"},
        "action": {"action"},
        "target_objects": {"target_objects"},
        "evidence_need": {"evidence_need"},
        "jurisdiction": {"jurisdiction"},
        "policy_domain": {"policy_domain"},
        "information_needs": {"information_needs"},
        "required_capabilities": {"required_capabilities", "capabilities"},
        "validation_codes": {"validation_codes", "validation_code"},
        "node": {"node"},
        "error_category": {"error_category", "error_type"},
        "scenario_type": {"scenario_type", "answer_shape", "granularity"},
        "component_version": {"component_version", "validation_contract_version"},
        "trigger_conditions": {"trigger_conditions", "applicable_conditions"},
    }

    def build(self, request: MemoryRecallRequest) -> MemoryQueryProfile:
        collected: dict[str, list[str]] = {key: [] for key in self._aliases}

        def visit(value: Any, *, depth: int = 0) -> None:
            if depth > 5:
                return
            if isinstance(value, dict):
                for key, item in list(value.items())[:50]:
                    for target, aliases in self._aliases.items():
                        if str(key) in aliases:
                            collected[target].extend(_values(item))
                    visit(item, depth=depth + 1)
            elif isinstance(value, list):
                for item in value[:24]:
                    visit(item, depth=depth + 1)

        visit(request.task_context)
        profile = {key: _unique(values) for key, values in collected.items() if values}
        question = str(request.task_context.get("question") or "").strip()[:600]
        labels = MemoryVectorProjectionBuilder._labels
        fields = MemoryVectorProjectionBuilder._fields.get(request.memory_type, ())
        parts = [f"当前任务描述：{question}"] if question and request.memory_type != MemoryType.ANSWER_STYLE else []
        for key in fields:
            values = _values(profile.get(key))
            if values:
                parts.append(f"{labels[key]}：{'、'.join(values)}")
        exact_filters: dict[str, list[str]] = {}
        for key in (
            "jurisdiction",
            "policy_domain",
            "validation_codes",
            "node",
            "error_category",
            "component_version",
        ):
            values = _values(profile.get(key))
            if values:
                exact_filters[key] = values
        result = MemoryQueryProfile(
            memory_type=request.memory_type,
            query_text="；".join(parts)[:2000],
            match_profile=profile,
            exact_filters=exact_filters,
        )
        result.graph_anchor_keys = MemoryGraphProjectionRegistry().request_anchor_keys(result)
        return result


class MemoryGraphProjectionRegistry:
    """Deny-by-default Graph projection owned by memory type."""

    EXPANDABLE_EDGE_TYPES = {
        "MATCHES_TASK",
        "MATCHES_ACTION",
        "TARGETS",
        "ROUTES_TO",
        "APPLIES_IN",
        "HAS_POLICY_DOMAIN",
        "HAS_INFORMATION_NEED",
        "HAS_FILTER",
        "FOR_TASK",
        "USES_CAPABILITY",
        "HAS_CONSTRAINT",
        "OCCURS_AT",
        "TRIGGERED_BY",
        "REPAIRS_WITH",
        "APPLIES_TO_VERSION",
        "HAS_PREFERENCE",
        "HAS_OVERRIDE",
        "DERIVED_FROM",
        "SUMMARIZES",
        "STABILIZED_FROM",
        "SUPERSEDES",
        "CONFLICTS_WITH",
    }

    def relations(self, memory: MemoryRecord) -> list[MemoryGraphRelation]:
        if memory.memory_level == MemoryLevel.L0:
            return []
        profile = memory_match_profile(memory)
        action_payload = memory_action_payload(memory)
        structured, recommended, stable = _sections(memory)
        relations: list[MemoryGraphRelation] = []

        def add(node_type: str, prefix: str, value: Any, edge_type: str) -> None:
            for item in _values(value):
                relations.append(
                    MemoryGraphRelation(
                        node_type=node_type,
                        node_key=f"{prefix}:{item.casefold()}",
                        label=item,
                        edge_type=edge_type,
                    )
                )

        if memory.memory_type == MemoryType.INTENT_ROUTE:
            add("task_type", "task_type", profile.get("task_type"), "MATCHES_TASK")
            add("action", "action", profile.get("action"), "MATCHES_ACTION")
            add("target_object", "target_object", profile.get("target_objects"), "TARGETS")
            add("intent", "intent", action_payload.get("intent") or structured.get("route_tags"), "MATCHES_TASK")
            add("capability", "capability", action_payload.get("capabilities") or structured.get("capabilities"), "ROUTES_TO")
        elif memory.memory_type == MemoryType.POLICY_SEARCH:
            add("jurisdiction", "jurisdiction", profile.get("jurisdiction"), "APPLIES_IN")
            add("policy_domain", "policy_domain", profile.get("policy_domain"), "HAS_POLICY_DOMAIN")
            add("target_object", "target_object", profile.get("target_objects"), "TARGETS")
            add("information_need", "information_need", profile.get("information_needs"), "HAS_INFORMATION_NEED")
            filters = {}
            for section in (structured, recommended, stable):
                filters.update(_mapping(section.get("filters")))
                filters.update(_mapping(_mapping(section.get("policy_search")).get("filters")))
            for key, value in sorted(filters.items()):
                for item in _values(value):
                    add("filter", "filter", f"{key}={item}", "HAS_FILTER")
        elif memory.memory_type == MemoryType.DECISION_PLAN:
            add("task_type", "task_type", profile.get("task_type"), "FOR_TASK")
            add("target_object", "target_object", profile.get("target_objects"), "TARGETS")
            capabilities = profile.get("required_capabilities") or structured.get("planning_steps") or recommended.get("planning_steps") or memory.payload.get("standard_steps")
            add("capability", "capability", capabilities, "USES_CAPABILITY")
            add("constraint", "constraint", structured.get("constraints") or recommended.get("constraints") or memory.payload.get("constraints"), "HAS_CONSTRAINT")
        elif memory.memory_type == MemoryType.FAILURE:
            add("node", "node", profile.get("node"), "OCCURS_AT")
            add("validation_code", "validation_code", profile.get("validation_codes"), "TRIGGERED_BY")
            add("error_category", "error_category", profile.get("error_category"), "TRIGGERED_BY")
            add("component_version", "component_version", profile.get("component_version"), "APPLIES_TO_VERSION")
            repair = structured.get("repair_strategy") or recommended.get("repair_strategy") or action_payload.get("repair_strategy")
            add("repair_strategy", "repair_strategy", repair, "REPAIRS_WITH")
        elif memory.memory_type == MemoryType.ANSWER_STYLE and memory.memory_level == MemoryLevel.L3:
            preferences = _mapping(stable.get("style_preferences")) or stable
            for key, value in sorted(preferences.items()):
                if isinstance(value, (dict, list)):
                    continue
                add("answer_preference", "answer_preference", f"{key}={value}", "HAS_PREFERENCE")
            add("override_rule", "override_rule", memory.payload.get("override_rules"), "HAS_OVERRIDE")

        deduped: dict[tuple[str, str, str], MemoryGraphRelation] = {}
        for relation in relations:
            deduped[(relation.node_type, relation.node_key, relation.edge_type)] = relation
        return list(deduped.values())

    def request_anchor_keys(self, query: MemoryQueryProfile) -> list[str]:
        profile = query.match_profile
        relations: list[MemoryGraphRelation] = []

        def add(node_type: str, prefix: str, value: Any, edge_type: str) -> None:
            for item in _values(value):
                relations.append(MemoryGraphRelation(node_type, f"{prefix}:{item.casefold()}", item, edge_type))

        if query.memory_type == MemoryType.INTENT_ROUTE:
            add("task_type", "task_type", profile.get("task_type"), "MATCHES_TASK")
            add("action", "action", profile.get("action"), "MATCHES_ACTION")
            add("target_object", "target_object", profile.get("target_objects"), "TARGETS")
        elif query.memory_type == MemoryType.POLICY_SEARCH:
            add("jurisdiction", "jurisdiction", profile.get("jurisdiction"), "APPLIES_IN")
            add("policy_domain", "policy_domain", profile.get("policy_domain"), "HAS_POLICY_DOMAIN")
            add("target_object", "target_object", profile.get("target_objects"), "TARGETS")
            add("information_need", "information_need", profile.get("information_needs"), "HAS_INFORMATION_NEED")
        elif query.memory_type == MemoryType.DECISION_PLAN:
            add("task_type", "task_type", profile.get("task_type"), "FOR_TASK")
            add("target_object", "target_object", profile.get("target_objects"), "TARGETS")
            add("capability", "capability", profile.get("required_capabilities"), "USES_CAPABILITY")
        elif query.memory_type == MemoryType.FAILURE:
            add("node", "node", profile.get("node"), "OCCURS_AT")
            add("validation_code", "validation_code", profile.get("validation_codes"), "TRIGGERED_BY")
            add("error_category", "error_category", profile.get("error_category"), "TRIGGERED_BY")
            add("component_version", "component_version", profile.get("component_version"), "APPLIES_TO_VERSION")
        return list(dict.fromkeys(relation.anchor_key for relation in relations))


def matches_exact_constraints(memory: MemoryRecord, query: MemoryQueryProfile) -> bool:
    """Reject only explicit contradictions; missing legacy fields remain eligible."""

    profile = memory_match_profile(memory)

    def conflicts(key: str) -> bool:
        requested = {value.casefold() for value in _values(query.exact_filters.get(key))}
        stored = {value.casefold() for value in _values(profile.get(key))}
        return bool(requested and stored and requested.isdisjoint(stored))

    if memory.memory_type == MemoryType.POLICY_SEARCH:
        for key in ("jurisdiction", "policy_domain"):
            if conflicts(key):
                return False
    if memory.memory_type == MemoryType.FAILURE:
        for key in ("validation_codes", "node", "error_category", "component_version"):
            if conflicts(key):
                return False
    return True
