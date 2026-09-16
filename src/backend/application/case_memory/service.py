"""Use cases for capture, governance, retrieval and maturity promotion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from ...application.ports.case_memory import (
    EntityGraphProjectionRepository,
    MemoMemoryPort,
    MemoProjectionError,
    MemoryEventRepository,
    MemoryOutboxRepository,
    MemoryRepository,
)
from ...application.ports.repositories import UserRepository
from ...domain.case_memory.entities import (
    ConsumerViewPolicy,
    L0SourceEventPayload,
    L1AtomicMemoryPayload,
    L2ScenarioMemoryPayload,
    L3StableProfileOrPlaybookPayload,
    MemoryAdmissionDecision,
    MemoryHintPack,
    MemoryLevel,
    MemoryPreview,
    MemoryPresentationItem,
    MemoryRecallRequest,
    MemoryRecord,
    MemorySafeDetail,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from .retrieval import (
    MemoryQueryBuilder,
    MemoryTypeRetrievalPolicyRegistry,
    MemoryVectorProjectionBuilder,
    matches_exact_constraints,
    memory_action_payload,
    memory_match_profile,
)


ACTIVE_RECALL_STATUSES = {MemoryStatus.ACTIVE}
PREFETCH_CONSUMERS: dict[MemoryType, tuple[str, ...]] = {
    MemoryType.INTENT_ROUTE: ("intent_router",),
    MemoryType.POLICY_SEARCH: ("policy_filter_resolver", "expert_analysis"),
    MemoryType.ANSWER_STYLE: ("answer_generator",),
    MemoryType.DECISION_PLAN: ("decision_planner",),
}
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", str(value)):
        token = token.lower()
        if len(token) <= 1:
            continue
        tokens.add(token)
        if re.search(r"[\u4e00-\u9fff]", token):
            tokens.update(token[index] for index in range(len(token)))
            tokens.update(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_text(memory: MemoryRecord) -> str:
    return " ".join([memory.summary, str(memory.payload), memory.memory_type.value])


class MemoryViewRegistry:
    """Static allow-list registry owned by MediGuard, never by the LLM."""

    def __init__(self) -> None:
        self._policies: dict[tuple[str, MemoryType, MemoryLevel], ConsumerViewPolicy] = {}
        self._register_defaults()

    def register(self, policy: ConsumerViewPolicy) -> None:
        self._policies[(policy.consumer, policy.memory_type, policy.memory_level)] = policy

    def get_policy(self, consumer: str, memory_type: MemoryType, level: MemoryLevel) -> ConsumerViewPolicy:
        exact = self._policies.get((consumer, memory_type, level))
        if exact is not None:
            return exact
        fallback = self._policies.get(("default", memory_type, level))
        if fallback is not None:
            return fallback.model_copy(update={"consumer": consumer})
        return ConsumerViewPolicy(
            policy_id="deny-by-default",
            consumer=consumer,
            memory_type=memory_type,
            memory_level=level,
        )

    def _register_defaults(self) -> None:
        def add(
            consumer: str,
            memory_type: MemoryType,
            level: MemoryLevel,
            *,
            control: list[str] | None = None,
            prompt: list[str] | None = None,
            tool: list[str] | None = None,
            trace: list[str] | None = None,
        ) -> None:
            self.register(
                ConsumerViewPolicy(
                    policy_id=f"{consumer}:{memory_type.value}:{level.value}",
                    consumer=consumer,
                    memory_type=memory_type,
                    memory_level=level,
                    control_fields=control or [],
                    prompt_fields=prompt or [],
                    tool_param_fields=tool or [],
                    trace_fields=trace or [],
                )
            )

        for level in (MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.L3):
            add(
                "intent_router",
                MemoryType.INTENT_ROUTE,
                level,
                control=[
                    "summary",
                    "scenario_summary",
                    "profile_or_playbook_summary",
                    "structured_content.route_tags",
                    "recommended_action.route_tags",
                    "stable_preferences.route_tags",
                ],
                trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"],
            )
            add(
                "decision_planner",
                MemoryType.DECISION_PLAN,
                level,
                control=["summary", "scenario_summary", "profile_or_playbook_summary", "standard_steps"],
                prompt=["detail", "scenario_summary", "profile_or_playbook_summary"],
                tool=[
                    "structured_content.planning_steps",
                    "structured_content.constraints",
                    "recommended_action.planning_steps",
                    "recommended_action.constraints",
                    "recommended_action.standard_steps",
                    "stable_preferences.planning_steps",
                    "constraints",
                ],
                trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"],
            )
            add(
                "answer_generator",
                MemoryType.ANSWER_STYLE,
                level,
                prompt=["summary", "detail", "scenario_summary", "profile_or_playbook_summary"],
                tool=[
                    "structured_content.style_preferences",
                    "recommended_action.style_preferences",
                    "stable_preferences",
                    "override_rules",
                ],
                trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"],
            )
            add(
                "recovery_handler",
                MemoryType.FAILURE,
                level,
                control=["summary", "scenario_summary", "profile_or_playbook_summary"],
                tool=[
                    "structured_content.repair_strategy",
                    "structured_content.avoid",
                    "structured_content.recover",
                    "recommended_action.repair_strategy",
                    "recommended_action.avoid",
                    "recommended_action.recover",
                ],
                prompt=["detail", "scenario_summary"],
                trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"],
            )
            policy_fields = [
                "filters",
                "information_needs",
                "coverage_requirements",
                "evidence_focus",
                "citation_expectations",
                "avoid_claims",
            ]
            add(
                "policy_filter_resolver",
                MemoryType.POLICY_SEARCH,
                level,
                control=["summary", "scenario_summary", "profile_or_playbook_summary"],
                tool=[
                    *(f"structured_content.{field}" for field in policy_fields),
                    *(f"recommended_action.{field}" for field in policy_fields),
                    "stable_preferences.policy_search",
                ],
                prompt=["summary", "detail", "scenario_summary", "applicable_conditions", "exception_rules"],
                trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"],
            )
            add(
                "expert_analysis",
                MemoryType.POLICY_SEARCH,
                level,
                prompt=["summary", "detail", "scenario_summary", "applicable_conditions", "exception_rules"],
                tool=[
                    *(f"structured_content.{field}" for field in policy_fields),
                    *(f"recommended_action.{field}" for field in policy_fields),
                    "stable_preferences.policy_search",
                ],
                trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"],
            )

        for memory_type in MemoryType:
            for level in (MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.L3):
                if ("default", memory_type, level) not in self._policies:
                    add("default", memory_type, level, prompt=["summary"], trace=["source_refs", "supporting_l1_refs", "supporting_l2_refs"])


class LevelRecallPolicyRegistry:
    """Per-purpose maturity order; this is not a second memory taxonomy."""

    def __init__(self) -> None:
        self._registry = MemoryTypeRetrievalPolicyRegistry()

    def get(self, memory_type: MemoryType) -> tuple[MemoryLevel, ...]:
        return self._registry.get(memory_type).levels


class RequestMemoryCache:
    """Request-lifetime candidate cache, never a prompt or a durable store."""

    def __init__(self) -> None:
        self._items: dict[str, dict[MemoryType, list[MemoryRecord]]] = {}
        self._statuses: dict[str, dict[MemoryType, str]] = {}
        self._created_at: dict[str, datetime] = {}
        self._lock = RLock()

    def put(self, request_id: str, memories: Iterable[MemoryRecord]) -> None:
        grouped: dict[MemoryType, list[MemoryRecord]] = defaultdict(list)
        for memory in memories:
            grouped[memory.memory_type].append(memory)
        with self._lock:
            self._items[request_id] = dict(grouped)
            self._statuses[request_id] = {
                memory_type: "hit" if rows else "empty"
                for memory_type, rows in grouped.items()
            }
            self._created_at[request_id] = _now()

    def begin(self, request_id: str, memory_type: MemoryType) -> str:
        """Mark one request/type as in flight unless it already reached a terminal state."""

        with self._lock:
            self._expire_locked(request_id)
            current = self._statuses.get(request_id, {}).get(memory_type, "not_requested")
            if current != "not_requested":
                return current
            self._statuses.setdefault(request_id, {})[memory_type] = "in_flight"
            self._items.setdefault(request_id, {})
            self._created_at[request_id] = _now()
            return "in_flight"

    def complete(
        self,
        request_id: str,
        memory_type: MemoryType,
        memories: Iterable[MemoryRecord],
    ) -> str:
        """Commit an async result only while the request is still waiting for it."""

        values = list(memories)
        with self._lock:
            self._expire_locked(request_id)
            current = self._statuses.get(request_id, {}).get(memory_type, "not_requested")
            if current != "in_flight":
                return current
            status = "hit" if values else "empty"
            self._items.setdefault(request_id, {})[memory_type] = values
            self._statuses.setdefault(request_id, {})[memory_type] = status
            self._created_at[request_id] = _now()
            return status

    def mark_terminal(
        self,
        request_id: str,
        memory_type: MemoryType,
        status: str,
    ) -> str:
        if status not in {"empty", "timed_out", "failed"}:
            raise ValueError(f"unsupported request memory terminal status: {status}")
        with self._lock:
            self._expire_locked(request_id)
            current = self._statuses.get(request_id, {}).get(memory_type, "not_requested")
            if current in {"hit", "empty", "timed_out", "failed"}:
                return current
            self._statuses.setdefault(request_id, {})[memory_type] = status
            self._items.setdefault(request_id, {}).setdefault(memory_type, [])
            self._created_at[request_id] = _now()
            return status

    def status(self, request_id: str, memory_type: MemoryType) -> str:
        with self._lock:
            self._expire_locked(request_id)
            return self._statuses.get(request_id, {}).get(memory_type, "not_requested")

    def get(self, request_id: str, memory_type: MemoryType | None = None) -> list[MemoryRecord]:
        with self._lock:
            self._expire_locked(request_id)
            typed = self._items.get(request_id, {})
            if memory_type is not None:
                return list(typed.get(memory_type, []))
            return [item for values in typed.values() for item in values]

    def has(self, request_id: str, memory_type: MemoryType | None = None) -> bool:
        with self._lock:
            self._expire_locked(request_id)
            if memory_type is None:
                return request_id in self._statuses
            return self._statuses.get(request_id, {}).get(memory_type) is not None

    def clear(self, request_id: str) -> None:
        with self._lock:
            self._items.pop(request_id, None)
            self._statuses.pop(request_id, None)
            self._created_at.pop(request_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._items.clear()
            self._statuses.clear()
            self._created_at.clear()

    def _expire_locked(self, request_id: str) -> None:
        created_at = self._created_at.get(request_id)
        if created_at is None or _now() - created_at <= timedelta(minutes=10):
            return
        self._items.pop(request_id, None)
        self._statuses.pop(request_id, None)
        self._created_at.pop(request_id, None)


class PromotionSanitizer:
    """Deterministic checks that stop private preferences becoming shared policy."""

    def validate_l1_to_l2(
        self,
        memories: list[MemoryRecord],
        target_scope: MemoryScope | None = None,
    ) -> tuple[bool, str]:
        if len(memories) < 2:
            return False, "at least two similar L1 memories are required"
        if target_scope and target_scope.scope_type in {"team", "department", "global"}:
            auditor_scopes = {
                memory.scope.scope_id
                for memory in memories
                if memory.scope.scope_type == "auditor"
            }
            if len(auditor_scopes) < 3:
                return False, "team/global promotion requires three distinct auditor observations"
        if len({memory.scope.scope_id for memory in memories}) < 2 and target_scope and target_scope.scope_type != "auditor":
            return False, "team/global promotion requires three distinct auditor observations"
        return True, "validated"

    def validate_l2_to_l3(
        self,
        memories: list[MemoryRecord],
        target_scope: MemoryScope | None = None,
    ) -> tuple[bool, str]:
        if len(memories) < 2:
            return False, "at least two stable L2 memories are required"
        if any(memory.conflict_count > 0 for memory in memories):
            return False, "conflicting memories cannot be stabilized"
        if target_scope and target_scope.scope_type in {"team", "department", "global"}:
            if len({memory.memory_id for memory in memories}) < 2:
                return False, "shared playbook promotion requires multiple supporting scenarios"
        return True, "validated"


class CaseMemoryService:
    """Single application facade for the CaseMemoryService."""

    def __init__(
        self,
        repository: MemoryRepository,
        events: MemoryEventRepository,
        outbox: MemoryOutboxRepository,
        memo: MemoMemoryPort,
        graph: EntityGraphProjectionRepository,
        users: UserRepository | None = None,
    ) -> None:
        self.repository = repository
        self.events = events
        self.outbox = outbox
        self.memo = memo
        self.graph = graph
        self.users = users
        self.views = MemoryViewRegistry()
        self.levels = LevelRecallPolicyRegistry()
        self.retrieval_policies = MemoryTypeRetrievalPolicyRegistry()
        self.query_builder = MemoryQueryBuilder()
        self.vector_builder = MemoryVectorProjectionBuilder()
        self.cache = RequestMemoryCache()
        self.sanitizer = PromotionSanitizer()
        self.consolidation_worker = MemoryConsolidationWorker(self)
        self._maintenance_stop = Event()
        self._maintenance_thread: Thread | None = None
        self._maintenance_lock = RLock()
        self._preference_lock = RLock()
        self._preference_cache: dict[str, tuple[bool, float]] = {}
        self._preference_cache_ttl_seconds = 5.0
        self._maintenance_last_error: str | None = None
        self._last_reconciliation_at: datetime | None = None
        self._last_consolidation_at: datetime | None = None

    def personal_memory_enabled(self, actor_id: str) -> bool:
        """Return the reviewer preference without coupling callers to user storage."""

        if self.users is None:
            return True
        now = monotonic()
        with self._preference_lock:
            cached = self._preference_cache.get(actor_id)
            if cached is not None and now - cached[1] < self._preference_cache_ttl_seconds:
                return cached[0]
        try:
            enabled = self.users.get_case_memory_enabled(actor_id)
        except Exception:
            enabled = False
        with self._preference_lock:
            self._preference_cache[actor_id] = (enabled, now)
        return enabled

    def set_personal_memory_enabled(self, actor_id: str, enabled: bool) -> bool:
        if self.users is None:
            raise RuntimeError("personal memory preference storage is unavailable")
        persisted = self.users.set_case_memory_enabled(actor_id, enabled)
        with self._preference_lock:
            self._preference_cache[actor_id] = (persisted, monotonic())
        return persisted

    def _scope_memory_enabled(self, scope: MemoryScope) -> bool:
        return (
            scope.scope_type != "auditor"
            or self.personal_memory_enabled(scope.scope_id)
        )

    def capture(
        self,
        *,
        memory_type: MemoryType,
        level: MemoryLevel,
        scope: MemoryScope,
        payload: dict[str, Any],
        source_event_ids: list[str],
        allowed_consumers: list[str],
        confidence: float = 0.0,
        feature: str | None = None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        if memory_id:
            existing = self.repository.get(memory_id)
            if existing is not None:
                return self._record_repeat_observation(
                    existing,
                    payload=payload,
                    source_event_ids=source_event_ids,
                )
        admission = (
            MemoryAdmissionDecision(
                policy="human_review",
                status=MemoryStatus.CANDIDATE,
                reason="L0 is an audit anchor and never enters active recall",
                importance_score=0.0,
            )
            if level == MemoryLevel.L0
            else self.admission_decision(
                memory_type=memory_type,
                payload=payload,
                confidence=confidence,
                feature=feature,
            )
        )
        memory = MemoryRecord(
            memory_id=memory_id or f"mem_{uuid4().hex}",
            memory_type=memory_type,
            memory_level=level,
            scope=scope,
            status=admission.status,
            payload=payload,
            allowed_consumers=allowed_consumers,
            admission_policy=admission.policy,
            admission_confidence=confidence,
            importance_score=admission.importance_score,
            shadow_observation_count=1,
            projection_sync_status=(
                "pending" if admission.status == MemoryStatus.ACTIVE else "not_applicable"
            ),
            source_event_ids=source_event_ids,
        )
        self.repository.save(memory)
        self.events.append(memory.memory_id, "MEMORY_CAPTURED", None, {"status": memory.status.value, "memory_level": level.value})
        if memory.status == MemoryStatus.ACTIVE:
            self.outbox.enqueue("MEMORY_STATUS_CHANGED", memory.memory_id, {"status": memory.status.value})
        return memory

    def _record_repeat_observation(
        self,
        memory: MemoryRecord,
        *,
        payload: dict[str, Any],
        source_event_ids: list[str],
    ) -> MemoryRecord:
        new_source_ids = [
            item for item in source_event_ids
            if item and item not in memory.source_event_ids
        ]
        provenance_key = {
            MemoryLevel.L1: "derived_from_l0_refs",
            MemoryLevel.L2: "supporting_l1_refs",
            MemoryLevel.L3: "supporting_l2_refs",
        }.get(memory.memory_level)
        incoming_refs = (
            [str(item) for item in payload.get(provenance_key, [])]
            if provenance_key
            else []
        )
        existing_refs = (
            [str(item) for item in memory.payload.get(provenance_key, [])]
            if provenance_key
            else []
        )
        new_refs = [item for item in incoming_refs if item not in existing_refs]
        if not new_source_ids and not new_refs:
            return memory
        memory.source_event_ids.extend(new_source_ids)
        if provenance_key and new_refs:
            memory.payload[provenance_key] = [*existing_refs, *new_refs]
        if memory.memory_level == MemoryLevel.L1:
            existing_source_refs = [str(item) for item in memory.payload.get("source_refs", [])]
            incoming_source_refs = [str(item) for item in payload.get("source_refs", [])]
            memory.payload["source_refs"] = list(
                dict.fromkeys([*existing_source_refs, *incoming_source_refs])
            )[:50]
        memory.shadow_observation_count += 1
        if memory.status == MemoryStatus.SHADOW and memory.shadow_observation_count >= 3:
            memory.status = MemoryStatus.CANDIDATE
            self.events.append(
                memory.memory_id,
                "MEMORY_SHADOW_THRESHOLD_REACHED",
                None,
                {"observation_count": memory.shadow_observation_count},
            )
        if memory.status == MemoryStatus.ACTIVE:
            memory.projection_sync_status = "pending"
        self.repository.update(memory)
        if memory.status == MemoryStatus.ACTIVE:
            self.outbox.enqueue(
                "MEMORY_STATUS_CHANGED",
                memory.memory_id,
                {"status": memory.status.value, "action": "observation_added"},
            )
        self.events.append(
            memory.memory_id,
            "MEMORY_OBSERVED",
            None,
            {"observation_count": memory.shadow_observation_count},
        )
        return memory

    def enqueue_capture_request(
        self,
        *,
        source_run_id: str,
        scope: MemoryScope,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Persist one small, redacted capture request without doing projection work."""

        if not self._scope_memory_enabled(scope):
            return None

        drafts: list[dict[str, Any]] = []
        for observation in observations[:8]:
            if not isinstance(observation, dict):
                continue
            memory_type = MemoryType(str(observation.get("memory_type") or ""))
            event = observation.get("event")
            atomic = observation.get("memory")
            if not isinstance(event, dict) or not isinstance(atomic, dict):
                continue
            source_identity_payload = {
                "source_run_id": source_run_id,
                "memory_type": memory_type.value,
                "event": event,
                "scope": scope.model_dump(mode="json"),
            }
            source_fingerprint = _hash(json.dumps(source_identity_payload, ensure_ascii=False, sort_keys=True, default=str))
            memory_identity_payload = {
                "memory_type": memory_type.value,
                "summary": atomic.get("summary"),
                "structured_content": atomic.get("structured_content"),
                "detail": atomic.get("detail"),
                "scope": scope.model_dump(mode="json"),
                "allowed_consumers": sorted(str(item) for item in observation.get("allowed_consumers", [])),
                "feature": observation.get("feature"),
            }
            memory_fingerprint = _hash(json.dumps(memory_identity_payload, ensure_ascii=False, sort_keys=True, default=str))
            l0_id = f"mem_l0_{source_fingerprint[:32]}"
            l1_id = f"mem_l1_{memory_fingerprint[:32]}"
            l0_payload = L0SourceEventPayload(
                event_type=str(event.get("event_type") or "memory_observation")[:80],
                source_run_id=source_run_id,
                source_event_id=str(event.get("source_event_id") or source_fingerprint)[:160],
                node=str(event.get("node") or "case_agent")[:80],
                validation_codes=[str(item)[:80] for item in event.get("validation_codes", [])[:12]],
                source_ref_ids=[str(item)[:160] for item in event.get("source_ref_ids", [])[:20]],
                derived_from=[],
            ).model_dump(mode="json")
            l1_payload = L1AtomicMemoryPayload(
                summary=str(atomic.get("summary") or "")[:500],
                structured_content=(
                    dict(atomic.get("structured_content") or {})
                    if isinstance(atomic.get("structured_content"), dict)
                    else {}
                ),
                detail=str(atomic.get("detail") or "")[:1000],
                source_refs=[str(item)[:160] for item in atomic.get("source_refs", [])[:20]],
                derived_from_l0_refs=[l0_id],
            ).model_dump(mode="json")
            allowed_consumers = [
                str(item)[:80]
                for item in observation.get("allowed_consumers", [])[:12]
                if str(item).strip()
            ]
            confidence = max(0.0, min(1.0, float(observation.get("confidence") or 0.0)))
            feature = str(observation.get("feature") or "runtime_capture")[:80]
            drafts.extend(
                [
                    {
                        "memory_id": l0_id,
                        "memory_type": memory_type.value,
                        "memory_level": MemoryLevel.L0.value,
                        "scope": scope.model_dump(mode="json"),
                        "payload": l0_payload,
                        "source_event_ids": [],
                        "allowed_consumers": allowed_consumers,
                        "confidence": confidence,
                        "feature": feature,
                    },
                    {
                        "memory_id": l1_id,
                        "memory_type": memory_type.value,
                        "memory_level": MemoryLevel.L1.value,
                        "scope": scope.model_dump(mode="json"),
                        "payload": l1_payload,
                        "source_event_ids": [l0_id],
                        "allowed_consumers": allowed_consumers,
                        "confidence": confidence,
                        "feature": feature,
                    },
                ]
            )
        if not drafts:
            return None
        request_fingerprint = _hash(
            json.dumps([draft["memory_id"] for draft in drafts], sort_keys=True)
        )
        return self.outbox.enqueue(
            "MEMORY_CAPTURE_REQUESTED",
            f"capture_{request_fingerprint[:32]}",
            {"source_run_id": source_run_id, "drafts": drafts},
        )

    def admission_decision(self, *, memory_type: MemoryType, payload: dict[str, Any], confidence: float, feature: str | None) -> MemoryAdmissionDecision:
        if memory_type == MemoryType.POLICY_SEARCH or feature == "policy_search":
            return MemoryAdmissionDecision(policy="human_review", status=MemoryStatus.CANDIDATE, reason="retrieval filters and information needs require explicit review", importance_score=0.6)
        if feature == "deterministic_structure_repair":
            return MemoryAdmissionDecision(policy="auto_active", status=MemoryStatus.ACTIVE, reason="deterministic repair passed validation", importance_score=0.25)
        if memory_type == MemoryType.ANSWER_STYLE or feature == "style_preference":
            return MemoryAdmissionDecision(policy="shadow", status=MemoryStatus.SHADOW, reason="style preference needs repeated observation", importance_score=0.35)
        return MemoryAdmissionDecision(policy="human_review", status=MemoryStatus.CANDIDATE, reason="default human review admission", importance_score=max(0.4, confidence))

    def mark_used(self, memory_id: str) -> None:
        """Record lightweight usage telemetry without changing memory truth."""

        memory = self.repository.get(memory_id)
        if memory is None or memory.status != MemoryStatus.ACTIVE:
            return
        memory.usage_count += 1
        memory.last_used_at = _now()
        memory.freshness_score = self._freshness(memory)
        self.repository.update(memory)

    def recall(self, request: MemoryRecallRequest) -> MemoryHintPack:
        if not self._scope_memory_enabled(request.scope):
            return MemoryHintPack(
                request_id=request.request_id,
                consumer=request.consumer,
                memory_type=request.memory_type,
            )
        ranked = self._ranked_candidates(request)
        return self._build_hint_pack(request, ranked[: request.max_items])

    def prefetch_request(
        self,
        *,
        request_id: str,
        scope: MemoryScope,
        task_context: dict[str, Any],
        max_items_per_type: int = 8,
    ) -> dict[str, int]:
        """Retrieve a request-wide candidate set once for downstream node views."""

        for memory_type in PREFETCH_CONSUMERS:
            self.cache.begin(request_id, memory_type)

        def retrieve(memory_type: MemoryType) -> tuple[MemoryType, int]:
            consumers = PREFETCH_CONSUMERS[memory_type]
            request = MemoryRecallRequest(
                request_id=request_id,
                consumer=consumers[0],
                memory_type=memory_type,
                scope=scope,
                task_context=task_context,
                max_items=max_items_per_type,
            )
            count = self.prefetch_type(
                request,
                allowed_consumers=set(consumers),
            )
            return memory_type, count

        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=len(PREFETCH_CONSUMERS), thread_name_prefix="memory-prefetch") as executor:
            for memory_type, count in executor.map(retrieve, PREFETCH_CONSUMERS):
                counts[memory_type.value] = count
        return counts

    def prefetch_type(
        self,
        request: MemoryRecallRequest,
        *,
        allowed_consumers: set[str] | None = None,
    ) -> int:
        """Fetch one planned memory type and commit its request-local terminal state."""

        self.cache.begin(request.request_id, request.memory_type)
        try:
            if not self._scope_memory_enabled(request.scope):
                self.cache.complete(request.request_id, request.memory_type, [])
                return 0
            rows = self._ranked_candidates(
                request,
                allowed_consumers=allowed_consumers,
            )[: request.max_items]
            self.cache.complete(request.request_id, request.memory_type, rows)
            return len(rows)
        except Exception:
            self.cache.mark_terminal(request.request_id, request.memory_type, "failed")
            raise

    def recall_cached(self, request: MemoryRecallRequest) -> MemoryHintPack:
        """Trim request-prefetched candidates for one consumer, with node fallback."""

        if not self._scope_memory_enabled(request.scope):
            return MemoryHintPack(
                request_id=request.request_id,
                consumer=request.consumer,
                memory_type=request.memory_type,
            )

        status = self.cache.status(request.request_id, request.memory_type)
        if status != "not_requested":
            query_profile = self.query_builder.build(request)
            rows = [
                row for row in self.cache.get(request.request_id, request.memory_type)
                if row.status == MemoryStatus.ACTIVE
                and row.memory_type == request.memory_type
                and row.memory_level != MemoryLevel.L0
                and self._scope_allowed(row, request)
                and (request.consumer in row.allowed_consumers or "*" in row.allowed_consumers)
                and (not row.expires_at or row.expires_at > _now())
                and matches_exact_constraints(row, query_profile)
            ]
            query_tokens = _tokens(query_profile.query_text)
            documents = self._projection_documents(rows)
            bm25_scores = self._bm25_scores(rows, query_tokens, documents)
            ranked = sorted(
                rows,
                key=lambda row: self._governed_score(
                    row,
                    fusion_score=(
                        bm25_scores.get(row.memory_id, 0.0) * 0.65
                        + self._token_overlap(documents.get(row.memory_id, row.summary), query_tokens) * 0.35
                    ),
                    scope_bonus=self._scope_bonus(row, request),
                ),
                reverse=True,
            )
            return self._build_hint_pack(
                request,
                self._order_by_level(request, ranked)[: request.max_items],
            )
        return self.recall(request)

    def _ranked_candidates(
        self,
        request: MemoryRecallRequest,
        *,
        allowed_consumers: set[str] | None = None,
    ) -> list[MemoryRecord]:
        if not self._scope_memory_enabled(request.scope):
            return []
        retrieval_policy = self.retrieval_policies.get(request.memory_type)
        levels = (
            (request.memory_level_override,)
            if request.memory_level_override
            else retrieval_policy.levels
        )
        level_set = set(levels)
        consumers = allowed_consumers or {request.consumer}
        query_profile = self.query_builder.build(request)
        rows: list[MemoryRecord] = []
        for index, scope in enumerate(self._scope_chain(request)):
            rows += self.repository.list(
                status=MemoryStatus.ACTIVE.value,
                scope_type=scope.scope_type,
                scope_id=scope.scope_id,
                memory_type=request.memory_type.value,
                limit=(
                    max(32, request.max_items * 16)
                    if index == 0
                    else max(16, request.max_items * 8)
                ),
            )
        candidates: list[MemoryRecord] = []

        def eligible(row: MemoryRecord) -> bool:
            if row.memory_level not in level_set or row.memory_level == MemoryLevel.L0:
                return False
            if row.status != MemoryStatus.ACTIVE or row.memory_type != request.memory_type:
                return False
            if not consumers.intersection(row.allowed_consumers) and "*" not in row.allowed_consumers:
                return False
            if row.expires_at and row.expires_at <= _now():
                return False
            return (
                self._scope_allowed(row, request)
                and matches_exact_constraints(row, query_profile)
            )

        for row in rows:
            if not eligible(row):
                continue
            row.freshness_score = self._freshness(row)
            candidates.append(row)

        if retrieval_policy.channels == ("sql",):
            if not candidates:
                return []
            ranked = sorted(
                candidates,
                key=lambda row: self._governed_score(
                    row,
                    fusion_score=1.0,
                    scope_bonus=self._scope_bonus(row, request),
                ),
                reverse=True,
            )
            return self._order_by_level(request, ranked)

        graph_ids: list[str] = []
        memo_hits: list[dict[str, Any]] = []

        def graph_search() -> list[str]:
            if "graph" not in retrieval_policy.channels or not query_profile.graph_anchor_keys:
                return []
            try:
                return self.graph.expand_candidates(
                    request,
                    [],
                    query_profile=query_profile,
                )[: retrieval_policy.max_graph_candidates]
            except Exception:
                return []

        def vector_search() -> list[dict[str, Any]]:
            if "vector" not in retrieval_policy.channels or not query_profile.query_text:
                return []
            try:
                return self.memo.search(
                    request,
                    query_text=query_profile.query_text,
                    metadata_filter={
                        "memory_type": request.memory_type.value,
                        "status": "active",
                        "projection_version": "v2",
                    },
                )[: retrieval_policy.max_vector_candidates]
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-hybrid") as executor:
            graph_future = executor.submit(graph_search)
            vector_future = executor.submit(vector_search)
            graph_ids = graph_future.result()
            memo_hits = vector_future.result()

        memo_scores = {str(hit.get("case_memory_id") or ""): float(hit.get("score") or 0.0) for hit in memo_hits}
        discovered_ids = list(
            dict.fromkeys(
                [
                    *graph_ids[: max(32, request.max_items * 8)],
                    *list(memo_scores)[: max(32, request.max_items * 8)],
                ]
            )
        )
        existing_ids = {row.memory_id for row in candidates}
        for row in self.repository.get_many(discovered_ids):
            if row.memory_id in existing_ids or not eligible(row):
                continue
            row.freshness_score = self._freshness(row)
            candidates.append(row)
            existing_ids.add(row.memory_id)
        if not candidates:
            return []
        query_tokens = _tokens(query_profile.query_text)
        documents = self._projection_documents(candidates)
        bm25_scores = self._bm25_scores(candidates, query_tokens, documents)
        overlap_scores = {
            row.memory_id: self._token_overlap(
                documents.get(row.memory_id, row.summary),
                query_tokens,
            )
            for row in candidates
        }
        rankings: list[tuple[float, list[str]]] = []
        vector_ranking = [
            str(hit.get("case_memory_id") or "")
            for hit in memo_hits
            if str(hit.get("case_memory_id") or "") and float(hit.get("score") or 0.0) > 0.0
        ]
        if vector_ranking:
            rankings.append((1.0, vector_ranking))
        if graph_ids:
            rankings.append((0.9, list(dict.fromkeys(graph_ids))))
        bm25_ranking = [
            memory_id
            for memory_id, score in sorted(
                bm25_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0.0
        ]
        if bm25_ranking:
            rankings.append((0.65, bm25_ranking))
        overlap_ranking = [
            memory_id
            for memory_id, score in sorted(
                overlap_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0.0
        ]
        if overlap_ranking:
            rankings.append((0.35, overlap_ranking))
        fusion_scores = self._rrf_scores(rankings)
        maximum_fusion = max(fusion_scores.values(), default=0.0)
        if maximum_fusion > 0.0:
            fusion_scores = {
                memory_id: score / maximum_fusion
                for memory_id, score in fusion_scores.items()
            }
        if retrieval_policy.min_signal_required and (query_profile.query_text or query_profile.graph_anchor_keys):
            candidates = [row for row in candidates if row.memory_id in fusion_scores]
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda row: self._governed_score(
                row,
                fusion_score=fusion_scores.get(row.memory_id, 0.0),
                scope_bonus=self._scope_bonus(row, request),
            ),
            reverse=True,
        )
        return self._order_by_level(request, ranked)

    def batch_recall(self, *, request_id: str, consumer: str, scope: MemoryScope, task_context: dict[str, Any]) -> dict[MemoryType, MemoryHintPack]:
        result: dict[MemoryType, MemoryHintPack] = {}
        all_rows: list[MemoryRecord] = []
        for memory_type in (MemoryType.INTENT_ROUTE, MemoryType.POLICY_SEARCH, MemoryType.ANSWER_STYLE, MemoryType.DECISION_PLAN):
            request = MemoryRecallRequest(request_id=request_id, consumer=consumer, memory_type=memory_type, scope=scope, task_context=task_context)
            pack = self.recall(request)
            result[memory_type] = pack
            all_rows.extend(self.repository.get(item_id) for item_id in pack.memory_ids if self.repository.get(item_id) is not None)
        self.cache.put(request_id, all_rows)
        return result

    def _build_hint_pack(self, request: MemoryRecallRequest, rows: list[MemoryRecord]) -> MemoryHintPack:
        pack = MemoryHintPack(request_id=request.request_id, consumer=request.consumer, memory_type=request.memory_type)
        for row in rows:
            policy = self.views.get_policy(request.consumer, row.memory_type, row.memory_level)
            control_trimmed = policy.trim(row.payload, policy.control_fields)
            prompt_trimmed = policy.trim(row.payload, policy.prompt_fields)
            tool_trimmed = policy.trim(row.payload, policy.tool_param_fields)
            trace_trimmed = policy.trim(row.payload, policy.trace_fields)
            if control_trimmed:
                pack.control_hints.append({"memory_id": row.memory_id, "memory_level": row.memory_level.value, **control_trimmed})
            if prompt_trimmed:
                pack.prompt_contexts.append({"memory_id": row.memory_id, "memory_level": row.memory_level.value, **prompt_trimmed})
            if tool_trimmed:
                pack.tool_param_hints.append({"memory_id": row.memory_id, "memory_level": row.memory_level.value, **tool_trimmed})
            if trace_trimmed:
                pack.trace_refs.append({"memory_id": row.memory_id, "memory_level": row.memory_level.value, **trace_trimmed})
            if control_trimmed or prompt_trimmed or tool_trimmed or trace_trimmed:
                pack.memory_ids.append(row.memory_id)
                pack.retrieved_levels.append(row.memory_level)
        return pack

    def _freshness(self, memory: MemoryRecord) -> float:
        now = _now()
        if memory.expires_at and memory.expires_at <= now:
            return 0.0
        anchor = memory.last_verified_at or memory.created_at
        age_days = max(0.0, (now - anchor).total_seconds() / 86400)
        value = math.exp(-age_days / 180.0)
        if memory.review_after and memory.review_after <= now:
            value *= 0.7
        return round(max(0.0, min(1.0, value)), 4)

    def _governed_score(
        self,
        memory: MemoryRecord,
        *,
        fusion_score: float,
        scope_bonus: float = 0.0,
    ) -> float:
        success_ratio = memory.success_count / max(1, memory.usage_count)
        level_bonus = {
            MemoryLevel.L1: 0.02,
            MemoryLevel.L2: 0.04,
            MemoryLevel.L3: 0.03,
        }.get(memory.memory_level, 0.0)
        return (
            min(1.0, max(0.0, fusion_score)) * 0.7
            + memory.freshness_score * 0.12
            + memory.importance_score * 0.08
            + min(1.0, success_ratio) * 0.05
            + level_bonus
            + scope_bonus
        )

    def _projection_documents(
        self,
        memories: list[MemoryRecord],
    ) -> dict[str, str]:
        documents: dict[str, str] = {}
        for memory in memories:
            projection = self.vector_builder.build(memory)
            documents[memory.memory_id] = (
                projection.embedding_text if projection is not None else memory.summary
            )
        return documents

    def _order_by_level(
        self,
        request: MemoryRecallRequest,
        ranked: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        if request.memory_level_override:
            return ranked
        levels = self.retrieval_policies.get(request.memory_type).levels
        by_level = {
            level: [row for row in ranked if row.memory_level == level]
            for level in levels
        }
        return [row for level in levels for row in by_level[level]]

    @staticmethod
    def _token_overlap(document: str, query_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        return len(_tokens(document) & query_tokens) / len(query_tokens)

    @staticmethod
    def _rrf_scores(
        rankings: list[tuple[float, list[str]]],
        *,
        rank_constant: int = 60,
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for weight, ranking in rankings:
            for rank, memory_id in enumerate(dict.fromkeys(ranking), start=1):
                if memory_id:
                    scores[memory_id] += weight / (rank_constant + rank)
        return dict(scores)

    @staticmethod
    def _scope_chain(request: MemoryRecallRequest) -> list[MemoryScope]:
        chain = [request.scope, *request.scope_fallbacks]
        if not any(scope.scope_type == "global" and scope.scope_id == "global" for scope in chain):
            chain.append(MemoryScope(scope_type="global", scope_id="global"))
        deduped: list[MemoryScope] = []
        seen: set[tuple[str, str]] = set()
        for scope in chain:
            key = (scope.scope_type, scope.scope_id)
            if key not in seen:
                seen.add(key)
                deduped.append(scope)
        return deduped

    def _scope_allowed(self, memory: MemoryRecord, request: MemoryRecallRequest) -> bool:
        return any(memory.scope == scope for scope in self._scope_chain(request))

    def _scope_bonus(self, memory: MemoryRecord, request: MemoryRecallRequest) -> float:
        for index, scope in enumerate(self._scope_chain(request)):
            if memory.scope == scope:
                return max(0.0, 0.03 - index * 0.01)
        return 0.0

    @staticmethod
    def _bm25_scores(
        memories: list[MemoryRecord],
        query_tokens: set[str],
        documents_by_id: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Compute a small in-process Okapi BM25 pass over hard-filtered rows."""

        if not memories or not query_tokens:
            return {}
        documents = {
            memory.memory_id: list(
                _tokens(
                    (documents_by_id or {}).get(memory.memory_id, _payload_text(memory))
                )
            )
            for memory in memories
        }
        document_count = len(documents)
        average_length = sum(len(tokens) for tokens in documents.values()) / document_count
        document_frequency: dict[str, int] = defaultdict(int)
        for tokens in documents.values():
            for token in set(tokens):
                document_frequency[token] += 1
        k1 = 1.2
        b = 0.75
        scores: dict[str, float] = {}
        for memory_id, tokens in documents.items():
            term_frequency: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_frequency[token] += 1
            score = 0.0
            for token in query_tokens:
                frequency = term_frequency.get(token, 0)
                if frequency == 0:
                    continue
                df = document_frequency.get(token, 0)
                inverse_document_frequency = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * len(tokens) / max(1.0, average_length))
                score += inverse_document_frequency * (frequency * (k1 + 1.0)) / denominator
            scores[memory_id] = score
        maximum = max(scores.values(), default=0.0)
        if maximum <= 0.0:
            return scores
        return {memory_id: score / maximum for memory_id, score in scores.items()}

    def govern(
        self,
        memory_id: str,
        action: str,
        actor_id: str,
        until: datetime | None = None,
        related_memory_id: str | None = None,
    ) -> MemoryRecord:
        memory = self.repository.get(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        self._assert_actor_scope(memory, actor_id)
        if action == "supersede":
            return self._supersede(memory, related_memory_id, actor_id)
        if action == "mark_conflict":
            return self._mark_conflict(memory, related_memory_id, actor_id)
        transitions = {
            "confirm": MemoryStatus.ACTIVE,
            "auto_activate": MemoryStatus.ACTIVE,
            "restore": MemoryStatus.ACTIVE,
            "reject": MemoryStatus.REJECTED,
            "snooze": MemoryStatus.SNOOZED,
            "archive": MemoryStatus.ARCHIVED,
            "revoke": MemoryStatus.REVOKED,
            "tombstone": MemoryStatus.TOMBSTONED,
        }
        if action not in transitions:
            raise ValueError(f"unsupported memory action: {action}")
        if action == "restore" and memory.status != MemoryStatus.ARCHIVED:
            raise ValueError("only archived memories can be restored")
        if action == "tombstone" and memory.status not in {
            MemoryStatus.REJECTED,
            MemoryStatus.ARCHIVED,
            MemoryStatus.REVOKED,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.CONFLICT_REVIEW,
        }:
            raise ValueError("active or pending memories must be revoked or archived before tombstone")
        if action in {"confirm", "auto_activate", "restore"} and memory.memory_level == MemoryLevel.L0:
            raise ValueError("L0 source events cannot be activated")
        memory.status = transitions[action]
        memory.snoozed_until = until if action == "snooze" else None
        memory.last_verified_at = _now() if action in {"confirm", "auto_activate", "restore"} else memory.last_verified_at
        memory.projection_sync_status = "pending"
        self.repository.update(memory)
        self.events.append(memory.memory_id, f"MEMORY_{action.upper()}", actor_id, {"status": memory.status.value})
        self.outbox.enqueue("MEMORY_STATUS_CHANGED", memory.memory_id, {"status": memory.status.value, "action": action})
        return memory

    def _supersede(
        self,
        old_memory: MemoryRecord,
        new_memory_id: str | None,
        actor_id: str,
    ) -> MemoryRecord:
        if not new_memory_id:
            raise ValueError("supersede requires related_memory_id")
        new_memory = self.repository.get(new_memory_id)
        if new_memory is None:
            raise KeyError(new_memory_id)
        self._assert_actor_scope(new_memory, actor_id)
        if new_memory.status != MemoryStatus.ACTIVE:
            raise ValueError("replacement memory must be active before superseding")
        if new_memory.memory_type != old_memory.memory_type:
            raise ValueError("replacement memory must have the same memory_type")
        if new_memory.scope != old_memory.scope:
            raise ValueError("replacement memory must use the same scope")
        old_memory.status = MemoryStatus.SUPERSEDED
        old_memory.projection_sync_status = "pending"
        new_memory.supersedes_id = old_memory.memory_id
        new_memory.projection_sync_status = "pending"
        self.repository.update(old_memory)
        self.repository.update(new_memory)
        self.events.append(old_memory.memory_id, "MEMORY_SUPERSEDED", actor_id, {"replacement_id": new_memory.memory_id})
        self.events.append(new_memory.memory_id, "MEMORY_SUPERSEDES", actor_id, {"replaced_id": old_memory.memory_id})
        self.outbox.enqueue("MEMORY_STATUS_CHANGED", old_memory.memory_id, {"status": old_memory.status.value, "action": "supersede"})
        self.outbox.enqueue("MEMORY_STATUS_CHANGED", new_memory.memory_id, {"status": new_memory.status.value, "action": "supersede"})
        return old_memory

    def _mark_conflict(
        self,
        memory: MemoryRecord,
        conflict_memory_id: str | None,
        actor_id: str,
    ) -> MemoryRecord:
        if not conflict_memory_id:
            raise ValueError("mark_conflict requires related_memory_id")
        conflict = self.repository.get(conflict_memory_id)
        if conflict is None:
            raise KeyError(conflict_memory_id)
        self._assert_actor_scope(conflict, actor_id)
        if conflict.memory_id == memory.memory_id:
            raise ValueError("memory cannot conflict with itself")
        if conflict.memory_type != memory.memory_type:
            raise ValueError("conflict review requires the same memory_type")
        for current, other in ((memory, conflict), (conflict, memory)):
            current.status = MemoryStatus.CONFLICT_REVIEW
            current.conflict_count += 1
            if other.memory_id not in current.conflict_with_ids:
                current.conflict_with_ids.append(other.memory_id)
            current.projection_sync_status = "pending"
            self.repository.update(current)
            self.events.append(current.memory_id, "MEMORY_CONFLICT_MARKED", actor_id, {"conflict_id": other.memory_id})
            self.outbox.enqueue("MEMORY_STATUS_CHANGED", current.memory_id, {"status": current.status.value, "action": "mark_conflict"})
        return memory

    def process_pending_projections(self, limit: int = 50) -> dict[str, int]:
        """Apply outbox events idempotently to Mem0 and the governed graph."""

        processed = 0
        failed = 0
        for event in self.outbox.claim_next_batch(limit):
            memory: MemoryRecord | None = None
            try:
                if str(event.get("event_type") or "") == "MEMORY_CAPTURE_REQUESTED":
                    self._process_capture_request(event)
                    self.outbox.mark_processed(str(event["event_id"]))
                    processed += 1
                    continue
                memory = self.repository.get(str(event["aggregate_id"]))
                if memory is None:
                    self.outbox.mark_processed(str(event["event_id"]))
                    processed += 1
                    continue
                if memory.status == MemoryStatus.ACTIVE:
                    vector_projection = self.vector_builder.build(memory)
                    if vector_projection is None:
                        if memory.memo_memory_id:
                            self.memo.delete_projection(memory)
                            memory.memo_memory_id = None
                            self.repository.update(memory)
                    elif memory.memo_memory_id:
                        self.memo.update_projection(memory, vector_projection)
                    else:
                        memo_id = self.memo.add_active_projection(memory, vector_projection)
                        if memo_id:
                            memory.memo_memory_id = memo_id
                            self.repository.update(memory)
                        elif bool(getattr(self.memo, "enabled", False)):
                            raise MemoProjectionError("Mem0 did not return a projection id")
                    self.graph.upsert(memory)
                else:
                    self.memo.delete_projection(memory)
                    self.graph.delete_projection(memory.memory_id)
                    memory.memo_memory_id = None
                memory.projection_sync_status = "synced"
                self.repository.update(memory)
                self.outbox.mark_processed(str(event["event_id"]))
                processed += 1
            except Exception as exc:  # pragma: no cover - adapter failures are environment-dependent
                failed += 1
                if memory is not None:
                    memory.projection_sync_status = "out_of_sync"
                    self.repository.update(memory)
                self.outbox.mark_failed(str(event["event_id"]), exc.__class__.__name__)
        return {"processed": processed, "failed": failed}

    def _process_capture_request(self, event: dict[str, Any]) -> None:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("capture request payload must be an object")
        for draft in payload.get("drafts", [])[:16]:
            if not isinstance(draft, dict):
                continue
            scope = MemoryScope.model_validate(draft.get("scope") or {})
            if not self._scope_memory_enabled(scope):
                continue
            self.capture(
                memory_id=str(draft.get("memory_id") or ""),
                memory_type=MemoryType(str(draft.get("memory_type") or "")),
                level=MemoryLevel(str(draft.get("memory_level") or "")),
                scope=scope,
                payload=dict(draft.get("payload") or {}),
                source_event_ids=[str(item) for item in draft.get("source_event_ids", [])],
                allowed_consumers=[str(item) for item in draft.get("allowed_consumers", [])],
                confidence=float(draft.get("confidence") or 0.0),
                feature=str(draft.get("feature") or "runtime_capture"),
            )

    def reconcile(self, limit: int = 100) -> dict[str, int]:
        """Requeue stale projections and run one bounded compensation pass."""

        requeued = 0
        cleanup_statuses = {
            MemoryStatus.ARCHIVED,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.REVOKED,
            MemoryStatus.TOMBSTONED,
            MemoryStatus.REJECTED,
        }
        for memory in self.repository.list():
            needs_active_retry = memory.status == MemoryStatus.ACTIVE and memory.projection_sync_status != "synced"
            needs_cleanup = memory.status in cleanup_statuses and (
                memory.memo_memory_id is not None
                or memory.projection_sync_status != "synced"
            )
            if needs_active_retry or needs_cleanup:
                self.outbox.enqueue("MEMORY_RECONCILIATION_REQUESTED", memory.memory_id, {"status": memory.status.value})
                requeued += 1
                if requeued >= limit:
                    break
        result = self.process_pending_projections(limit=limit)
        return {"requeued": requeued, **result}

    def promote(self, *, level: MemoryLevel, scope: MemoryScope, memory_type: MemoryType) -> MemoryRecord | None:
        if level == MemoryLevel.L2:
            created = self.consolidation_worker.promote_l1_to_l2_candidates(
                scope=scope,
                memory_type=memory_type,
                limit=1,
            )
        elif level == MemoryLevel.L3:
            created = self.consolidation_worker.promote_l2_to_l3_candidates(
                scope=scope,
                memory_type=memory_type,
                limit=1,
            )
        else:
            created = []
        return created[0] if created else None

    def pending_count(self, scope: MemoryScope) -> int:
        rows = self.repository.list(
            status=MemoryStatus.CANDIDATE.value,
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
        )
        return sum(1 for row in rows if row.memory_level != MemoryLevel.L0)

    def presentation_items(self, *, scope: MemoryScope, status: str | None = None, memory_type: str | None = None) -> list[MemoryPresentationItem]:
        rows = self.repository.list(status=status, scope_type=scope.scope_type, scope_id=scope.scope_id, memory_type=memory_type)
        return [self._safe_item(row) for row in rows if row.memory_level != MemoryLevel.L0]

    def safe_detail(self, memory_id: str, actor_id: str) -> MemorySafeDetail:
        memory = self.repository.get(memory_id)
        if memory is None or memory.memory_level == MemoryLevel.L0:
            raise KeyError(memory_id)
        self._assert_actor_scope(memory, actor_id)
        events = self.events.list_for_memory(memory_id)
        detail = self._safe_item(memory, events)
        action = memory.payload.get("structured_content") or memory.payload.get("recommended_action") or memory.payload.get("standard_steps") or memory.payload.get("stable_preferences") or {}
        source_refs = memory.payload.get("source_refs") or memory.payload.get("supporting_l1_refs") or memory.payload.get("supporting_l2_refs") or []
        return MemorySafeDetail(**detail.model_dump(), structured_action=action if isinstance(action, dict) else {"steps": action}, detail=str(memory.payload.get("detail") or memory.payload.get("scenario_summary") or memory.payload.get("profile_or_playbook_summary") or ""), source_refs=[str(item) for item in source_refs], status_events=events)

    def preview(
        self,
        memory_id: str,
        consumer: str,
        task_context: dict[str, Any],
        actor_id: str | None = None,
    ) -> MemoryPreview:
        memory = self.repository.get(memory_id)
        if memory is None or memory.memory_level == MemoryLevel.L0:
            return MemoryPreview(memory_id=memory_id, consumer=consumer, available=False, reason="L0 or missing memory is not runtime-recallable")
        if actor_id is not None:
            self._assert_actor_scope(memory, actor_id)
        if not self._scope_memory_enabled(memory.scope):
            return MemoryPreview(
                memory_id=memory_id,
                consumer=consumer,
                available=False,
                reason="personal_memory_disabled",
            )
        request = MemoryRecallRequest(request_id=f"preview_{uuid4().hex}", consumer=consumer, memory_type=memory.memory_type, scope=memory.scope, task_context=task_context)
        policy = self.views.get_policy(consumer, memory.memory_type, memory.memory_level)
        consumer_allowed = consumer in memory.allowed_consumers or "*" in memory.allowed_consumers
        not_expired = not memory.expires_at or memory.expires_at > _now()
        available = consumer_allowed and not_expired and bool(policy.allowed_fields)
        pack = self._build_hint_pack(request, [memory]) if available else None
        return MemoryPreview(memory_id=memory_id, consumer=consumer, node=str(task_context.get("node") or ""), available=available, memory_level=memory.memory_level if available else None, allowed_fields=sorted(policy.allowed_fields), delivery_locations=[name for name, fields in (("control_hints", policy.control_fields), ("prompt_contexts", policy.prompt_fields), ("tool_param_hints", policy.tool_param_fields), ("trace_refs", policy.trace_fields)) if fields], hint_pack=pack, reason=None if available else "consumer, freshness or field policy prevented preview")

    def release_expired_snoozes(self, limit: int = 100) -> int:
        released = 0
        now = _now()
        for memory in self.repository.list(status=MemoryStatus.SNOOZED.value):
            if memory.snoozed_until is None or memory.snoozed_until > now:
                continue
            memory.status = MemoryStatus.CANDIDATE
            memory.snoozed_until = None
            self.repository.update(memory)
            self.events.append(memory.memory_id, "MEMORY_SNOOZE_EXPIRED", None, {"status": "candidate"})
            released += 1
            if released >= limit:
                break
        return released

    def run_consolidation_once(self, limit_per_scope: int = 25) -> dict[str, int]:
        totals = {"l0_to_l1": 0, "l1_to_l2": 0, "l2_to_l3": 0, "archived": 0}
        scopes = {
            (memory.scope.scope_type, memory.scope.scope_id)
            for memory in self.repository.list(limit=2000)
            if memory.scope.scope_type == "auditor"
        }
        for scope_type, scope_id in scopes:
            scope = MemoryScope(scope_type=scope_type, scope_id=scope_id)
            if not self._scope_memory_enabled(scope):
                continue
            result = self.consolidation_worker.run_once(
                scope,
                limit=limit_per_scope,
            )
            for key in totals:
                totals[key] += min(limit_per_scope, int(result.get(key, 0)))
        return totals

    def start_maintenance(
        self,
        *,
        poll_seconds: float = 2.0,
        reconciliation_seconds: float = 86400.0,
        consolidation_seconds: float = 3600.0,
    ) -> None:
        with self._maintenance_lock:
            if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
                return
            self._maintenance_stop.clear()
            self._maintenance_thread = Thread(
                target=self._maintenance_loop,
                kwargs={
                    "poll_seconds": max(0.25, poll_seconds),
                    "reconciliation_seconds": max(60.0, reconciliation_seconds),
                    "consolidation_seconds": max(60.0, consolidation_seconds),
                },
                name="case-memory-maintenance",
                daemon=True,
            )
            self._maintenance_thread.start()

    def shutdown(self) -> None:
        self._maintenance_stop.set()
        thread = self._maintenance_thread
        if thread is not None:
            thread.join(timeout=2.0)

    def runtime_status(self) -> dict[str, Any]:
        thread = self._maintenance_thread
        return {
            "available": True,
            "worker_running": bool(thread and thread.is_alive()),
            "outbox": self.outbox.status(),
            "mem0_enabled": bool(getattr(self.memo, "enabled", True)),
            "mem0_last_error": getattr(self.memo, "last_error", None),
            "last_error": self._maintenance_last_error,
            "last_reconciliation_at": self._last_reconciliation_at,
            "last_consolidation_at": self._last_consolidation_at,
        }

    def _maintenance_loop(
        self,
        *,
        poll_seconds: float,
        reconciliation_seconds: float,
        consolidation_seconds: float,
    ) -> None:
        next_reconciliation = _now() + timedelta(seconds=reconciliation_seconds)
        next_consolidation = _now() + timedelta(seconds=consolidation_seconds)
        while not self._maintenance_stop.wait(poll_seconds):
            try:
                self.process_pending_projections(limit=50)
                self.release_expired_snoozes(limit=50)
                now = _now()
                if now >= next_consolidation:
                    self.run_consolidation_once(limit_per_scope=25)
                    self._last_consolidation_at = now
                    next_consolidation = now + timedelta(seconds=consolidation_seconds)
                if now >= next_reconciliation:
                    self.reconcile(limit=100)
                    self._last_reconciliation_at = now
                    next_reconciliation = now + timedelta(seconds=reconciliation_seconds)
                self._maintenance_last_error = None
            except Exception as exc:  # pragma: no cover - background runtime safety
                self._maintenance_last_error = exc.__class__.__name__

    def _safe_item(
        self,
        memory: MemoryRecord,
        events: list[dict[str, Any]] | None = None,
    ) -> MemoryPresentationItem:
        status_events = events if events is not None else self.events.list_for_memory(memory.memory_id)
        activation_event = next(
            (
                event
                for event in reversed(status_events)
                if str(event.get("event_type") or "") in {"MEMORY_CONFIRM", "MEMORY_AUTO_ACTIVATE"}
            ),
            None,
        )
        activation_event_type = str((activation_event or {}).get("event_type") or "")
        if activation_event is not None or memory.admission_policy == "auto_active":
            activation_mode = (
                "auto_active"
                if memory.admission_policy == "auto_active" or activation_event_type == "MEMORY_AUTO_ACTIVATE"
                else "human_confirmed"
            )
        elif memory.status == MemoryStatus.SHADOW:
            activation_mode = "shadow"
        else:
            activation_mode = "pending_review"
        activated_at = (activation_event or {}).get("created_at")
        if activation_mode == "auto_active" and activated_at is None:
            activated_at = memory.created_at if activation_mode == "auto_active" else memory.last_verified_at
        archive_event = next(
            (
                event
                for event in reversed(status_events)
                if str(event.get("event_type") or "") == "MEMORY_ARCHIVE"
            ),
            None,
        )
        supporting_refs = (
            memory.payload.get("derived_from_l0_refs")
            or memory.payload.get("supporting_l1_refs")
            or memory.payload.get("supporting_l2_refs")
            or memory.source_event_ids
            or []
        )
        return MemoryPresentationItem(
            memory_id=memory.memory_id,
            memory_type=memory.memory_type,
            memory_level=memory.memory_level,
            status=memory.status,
            summary=memory.summary,
            source=[str(item) for item in (memory.payload.get("source_refs") or memory.payload.get("supporting_l1_refs") or memory.payload.get("supporting_l2_refs") or memory.source_event_ids)],
            created_at=memory.created_at,
            last_verified_at=memory.last_verified_at,
            confidence=memory.admission_confidence,
            observation_count=max(
                1,
                memory.shadow_observation_count,
                len(supporting_refs),
            ),
            scope=memory.scope,
            allowed_consumers=list(memory.allowed_consumers),
            freshness_score=memory.freshness_score,
            importance_score=memory.importance_score,
            projection_sync_status=memory.projection_sync_status,
            creation_mode=(
                "system_consolidated"
                if memory.memory_level in {MemoryLevel.L2, MemoryLevel.L3}
                else "system_extracted"
            ),
            activation_mode=activation_mode,
            activated_at=activated_at,
            activated_by=(
                str(activation_event.get("actor_id"))
                if activation_event and activation_event.get("actor_id")
                else None
            ),
            archived_at=(archive_event or {}).get("created_at"),
            archived_by=(
                str(archive_event.get("actor_id"))
                if archive_event and archive_event.get("actor_id")
                else None
            ),
        )

    @staticmethod
    def _assert_actor_scope(memory: MemoryRecord, actor_id: str) -> None:
        """Prevent auditor-scoped memories from being read or changed cross-user."""

        if memory.scope.scope_type == "auditor" and memory.scope.scope_id not in {actor_id, "global"}:
            raise PermissionError("memory is outside the current auditor scope")


class MemoryConsolidationWorker:
    """Deterministic maturity pipeline for L0 -> L1 -> L2 -> L3.

    The worker only creates higher-level candidates. It never bypasses the
    admission matrix or human governance, and every generated payload keeps
    the IDs of the lower-level records that support it.
    """

    def __init__(self, service: CaseMemoryService) -> None:
        self.service = service

    def promote_l0_to_l1_candidates(
        self,
        scope: MemoryScope,
        *,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        created: list[MemoryRecord] = []
        rows = self.service.repository.list(
            status=MemoryStatus.CANDIDATE.value,
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
        )
        for source in rows:
            if source.memory_level != MemoryLevel.L0:
                continue
            if self._has_derived_memory(source.memory_id, MemoryLevel.L1):
                continue
            payload = source.payload
            validation_codes = [str(item) for item in payload.get("validation_codes", [])]
            node = str(payload.get("node") or "unknown_node")
            event_type = str(payload.get("event_type") or "source_event")
            code = validation_codes[0] if validation_codes else event_type
            l1_payload = L1AtomicMemoryPayload(
                summary=f"{node} 的 {code} 事件形成可复核的原子经验",
                structured_content={
                    "event_type": event_type,
                    "node": node,
                    "validation_codes": validation_codes,
                    "repair_strategy": {
                        "mode": "controlled_recovery",
                        "instruction": "保持 fail-closed，执行受控重试或转人工处理",
                    },
                },
                detail="由脱敏来源事件锚点提炼，只保留可复用的校验恢复信息。",
                source_refs=[],
                derived_from_l0_refs=[source.memory_id],
            ).model_dump(mode="json")
            created.append(
                self.service.capture(
                    memory_type=source.memory_type,
                    level=MemoryLevel.L1,
                    scope=source.scope,
                    payload=l1_payload,
                    source_event_ids=[source.memory_id],
                    allowed_consumers=list(source.allowed_consumers),
                    confidence=source.admission_confidence,
                    feature="l0_consolidation",
                )
            )
            if len(created) >= limit:
                break
        return created

    def promote_l1_to_l2_candidates(
        self,
        scope: MemoryScope,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        rows = self._source_rows(MemoryLevel.L1, scope, memory_type)
        created: list[MemoryRecord] = []
        for group in self._similar_groups(rows):
            valid, _reason = self.service.sanitizer.validate_l1_to_l2(group, scope)
            if not valid or self._has_supporting_group(group, MemoryLevel.L2, scope):
                continue
            payload = self._build_l2_payload(group)
            created.append(
                self.service.capture(
                    memory_type=group[0].memory_type,
                    level=MemoryLevel.L2,
                    scope=scope,
                    payload=payload,
                    source_event_ids=[],
                    allowed_consumers=sorted({consumer for row in group for consumer in row.allowed_consumers}),
                    confidence=min(0.95, sum(row.admission_confidence for row in group) / len(group)),
                    feature="scenario_consolidation",
                )
            )
            if len(created) >= limit:
                break
        return created

    def promote_l2_to_l3_candidates(
        self,
        scope: MemoryScope,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        rows = self._source_rows(MemoryLevel.L2, scope, memory_type)
        created: list[MemoryRecord] = []
        for group in self._similar_groups(rows):
            valid, _reason = self.service.sanitizer.validate_l2_to_l3(group, scope)
            if not valid or self._has_supporting_group(group, MemoryLevel.L3, scope):
                continue
            payload = self._build_l3_payload(group)
            created.append(
                self.service.capture(
                    memory_type=group[0].memory_type,
                    level=MemoryLevel.L3,
                    scope=scope,
                    payload=payload,
                    source_event_ids=[],
                    allowed_consumers=sorted({consumer for row in group for consumer in row.allowed_consumers}),
                    confidence=min(0.98, sum(row.admission_confidence for row in group) / len(group)),
                    feature="playbook_consolidation",
                )
            )
            if len(created) >= limit:
                break
        return created

    def demote_or_archive_stale_memories(
        self,
        scope: MemoryScope,
        *,
        limit: int = 100,
    ) -> int:
        now = _now()
        archived = 0
        for memory in self._source_rows(None, scope, None):
            if memory.status != MemoryStatus.ACTIVE or not memory.expires_at or memory.expires_at > now:
                continue
            memory.status = MemoryStatus.ARCHIVED
            memory.projection_sync_status = "pending"
            self.service.repository.update(memory)
            self.service.events.append(memory.memory_id, "MEMORY_AUTO_ARCHIVED", None, {"reason": "expired"})
            self.service.outbox.enqueue("MEMORY_STATUS_CHANGED", memory.memory_id, {"status": memory.status.value, "reason": "expired"})
            archived += 1
            if archived >= limit:
                break
        return archived

    def run_once(
        self,
        scope: MemoryScope,
        *,
        target_scope: MemoryScope | None = None,
        limit: int = 25,
    ) -> dict[str, int]:
        destination = target_scope or scope
        bounded_limit = max(1, min(100, limit))
        l1 = self.promote_l0_to_l1_candidates(scope, limit=bounded_limit)
        l2 = self.promote_l1_to_l2_candidates(destination, limit=bounded_limit)
        l3 = self.promote_l2_to_l3_candidates(destination, limit=bounded_limit)
        archived = self.demote_or_archive_stale_memories(destination, limit=bounded_limit)
        return {
            "l0_to_l1": len(l1),
            "l1_to_l2": len(l2),
            "l2_to_l3": len(l3),
            "archived": archived,
        }

    async def run_once_async(self, scope: MemoryScope, *, target_scope: MemoryScope | None = None, limit: int = 25) -> dict[str, int]:
        """Async worker entry point for a scheduler backed by sync repositories."""

        import asyncio

        return await asyncio.to_thread(self.run_once, scope, target_scope=target_scope, limit=limit)

    def _source_rows(
        self,
        level: MemoryLevel | None,
        scope: MemoryScope,
        memory_type: MemoryType | None,
    ) -> list[MemoryRecord]:
        rows = self.service.repository.list(
            status=MemoryStatus.ACTIVE.value,
            memory_type=memory_type.value if memory_type else None,
        )
        result: list[MemoryRecord] = []
        for row in rows:
            if level is not None and row.memory_level != level:
                continue
            if memory_type is not None and row.memory_type != memory_type:
                continue
            if scope.scope_type == "auditor":
                if row.scope.scope_type != scope.scope_type or row.scope.scope_id != scope.scope_id:
                    continue
            elif row.scope.scope_type not in {"auditor", scope.scope_type}:
                continue
            result.append(row)
        return result

    def _similar_groups(self, rows: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        groups: list[list[MemoryRecord]] = []
        for row in rows:
            target = next(
                (
                    group
                    for group in groups
                    if group[0].memory_type == row.memory_type
                    and self._similarity(group[0], row) >= 0.45
                ),
                None,
            )
            if target is None:
                groups.append([row])
            else:
                target.append(row)
        return groups

    def _has_derived_memory(self, source_id: str, level: MemoryLevel) -> bool:
        for row in self.service.repository.list():
            if row.memory_level != level:
                continue
            if source_id in row.payload.get("derived_from_l0_refs", []):
                return True
            if source_id in row.payload.get("supporting_l1_refs", []) or source_id in row.payload.get("supporting_l2_refs", []):
                return True
        return False

    def _has_supporting_group(
        self,
        sources: list[MemoryRecord],
        level: MemoryLevel,
        scope: MemoryScope,
    ) -> bool:
        expected = {row.memory_id for row in sources}
        for row in self.service.repository.list(
            status=MemoryStatus.CANDIDATE.value,
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
        ):
            if row.memory_level != level or row.memory_type != sources[0].memory_type:
                continue
            refs = set(row.payload.get("supporting_l1_refs", []) or row.payload.get("supporting_l2_refs", []))
            if expected.issubset(refs):
                return True
        return False

    @staticmethod
    def _similarity(left: MemoryRecord, right: MemoryRecord) -> float:
        left_profile = memory_match_profile(left)
        right_profile = memory_match_profile(right)
        left_value = left_profile or {"summary": left.summary, "memory_type": left.memory_type.value}
        right_value = right_profile or {"summary": right.summary, "memory_type": right.memory_type.value}
        left_tokens = _tokens(left_value)
        right_tokens = _tokens(right_value)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    @staticmethod
    def _unique_values(values: Iterable[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for value in values:
            key = repr(value)
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    def _merge_value(self, left: Any, right: Any) -> Any:
        if left in (None, "", []):
            return right
        if right in (None, "", []):
            return left
        if isinstance(left, dict) and isinstance(right, dict):
            merged = dict(left)
            for key, value in right.items():
                merged[key] = self._merge_value(merged.get(key), value)
            return merged
        if left == right:
            return left
        left_values = left if isinstance(left, list) else [left]
        right_values = right if isinstance(right, list) else [right]
        return self._unique_values([*left_values, *right_values])

    def _merge_mapping(self, target: dict[str, Any], incoming: dict[str, Any]) -> None:
        for key, value in incoming.items():
            target[key] = self._merge_value(target.get(key), value)

    def _build_l2_payload(self, group: list[MemoryRecord]) -> dict[str, Any]:
        first = group[0]
        conditions: list[str] = [first.memory_type.value]
        exceptions: list[str] = []
        recommended: dict[str, Any] = {}
        match_profile: dict[str, Any] = {}
        action_payload: dict[str, Any] = {}
        for row in group:
            structured = row.payload.get("structured_content")
            if isinstance(structured, dict):
                for key, value in structured.items():
                    if key not in {"match_profile", "action_payload"}:
                        recommended[key] = self._merge_value(recommended.get(key), value)
            self._merge_mapping(match_profile, memory_match_profile(row))
            row_action = memory_action_payload(row)
            if not row_action and isinstance(structured, dict):
                row_action = {
                    key: value
                    for key, value in structured.items()
                    if key not in {"match_profile", "action_payload"}
                }
            self._merge_mapping(action_payload, row_action)
            for value in row.payload.get("applicable_conditions", []) or []:
                conditions.append(str(value))
            for value in row.payload.get("exception_rules", []) or []:
                exceptions.append(str(value))
        recommended["match_profile"] = match_profile
        recommended["action_payload"] = action_payload
        for key, value in action_payload.items():
            recommended.setdefault(key, value)
        return L2ScenarioMemoryPayload(
            scenario_summary=f"场景归纳：{first.summary}",
            applicable_conditions=self._unique_values(conditions),
            recommended_action=recommended,
            exception_rules=self._unique_values(exceptions),
            supporting_l1_refs=[row.memory_id for row in group],
        ).model_dump(mode="json")

    def _build_l3_payload(self, group: list[MemoryRecord]) -> dict[str, Any]:
        first = group[0]
        stable_preferences: dict[str, Any] = {}
        match_profile: dict[str, Any] = {}
        action_payload: dict[str, Any] = {}
        standard_steps: list[str] = []
        constraints: list[str] = []
        overrides: list[str] = []
        for row in group:
            recommended = row.payload.get("recommended_action")
            if isinstance(recommended, dict):
                self._merge_mapping(match_profile, memory_match_profile(row))
                row_action = memory_action_payload(row) or {
                    key: value
                    for key, value in recommended.items()
                    if key not in {"match_profile", "action_payload"}
                }
                self._merge_mapping(action_payload, row_action)
                for key in ("planning_steps", "standard_steps", "steps"):
                    values = row_action.get(key, [])
                    standard_steps.extend(str(value) for value in (values if isinstance(values, list) else [values]))
                values = row_action.get("constraints", [])
                constraints.extend(str(value) for value in (values if isinstance(values, list) else [values]))
                values = row_action.get("override_rules", [])
                overrides.extend(str(value) for value in (values if isinstance(values, list) else [values]))
            constraints.extend(str(value) for value in row.payload.get("exception_rules", []) or [])
        stable_preferences["match_profile"] = match_profile
        stable_preferences["action_payload"] = action_payload
        stable_preferences.update(action_payload)
        return L3StableProfileOrPlaybookPayload(
            profile_or_playbook_summary=f"稳定流程：{first.summary}",
            stable_preferences=stable_preferences,
            standard_steps=self._unique_values(standard_steps),
            constraints=self._unique_values(constraints),
            supporting_l2_refs=[row.memory_id for row in group],
            override_rules=self._unique_values(overrides),
        ).model_dump(mode="json")
