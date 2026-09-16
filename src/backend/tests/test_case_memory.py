"""Focused contract tests for the governed CaseMemoryService."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.backend.application.case_memory.service import CaseMemoryService
from src.backend.application.case_memory.retrieval import (
    MemoryGraphProjectionRegistry,
)
from src.backend.application.agent.case_agent.service.orchestrator import CaseAgentService
from src.backend.domain.case_memory.entities import (
    MemoryLevel,
    MemoryRecallRequest,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from src.backend.infrastructure.persistence.memory.case_memory_repository import (
    InMemoryEntityGraphProjectionRepository,
    InMemoryMemoMemoryAdapter,
    InMemoryMemoryEventRepository,
    InMemoryMemoryOutboxRepository,
    InMemoryMemoryRepository,
)
from src.backend.infrastructure.persistence.memory.user_repository import (
    MemoryUserRepository,
)


@pytest.fixture
def service() -> CaseMemoryService:
    return CaseMemoryService(
        InMemoryMemoryRepository(),
        InMemoryMemoryEventRepository(),
        InMemoryMemoryOutboxRepository(),
        InMemoryMemoMemoryAdapter(),
        InMemoryEntityGraphProjectionRepository(),
    )


def test_personal_memory_switch_blocks_recall_and_runtime_capture_but_keeps_history() -> None:
    users = MemoryUserRepository("password-for-test")
    actor = users.get_by_username("default_auditor")
    assert actor is not None
    service = CaseMemoryService(
        InMemoryMemoryRepository(),
        InMemoryMemoryEventRepository(),
        InMemoryMemoryOutboxRepository(),
        InMemoryMemoMemoryAdapter(),
        InMemoryEntityGraphProjectionRepository(),
        users,
    )
    scope = MemoryScope(scope_type="auditor", scope_id=actor.id)
    memory = service.capture(
        memory_type=MemoryType.INTENT_ROUTE,
        level=MemoryLevel.L1,
        scope=scope,
        payload={
            "summary": "相似案件读取基础证据包",
            "structured_content": {"route_tags": ["case_task"]},
            "detail": "",
            "source_refs": ["safe-ref"],
        },
        source_event_ids=["event-before-disabled"],
        allowed_consumers=["intent_router"],
        confidence=0.9,
    )
    service.govern(memory.memory_id, "confirm", actor.id)

    service.set_personal_memory_enabled(actor.id, False)
    request = MemoryRecallRequest(
        request_id="disabled-request",
        consumer="intent_router",
        memory_type=MemoryType.INTENT_ROUTE,
        scope=scope,
        task_context={"question": "查询案件证据"},
    )

    assert service.recall(request).memory_ids == []
    assert service.prefetch_type(request) == 0
    assert service.recall_cached(request).memory_ids == []
    assert service.enqueue_capture_request(
        source_run_id="run-while-disabled",
        scope=scope,
        observations=[
            {
                "memory_type": MemoryType.INTENT_ROUTE.value,
                "event": {"event_type": "route_confirmed"},
                "memory": {
                    "summary": "不应沉淀",
                    "structured_content": {"route_tags": ["case_task"]},
                },
                "allowed_consumers": ["intent_router"],
            }
        ],
    ) is None
    assert [item.memory_id for item in service.presentation_items(scope=scope)] == [
        memory.memory_id
    ]
    assert service.preview(
        memory.memory_id,
        "intent_router",
        {},
        actor.id,
    ).reason == "personal_memory_disabled"

    service.set_personal_memory_enabled(actor.id, True)
    assert service.recall(request).memory_ids == [memory.memory_id]


def test_policy_memory_requires_human_review_and_is_not_recalled_before_confirm(service: CaseMemoryService) -> None:
    memory = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={"summary": "北京和国家层面同时核验", "structured_content": {"filters": {"jurisdiction": ["beijing", "national"]}, "information_needs": ["材料清单"]}, "detail": "", "source_refs": ["safe-ref"]},
        source_event_ids=["event-1"],
        allowed_consumers=["policy_filter_resolver", "expert_analysis"],
        confidence=0.9,
        feature="policy_search",
    )
    assert memory.status == MemoryStatus.CANDIDATE
    presentation = service.presentation_items(scope=memory.scope, status="candidate")
    assert presentation[0].observation_count == 1
    empty = service.recall(MemoryRecallRequest(request_id="r1", consumer="policy_filter_resolver", memory_type=MemoryType.POLICY_SEARCH, scope=memory.scope, task_context={"question": "北京政策"}))
    assert empty.memory_ids == []
    service.govern(memory.memory_id, "confirm", "auditor-1")
    service.process_pending_projections()
    active = service.presentation_items(scope=memory.scope, status="active")
    assert active[0].creation_mode == "system_extracted"
    assert active[0].activation_mode == "human_confirmed"
    assert active[0].activated_by == "auditor-1"
    assert active[0].activated_at is not None
    pack = service.recall(MemoryRecallRequest(request_id="r2", consumer="policy_filter_resolver", memory_type=MemoryType.POLICY_SEARCH, scope=memory.scope, task_context={"question": "北京政策 材料"}))
    assert pack.memory_ids == [memory.memory_id]
    assert pack.tool_param_hints
    assert "filters" in pack.tool_param_hints[0]["structured_content"]


def test_intent_route_memory_is_delivered_as_control_not_tool_parameters(
    service: CaseMemoryService,
) -> None:
    scope = MemoryScope(scope_type="auditor", scope_id="auditor-1")
    memory = service.capture(
        memory_type=MemoryType.INTENT_ROUTE,
        level=MemoryLevel.L1,
        scope=scope,
        payload={
            "summary": "政策问题进入专家路由",
            "structured_content": {"route_tags": ["expert_task"]},
            "detail": "",
            "source_refs": ["safe-route-ref"],
        },
        source_event_ids=["intent-event-1"],
        allowed_consumers=["intent_router"],
        confidence=0.9,
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")

    pack = service.recall(
        MemoryRecallRequest(
            request_id="intent-recall-1",
            consumer="intent_router",
            memory_type=MemoryType.INTENT_ROUTE,
            scope=scope,
            task_context={"question": "查询政策依据"},
        )
    )

    assert pack.memory_ids == [memory.memory_id]
    assert pack.control_hints[0]["structured_content"]["route_tags"] == ["expert_task"]
    assert pack.tool_param_hints == []
    preview = service.preview(memory.memory_id, "intent_router", {})
    assert "control_hints" in preview.delivery_locations
    assert "tool_param_hints" not in preview.delivery_locations


def test_l0_never_reaches_prompt(service: CaseMemoryService) -> None:
    memory = service.capture(
        memory_type=MemoryType.FAILURE,
        level=MemoryLevel.L0,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={"event_type": "validation_failed", "source_event_id": "event-2", "node": "answer_generator", "validation_codes": ["missing_citation"], "source_ref_ids": []},
        source_event_ids=["event-2"],
        allowed_consumers=["recovery_handler"],
    )
    assert service.pending_count(memory.scope) == 0
    assert service.recall(MemoryRecallRequest(request_id="r1", consumer="recovery_handler", memory_type=MemoryType.FAILURE, scope=memory.scope, task_context={"node": "answer_generator"})).memory_ids == []


def test_expired_memory_is_filtered_and_preview_reports_unavailable(service: CaseMemoryService) -> None:
    memory = service.capture(
        memory_type=MemoryType.ANSWER_STYLE,
        level=MemoryLevel.L3,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={"profile_or_playbook_summary": "结论前置", "stable_preferences": {"conclusion_first": True}, "standard_steps": [], "constraints": [], "supporting_l2_refs": [], "override_rules": []},
        source_event_ids=[],
        allowed_consumers=["answer_generator"],
        feature="style_preference",
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")
    memory.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    service.repository.update(memory)
    pack = service.recall(MemoryRecallRequest(request_id="r1", consumer="answer_generator", memory_type=MemoryType.ANSWER_STYLE, scope=memory.scope, task_context={}))
    assert pack.memory_ids == []
    preview = service.preview(memory.memory_id, "answer_generator", {})
    assert preview.available is False


def test_graph_and_outbox_are_updated_after_activation(service: CaseMemoryService) -> None:
    memory = service.capture(
        memory_type=MemoryType.FAILURE,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={"summary": "缺少引用时执行确定性补全", "structured_content": {"repair_strategy": "add_citation", "validation_code": "missing_citation", "recover": ["rerun validator"]}, "detail": "", "source_refs": ["safe-ref"]},
        source_event_ids=["event-3"],
        allowed_consumers=["recovery_handler"],
        feature="deterministic_structure_repair",
    )
    assert memory.status == MemoryStatus.ACTIVE
    presentation = service.presentation_items(scope=memory.scope, status="active")
    assert presentation[0].creation_mode == "system_extracted"
    assert presentation[0].activation_mode == "auto_active"
    assert presentation[0].activated_by is None
    assert presentation[0].activated_at == memory.created_at
    result = service.process_pending_projections()
    assert result["processed"] >= 1
    assert memory.projection_sync_status == "synced"


def test_promotion_requires_multiple_supporting_memories(service: CaseMemoryService) -> None:
    for auditor in ("a1", "a2", "a3"):
        service.capture(
            memory_type=MemoryType.DECISION_PLAN,
            level=MemoryLevel.L1,
            scope=MemoryScope(scope_type="auditor", scope_id=auditor),
            payload={"summary": "先核验适用地域再查支付范围", "structured_content": {"planning_steps": ["scope", "retrieve"]}, "detail": "", "source_refs": []},
            source_event_ids=[],
            allowed_consumers=["decision_planner"],
            confidence=0.8,
        )
    for row in service.repository.list():
        service.govern(row.memory_id, "confirm", row.scope.scope_id)
    promoted = service.promote(level=MemoryLevel.L2, scope=MemoryScope(scope_type="auditor", scope_id="a1"), memory_type=MemoryType.DECISION_PLAN)
    assert promoted is None


def test_consolidation_worker_promotes_l0_to_l1_without_recalling_l0(service: CaseMemoryService) -> None:
    source = service.capture(
        memory_type=MemoryType.FAILURE,
        level=MemoryLevel.L0,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "event_type": "reusable_node_failure",
            "source_run_id": "run-1",
            "source_event_id": "event-1",
            "node": "validate_answer",
            "validation_codes": ["missing_citation"],
            "source_ref_ids": [],
            "derived_from": [],
        },
        source_event_ids=[],
        allowed_consumers=["recovery_handler"],
        confidence=0.7,
    )

    created = service.consolidation_worker.promote_l0_to_l1_candidates(source.scope)

    assert len(created) == 1
    assert created[0].memory_level == MemoryLevel.L1
    assert created[0].status == MemoryStatus.CANDIDATE
    assert created[0].payload["derived_from_l0_refs"] == [source.memory_id]
    assert service.recall(
        MemoryRecallRequest(
            request_id="l0-r1",
            consumer="recovery_handler",
            memory_type=MemoryType.FAILURE,
            scope=source.scope,
            task_context={"node": "validate_answer"},
        )
    ).memory_ids == []


def test_consolidation_worker_builds_distinct_l2_and_l3_payloads(service: CaseMemoryService) -> None:
    for auditor in ("a1", "a2", "a3"):
        row = service.capture(
            memory_type=MemoryType.DECISION_PLAN,
            level=MemoryLevel.L1,
            scope=MemoryScope(scope_type="auditor", scope_id=auditor),
            payload={
                "summary": "先核验适用地域，再检索支付范围",
                "structured_content": {
                    "planning_steps": ["scope", "retrieve"],
                    "constraints": ["引用可核验依据"],
                },
                "detail": "脱敏规划经验",
                "source_refs": [],
            },
            source_event_ids=[],
            allowed_consumers=["decision_planner"],
            confidence=0.8,
        )
        service.govern(row.memory_id, "confirm", auditor)

    team_scope = MemoryScope(scope_type="team", scope_id="team-a")
    l2_rows = service.consolidation_worker.promote_l1_to_l2_candidates(
        team_scope,
        memory_type=MemoryType.DECISION_PLAN,
    )

    assert len(l2_rows) == 1
    assert l2_rows[0].memory_level == MemoryLevel.L2
    assert len(l2_rows[0].payload["supporting_l1_refs"]) == 3
    l2_presentation = service.presentation_items(scope=team_scope, status="candidate")
    assert l2_presentation[0].creation_mode == "system_consolidated"
    assert l2_presentation[0].activation_mode == "pending_review"
    assert l2_presentation[0].observation_count == 3
    service.govern(l2_rows[0].memory_id, "confirm", "team-lead")
    second_l2 = service.capture(
        memory_type=MemoryType.DECISION_PLAN,
        level=MemoryLevel.L2,
        scope=team_scope,
        payload={
            "scenario_summary": "场景归纳：先核验适用地域，再检索支付范围",
            "applicable_conditions": ["decision_plan_hint"],
            "recommended_action": {"planning_steps": ["scope", "retrieve"], "constraints": ["引用可核验依据"]},
            "exception_rules": [],
            "supporting_l1_refs": ["independent-l1-reference"],
        },
        source_event_ids=[],
        allowed_consumers=["decision_planner"],
        confidence=0.85,
    )
    service.govern(second_l2.memory_id, "confirm", "team-lead")

    l3_rows = service.consolidation_worker.promote_l2_to_l3_candidates(
        team_scope,
        memory_type=MemoryType.DECISION_PLAN,
    )

    assert len(l3_rows) == 1
    assert l3_rows[0].memory_level == MemoryLevel.L3
    assert set(l3_rows[0].payload["supporting_l2_refs"]) == {l2_rows[0].memory_id, second_l2.memory_id}
    assert l3_rows[0].payload["standard_steps"] == ["scope", "retrieve"]
    assert l3_rows[0].payload["constraints"] == ["引用可核验依据"]


def test_graph_expands_related_memories_and_removes_tombstoned_links(service: CaseMemoryService) -> None:
    first = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "北京政策检索需要材料清单",
            "structured_content": {
                "filters": {"jurisdiction": ["beijing"]},
                "information_needs": ["材料清单"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        confidence=0.8,
        feature="policy_search",
    )
    second = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=first.scope,
        payload={
            "summary": "北京政策检索需要材料清单和证据重点",
            "structured_content": {
                "filters": {"jurisdiction": ["beijing"]},
                "information_needs": ["材料清单", "证据重点"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        confidence=0.8,
        feature="policy_search",
    )
    service.govern(first.memory_id, "confirm", "auditor-1")
    service.govern(second.memory_id, "confirm", "auditor-1")
    service.process_pending_projections()

    request = MemoryRecallRequest(
        request_id="graph-r1",
        consumer="policy_filter_resolver",
        memory_type=MemoryType.POLICY_SEARCH,
        scope=first.scope,
        task_context={"question": "北京 材料清单"},
    )
    pack = service.recall(request)
    assert set(pack.memory_ids) == {first.memory_id, second.memory_id}

    service.govern(second.memory_id, "archive", "auditor-1")
    service.process_pending_projections()
    pack_after_archive = service.recall(
        request.model_copy(update={"request_id": "graph-r2"})
    )
    assert second.memory_id not in pack_after_archive.memory_ids


def test_archived_memory_keeps_audit_metadata_and_restore_rebuilds_projections(
    service: CaseMemoryService,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "北京政策检索需要同时核验材料清单",
            "structured_content": {
                "filters": {"jurisdiction": ["beijing"]},
                "information_needs": ["材料清单"],
            },
            "detail": "经审核员确认的检索经验",
            "source_refs": ["safe-ref"],
        },
        source_event_ids=["event-archive-1"],
        allowed_consumers=["policy_filter_resolver"],
        confidence=0.9,
        feature="policy_search",
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")
    service.process_pending_projections()

    service.govern(memory.memory_id, "archive", "auditor-1")
    service.process_pending_projections()

    archived = service.presentation_items(scope=memory.scope, status="archived")
    assert len(archived) == 1
    assert archived[0].activation_mode == "human_confirmed"
    assert archived[0].activated_by == "auditor-1"
    assert archived[0].activated_at is not None
    assert archived[0].archived_by == "auditor-1"
    assert archived[0].archived_at is not None
    assert memory.memory_id not in service.memo._items
    assert memory.memory_id not in service.graph._items

    archived_pack = service.recall(
        MemoryRecallRequest(
            request_id="archive-r1",
            consumer="policy_filter_resolver",
            memory_type=MemoryType.POLICY_SEARCH,
            scope=memory.scope,
            task_context={"question": "北京 材料清单"},
        )
    )
    assert archived_pack.memory_ids == []

    service.govern(memory.memory_id, "restore", "auditor-1")
    projection = service.process_pending_projections()
    assert projection["failed"] == 0
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.memory_id in service.memo._items
    assert memory.memory_id in service.graph._items

    restored = service.presentation_items(scope=memory.scope, status="active")
    assert restored[0].activation_mode == "human_confirmed"
    assert restored[0].activated_by == "auditor-1"
    event_types = [event["event_type"] for event in service.events.list_for_memory(memory.memory_id)]
    assert "MEMORY_ARCHIVE" in event_types
    assert "MEMORY_RESTORE" in event_types


def test_outbox_retry_uses_backoff_and_reconciliation_skips_unprojected_candidates(service: CaseMemoryService) -> None:
    candidate = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "候选政策检索经验",
            "structured_content": {"filters": {"jurisdiction": ["beijing"]}, "information_needs": ["材料清单"]},
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        feature="policy_search",
    )

    assert service.reconcile()["requeued"] == 0
    event = service.outbox.enqueue("MEMORY_STATUS_CHANGED", candidate.memory_id, {"status": "candidate"})
    claimed = service.outbox.claim_next_batch()
    assert claimed and claimed[0]["event_id"] == event["event_id"]
    service.outbox.mark_failed(event["event_id"], "temporary_projection_error")
    assert service.outbox.claim_next_batch() == []

    # A retry becomes claimable once the scheduled backoff has elapsed.
    retry_event = next(item for item in service.outbox._events if item["event_id"] == event["event_id"])
    retry_event["next_attempt_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    retried = service.outbox.claim_next_batch()
    assert retried and retried[0]["event_id"] == event["event_id"]


def test_recall_uses_bm25_signal_after_hard_filters(service: CaseMemoryService) -> None:
    memories = []
    for summary in ("北京远程医疗材料清单", "上海药品支付范围"):
        memory = service.capture(
            memory_type=MemoryType.POLICY_SEARCH,
            level=MemoryLevel.L1,
            scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
            payload={
                "summary": summary,
                "structured_content": {"filters": {"jurisdiction": [summary[:2]]}, "information_needs": ["材料清单"]},
                "detail": "",
                "source_refs": [],
            },
            source_event_ids=[],
            allowed_consumers=["policy_filter_resolver"],
            confidence=0.8,
            feature="policy_search",
        )
        service.govern(memory.memory_id, "confirm", "auditor-1")
        memories.append(memory)

    pack = service.recall(
        MemoryRecallRequest(
            request_id="bm25-r1",
            consumer="policy_filter_resolver",
            memory_type=MemoryType.POLICY_SEARCH,
            scope=memories[0].scope,
            task_context={"question": "北京 材料清单"},
        )
    )

    assert pack.memory_ids[0] == memories[0].memory_id


def test_empty_request_cache_is_terminal_and_does_not_recall_again(
    service: CaseMemoryService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = MemoryRecallRequest(
        request_id="empty-terminal-r1",
        consumer="decision_planner",
        memory_type=MemoryType.DECISION_PLAN,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        task_context={"question": "分析当前案件"},
    )
    service.cache.begin(request.request_id, request.memory_type)
    service.cache.complete(request.request_id, request.memory_type, [])

    def unexpected_recall(_request: MemoryRecallRequest):
        raise AssertionError("terminal empty cache must not trigger another recall")

    monkeypatch.setattr(service, "recall", unexpected_recall)

    pack = service.recall_cached(request)

    assert pack.memory_ids == []
    assert service.cache.status(request.request_id, request.memory_type) == "empty"
    assert service.cache.begin(request.request_id, request.memory_type) == "empty"


def test_hybrid_retrieval_can_discover_a_memory_outside_recent_sql_candidates(
    service: CaseMemoryService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "北京远程医疗材料清单",
            "structured_content": {
                "filters": {"jurisdiction": "beijing"},
                "information_needs": ["材料清单"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        feature="policy_search",
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")
    monkeypatch.setattr(service.repository, "list", lambda **_kwargs: [])
    monkeypatch.setattr(service.graph, "expand_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service.memo,
        "search",
        lambda *_args, **_kwargs: [{"case_memory_id": memory.memory_id, "score": 0.95}],
    )

    pack = service.recall(
        MemoryRecallRequest(
            request_id="empty-relational-r1",
            consumer="policy_filter_resolver",
            memory_type=MemoryType.POLICY_SEARCH,
            scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
            task_context={"question": "北京政策"},
        )
    )

    assert pack.memory_ids == [memory.memory_id]


def test_timed_out_request_cache_rejects_late_completion(service: CaseMemoryService) -> None:
    request_id = "late-memory-r1"
    memory_type = MemoryType.DECISION_PLAN
    service.cache.begin(request_id, memory_type)
    service.cache.mark_terminal(request_id, memory_type, "timed_out")

    status = service.cache.complete(request_id, memory_type, [])

    assert status == "timed_out"
    assert service.cache.status(request_id, memory_type) == "timed_out"
    assert service.cache.get(request_id, memory_type) == []


def test_capture_request_creates_idempotent_l0_l1_pair_off_path(
    service: CaseMemoryService,
) -> None:
    observation = {
        "memory_type": MemoryType.FAILURE.value,
        "event": {
            "event_type": "answer_validation_repaired",
            "source_event_id": "run-1:repair:citation_invalid",
            "node": "validate_answer",
            "validation_codes": ["citation_invalid"],
            "source_ref_ids": [],
        },
        "memory": {
            "summary": "回答引用校验失败后，受控重写通过二次校验",
            "structured_content": {
                "repair_strategy": "second_model_call",
                "validation_code": "citation_invalid",
                "validation": "passed",
            },
            "detail": "不保存失败回答或案件事实。",
            "source_refs": ["run-1"],
        },
        "allowed_consumers": ["recovery_handler"],
        "confidence": 0.95,
        "feature": "deterministic_structure_repair",
    }
    scope = MemoryScope(scope_type="auditor", scope_id="auditor-1")

    first = service.enqueue_capture_request(
        source_run_id="run-1",
        scope=scope,
        observations=[observation],
    )
    second = service.enqueue_capture_request(
        source_run_id="run-1",
        scope=scope,
        observations=[observation],
    )

    assert first is not None and second is not None
    assert first["event_id"] == second["event_id"]
    result = service.process_pending_projections(limit=10)
    assert result["failed"] == 0
    rows = service.repository.list(scope_type="auditor", scope_id="auditor-1")
    assert len(rows) == 2
    assert {row.memory_level for row in rows} == {MemoryLevel.L0, MemoryLevel.L1}
    atomic = next(row for row in rows if row.memory_level == MemoryLevel.L1)
    anchor = next(row for row in rows if row.memory_level == MemoryLevel.L0)
    assert atomic.status == MemoryStatus.ACTIVE
    assert atomic.payload["derived_from_l0_refs"] == [anchor.memory_id]


def test_candidate_preview_uses_consumer_view_without_making_candidate_recallable(
    service: CaseMemoryService,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "北京政策检索需要材料清单",
            "structured_content": {
                "filters": {"jurisdiction": "beijing"},
                "information_needs": ["材料清单"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        feature="policy_search",
    )

    preview = service.preview(
        memory.memory_id,
        "policy_filter_resolver",
        {"node": "validate_execution_plan"},
        "auditor-1",
    )
    recalled = service.recall(
        MemoryRecallRequest(
            request_id="candidate-preview-r1",
            consumer="policy_filter_resolver",
            memory_type=MemoryType.POLICY_SEARCH,
            scope=memory.scope,
            task_context={"question": "北京政策"},
        )
    )

    assert preview.available is True
    assert preview.hint_pack is not None
    assert preview.hint_pack.memory_ids == [memory.memory_id]
    assert recalled.memory_ids == []


def test_mem0_projection_without_id_stays_out_of_sync_and_retries() -> None:
    class MissingProjectionIdAdapter(InMemoryMemoMemoryAdapter):
        enabled = True

        def add_active_projection(self, memory, projection):
            _ = (memory, projection)
            return None

    service = CaseMemoryService(
        InMemoryMemoryRepository(),
        InMemoryMemoryEventRepository(),
        InMemoryMemoryOutboxRepository(),
        MissingProjectionIdAdapter(),
        InMemoryEntityGraphProjectionRepository(),
    )
    memory = service.capture(
        memory_type=MemoryType.FAILURE,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "引用校验失败后执行受控重写",
            "structured_content": {"repair_strategy": "second_model_call"},
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["recovery_handler"],
        feature="deterministic_structure_repair",
    )

    result = service.process_pending_projections()

    assert result == {"processed": 0, "failed": 1}
    assert memory.projection_sync_status == "out_of_sync"
    assert service.outbox.status()["retrying"] == 1


def test_completed_run_enqueues_only_successful_reusable_observations(
    service: CaseMemoryService,
) -> None:
    orchestrator = object.__new__(CaseAgentService)
    orchestrator._case_memory = service
    state = {
        "run": SimpleNamespace(run_id="run-complete-1", actor_id="auditor-1"),
        "actor_id": "auditor-1",
        "intent": "expert_task",
        "intent_confidence": 0.96,
        "validation_retry_count": 1,
        "validation_error": "citation_invalid: missing source",
        "planning_error": "",
        "query_semantics": {"information_needs": ["材料清单", "支付边界"]},
        "execution_plan": [
            {
                "step": 1,
                "capability": "query_case_basic_info",
                "arguments": {},
                "depends_on": [],
            },
            {
                "step": 2,
                "capability": "ask_policy_expert",
                "arguments": {
                    "filters": {"jurisdiction": "beijing", "case_id": "CASE-1"},
                    "information_needs": ["材料清单", "支付边界"],
                },
                "depends_on": [1],
            },
        ],
        "capability_results": [
            ("ask_policy_expert", SimpleNamespace(status="success")),
        ],
    }

    count = orchestrator.enqueue_completed_run_memory_capture(state)
    service.process_pending_projections(limit=20)

    assert count == 4
    l1_rows = [
        row for row in service.repository.list(scope_type="auditor", scope_id="auditor-1")
        if row.memory_level == MemoryLevel.L1
    ]
    assert {row.memory_type for row in l1_rows} == {
        MemoryType.FAILURE,
        MemoryType.INTENT_ROUTE,
        MemoryType.POLICY_SEARCH,
        MemoryType.DECISION_PLAN,
    }
    policy = next(row for row in l1_rows if row.memory_type == MemoryType.POLICY_SEARCH)
    assert policy.payload["structured_content"]["filters"] == {"jurisdiction": "beijing"}
    assert policy.payload["structured_content"]["match_profile"]["jurisdiction"] == "beijing"
    assert policy.payload["structured_content"]["match_profile"]["information_needs"] == [
        "材料清单",
        "支付边界",
    ]
    assert policy.payload["structured_content"]["action_payload"]["filters"] == {
        "jurisdiction": "beijing"
    }
    assert "case_id" not in str(policy.payload)


def test_repeated_capture_keeps_distinct_l0_anchors_and_one_frequency_weighted_l1(
    service: CaseMemoryService,
) -> None:
    scope = MemoryScope(scope_type="auditor", scope_id="auditor-1")

    def observation(run_id: str) -> dict:
        return {
            "memory_type": MemoryType.POLICY_SEARCH.value,
            "event": {
                "event_type": "validated_policy_search_completed",
                "source_event_id": f"{run_id}:policy_search",
                "node": "call_expert_analysis",
                "validation_codes": ["policy_result_available"],
                "source_ref_ids": [],
            },
            "memory": {
                "summary": "北京政策检索覆盖材料清单",
                "structured_content": {
                    "filters": {"jurisdiction": "beijing"},
                    "information_needs": ["材料清单"],
                },
                "detail": "治理后的检索方案",
                "source_refs": [run_id],
            },
            "allowed_consumers": ["policy_filter_resolver", "expert_analysis"],
            "confidence": 0.9,
            "feature": "policy_search",
        }

    for run_id in ("run-frequency-1", "run-frequency-2"):
        service.enqueue_capture_request(
            source_run_id=run_id,
            scope=scope,
            observations=[observation(run_id)],
        )
        service.process_pending_projections(limit=10)

    rows = service.repository.list(scope_type="auditor", scope_id="auditor-1")
    anchors = [row for row in rows if row.memory_level == MemoryLevel.L0]
    atomics = [row for row in rows if row.memory_level == MemoryLevel.L1]
    assert len(anchors) == 2
    assert len(atomics) == 1
    assert atomics[0].shadow_observation_count == 2
    assert set(atomics[0].payload["derived_from_l0_refs"]) == {
        row.memory_id for row in anchors
    }


def test_supersede_conflict_tombstone_and_snooze_expiry_are_governed(
    service: CaseMemoryService,
) -> None:
    scope = MemoryScope(scope_type="auditor", scope_id="auditor-1")

    def active(summary: str):
        memory = service.capture(
            memory_type=MemoryType.DECISION_PLAN,
            level=MemoryLevel.L1,
            scope=scope,
            payload={
                "summary": summary,
                "structured_content": {"planning_steps": [summary]},
                "detail": "",
                "source_refs": [],
            },
            source_event_ids=[],
            allowed_consumers=["decision_planner"],
            confidence=0.8,
        )
        service.govern(memory.memory_id, "confirm", "auditor-1")
        return memory

    old = active("先读取事实")
    replacement = active("先读取事实再核验证据")
    superseded = service.govern(
        old.memory_id,
        "supersede",
        "auditor-1",
        related_memory_id=replacement.memory_id,
    )
    assert superseded.status == MemoryStatus.SUPERSEDED
    assert replacement.supersedes_id == old.memory_id

    first = active("先执行规则核验")
    second = active("先执行政策核验")
    conflicted = service.govern(
        first.memory_id,
        "mark_conflict",
        "auditor-1",
        related_memory_id=second.memory_id,
    )
    assert conflicted.status == MemoryStatus.CONFLICT_REVIEW
    assert second.status == MemoryStatus.CONFLICT_REVIEW
    assert second.memory_id in first.conflict_with_ids

    service.govern(old.memory_id, "tombstone", "auditor-1")
    assert old.status == MemoryStatus.TOMBSTONED

    snoozed = service.capture(
        memory_type=MemoryType.INTENT_ROUTE,
        level=MemoryLevel.L1,
        scope=scope,
        payload={
            "summary": "政策问题进入专家路由",
            "structured_content": {"route_tags": ["expert_task"]},
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["intent_router"],
    )
    service.govern(
        snoozed.memory_id,
        "snooze",
        "auditor-1",
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert service.release_expired_snoozes() == 1
    assert snoozed.status == MemoryStatus.CANDIDATE


def test_level_recall_policy_preserves_maturity_order(service: CaseMemoryService) -> None:
    scope = MemoryScope(scope_type="auditor", scope_id="auditor-1")
    l1 = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=scope,
        payload={
            "summary": "北京材料清单",
            "structured_content": {
                "filters": {"jurisdiction": "beijing"},
                "information_needs": ["材料清单"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        feature="policy_search",
    )
    l2 = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L2,
        scope=scope,
        payload={
            "scenario_summary": "政策场景经验",
            "applicable_conditions": ["北京材料清单查询"],
            "recommended_action": {
                "filters": {"jurisdiction": "beijing"},
                "information_needs": ["材料清单"],
            },
            "exception_rules": [],
            "supporting_l1_refs": ["support-1"],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        feature="policy_search",
    )
    service.govern(l1.memory_id, "confirm", "auditor-1")
    service.govern(l2.memory_id, "confirm", "auditor-1")

    pack = service.recall(
        MemoryRecallRequest(
            request_id="level-order-r1",
            consumer="policy_filter_resolver",
            memory_type=MemoryType.POLICY_SEARCH,
            scope=scope,
            task_context={"question": "北京材料清单"},
            max_items=2,
        )
    )

    assert pack.memory_ids == [l2.memory_id, l1.memory_id]


def test_typed_vector_projection_matches_transient_task_profile(
    service: CaseMemoryService,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.INTENT_ROUTE,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "政策依据查询路由到专家能力",
            "structured_content": {
                "match_profile": {
                    "task_type": "expert_task",
                    "task_pattern": "intent=expert_task|action=query_policy_basis|layer=L3|evidence=policy_evidence",
                    "action": "query_policy_basis",
                    "target_objects": ["medical_policy"],
                    "evidence_need": "policy_evidence",
                },
                "action_payload": {
                    "intent": "expert_task",
                    "route_tags": ["expert_task"],
                    "capabilities": ["ask_policy_expert"],
                },
                "route_tags": ["expert_task"],
                "capabilities": ["ask_policy_expert"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["intent_router"],
        confidence=0.96,
        feature="intent_route",
    )
    projection = service.vector_builder.build(memory)
    request = MemoryRecallRequest(
        request_id="typed-query-r1",
        consumer="intent_router",
        memory_type=MemoryType.INTENT_ROUTE,
        scope=memory.scope,
        task_context={
            "question": "请查询这项医疗服务的政策依据",
            "task_type": "expert_task",
            "task_pattern": "intent=expert_task|action=query_policy_basis|layer=L3|evidence=policy_evidence",
            "action": "query_policy_basis",
            "target_objects": ["medical_policy"],
            "evidence_need": "policy_evidence",
        },
    )
    query = service.query_builder.build(request)

    assert projection is not None
    for value in ("expert_task", "query_policy_basis", "medical_policy", "policy_evidence"):
        assert value in projection.embedding_text
        assert value in query.query_text
    assert "capabilities" not in projection.embedding_text


def test_policy_recall_rejects_explicit_jurisdiction_conflict(
    service: CaseMemoryService,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.POLICY_SEARCH,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "北京医保材料政策检索",
            "structured_content": {
                "match_profile": {
                    "jurisdiction": "beijing",
                    "policy_domain": "medical_insurance",
                    "information_needs": ["材料清单"],
                },
                "filters": {"jurisdiction": "beijing"},
                "information_needs": ["材料清单"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["policy_filter_resolver"],
        feature="policy_search",
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")

    pack = service.recall(
        MemoryRecallRequest(
            request_id="policy-conflict-r1",
            consumer="policy_filter_resolver",
            memory_type=MemoryType.POLICY_SEARCH,
            scope=memory.scope,
            task_context={
                "question": "查询上海医保材料清单",
                "jurisdiction": "shanghai",
                "policy_domain": "medical_insurance",
                "information_needs": ["材料清单"],
            },
        )
    )

    assert pack.memory_ids == []


def test_answer_style_skips_vector_projection(service: CaseMemoryService) -> None:
    memory = service.capture(
        memory_type=MemoryType.ANSWER_STYLE,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "结论前置并简洁分点",
            "structured_content": {
                "match_profile": {"scenario_type": "analysis"},
                "action_payload": {
                    "style_preferences": {"conclusion_first": True, "verbosity": "concise"}
                },
                "style_preferences": {"conclusion_first": True, "verbosity": "concise"},
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["answer_generator"],
        feature="style_preference",
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")
    service.process_pending_projections(limit=10)

    assert memory.memory_id not in service.memo._texts
    assert service.repository.get(memory.memory_id).projection_sync_status == "synced"


def test_graph_projection_is_allowlisted_and_failure_code_is_exact(
    service: CaseMemoryService,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.FAILURE,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "引用缺失时执行受控恢复",
            "structured_content": {
                "match_profile": {
                    "node": "validate_answer",
                    "validation_codes": ["citation_missing_for_grounded"],
                    "error_category": "citation_integrity",
                    "component_version": "case_agent_answer_contract_v1",
                },
                "action_payload": {"repair_strategy": "second_model_call"},
                "unknown_runtime_field": "must_not_become_graph_node",
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["recovery_handler"],
        confidence=0.95,
        feature="deterministic_structure_repair",
    )
    service.process_pending_projections(limit=10)
    anchors = service.graph._anchors[memory.memory_id]

    assert "validation_code:validation_code:citation_missing_for_grounded" in anchors
    assert all("unknown_runtime_field" not in anchor for anchor in anchors)
    assert all("memory_type" not in anchor and "consumer" not in anchor for anchor in anchors)
    relations = MemoryGraphProjectionRegistry().relations(memory)
    assert any(item.edge_type == "TRIGGERED_BY" for item in relations)


def test_maturity_consolidation_preserves_match_and_action_sections(
    service: CaseMemoryService,
) -> None:
    scope = MemoryScope(scope_type="auditor", scope_id="auditor-1")
    rows = []
    for index, jurisdiction in enumerate(("beijing", "national"), start=1):
        rows.append(
            service.capture(
                memory_type=MemoryType.POLICY_SEARCH,
                level=MemoryLevel.L1,
                scope=scope,
                payload={
                    "summary": f"政策材料检索经验 {index}",
                    "structured_content": {
                        "match_profile": {
                            "task_type": "expert_task",
                            "jurisdiction": jurisdiction,
                            "information_needs": ["材料清单"],
                        },
                        "action_payload": {
                            "filters": {"jurisdiction": jurisdiction},
                            "information_needs": ["材料清单"],
                        },
                        "filters": {"jurisdiction": jurisdiction},
                        "information_needs": ["材料清单"],
                    },
                    "detail": "",
                    "source_refs": [],
                },
                source_event_ids=[],
                allowed_consumers=["policy_filter_resolver"],
                feature="policy_search",
            )
        )
    payload = service.consolidation_worker._build_l2_payload(rows)

    assert isinstance(payload["recommended_action"]["match_profile"], dict)
    assert payload["recommended_action"]["match_profile"]["jurisdiction"] == [
        "beijing",
        "national",
    ]
    assert isinstance(payload["recommended_action"]["action_payload"], dict)
    assert payload["recommended_action"]["action_payload"]["information_needs"] == [
        "材料清单"
    ]


def test_intent_memory_shadow_records_suggestion_without_overriding_route(
    service: CaseMemoryService,
) -> None:
    memory = service.capture(
        memory_type=MemoryType.INTENT_ROUTE,
        level=MemoryLevel.L1,
        scope=MemoryScope(scope_type="auditor", scope_id="auditor-1"),
        payload={
            "summary": "政策依据查询使用专家路由",
            "structured_content": {
                "match_profile": {
                    "task_type": "expert_task",
                    "action": "query_policy_basis",
                    "target_objects": ["medical_policy"],
                    "evidence_need": "policy_evidence",
                },
                "action_payload": {
                    "intent": "expert_task",
                    "route_tags": ["expert_task"],
                    "capabilities": ["ask_policy_expert"],
                },
                "route_tags": ["expert_task"],
                "capabilities": ["ask_policy_expert"],
            },
            "detail": "",
            "source_refs": [],
        },
        source_event_ids=[],
        allowed_consumers=["intent_router"],
        confidence=0.96,
        feature="intent_route",
    )
    service.govern(memory.memory_id, "confirm", "auditor-1")
    service.process_pending_projections(limit=10)

    events: list[tuple[str, dict]] = []
    orchestrator = object.__new__(CaseAgentService)
    orchestrator._case_memory = service
    orchestrator._actor_memory_departments = {}
    orchestrator._repository = SimpleNamespace(
        append_event=lambda _run_id, event_type, _message, payload: events.append(
            (event_type, payload)
        )
    )
    state = {
        "run": SimpleNamespace(
            run_id="intent-shadow-r1",
            case_id="CASE-1",
            session_id="session-1",
        ),
        "actor_id": "auditor-1",
        "intent": "expert_task",
        "user_message": SimpleNamespace(content="请查询这项医疗服务的政策依据"),
        "perceptual_state": {
            "semantic": {
                "intent": "expert_task",
                "action": "query_policy_basis",
                "target_objects": ["medical_policy"],
                "evidence_need": "policy_evidence",
                "target_layer_hint": "L3",
                "source": "llm_semantic_parser",
            }
        },
    }
    service.prefetch_type(
        MemoryRecallRequest(
            request_id="intent-shadow-r1",
            consumer="intent_router",
            memory_type=MemoryType.INTENT_ROUTE,
            scope=memory.scope,
            task_context=orchestrator._memory_task_context(state),
        )
    )

    result = orchestrator.record_intent_memory_shadow(state)

    assert state["intent"] == "expert_task"
    assert result["mode"] == "shadow"
    assert result["agrees_with_current_route"] is True
    assert result["memory_ids"] == [memory.memory_id]
    assert events[-1][0] == "intent_memory_shadow_evaluated"


def test_expired_processing_outbox_lease_is_reclaimable(service: CaseMemoryService) -> None:
    event = service.outbox.enqueue(
        "MEMORY_RECONCILIATION_REQUESTED",
        "missing-memory",
        {"status": "archived"},
    )
    first = service.outbox.claim_next_batch()
    assert first and first[0]["event_id"] == event["event_id"]
    first[0]["locked_until"] = datetime.now(timezone.utc) - timedelta(seconds=1)

    reclaimed = service.outbox.claim_next_batch()

    assert reclaimed and reclaimed[0]["event_id"] == event["event_id"]


def test_explicit_style_preferences_stay_shadow_until_three_observations(
    service: CaseMemoryService,
) -> None:
    orchestrator = object.__new__(CaseAgentService)
    orchestrator._case_memory = service
    for index in range(3):
        run_id = f"style-run-{index}"
        state = {
            "run": SimpleNamespace(run_id=run_id, actor_id="auditor-1"),
            "actor_id": "auditor-1",
            "user_message": SimpleNamespace(content="请先说结论，简洁分点回答"),
            "intent": "case_task",
            "intent_confidence": 0.5,
            "validation_retry_count": 0,
            "planning_error": "",
            "execution_plan": [],
            "capability_results": [],
        }
        assert orchestrator.enqueue_completed_run_memory_capture(state) == 1
        service.process_pending_projections(limit=10)

    styles = [
        row for row in service.repository.list(scope_type="auditor", scope_id="auditor-1")
        if row.memory_level == MemoryLevel.L1
        and row.memory_type == MemoryType.ANSWER_STYLE
    ]
    assert len(styles) == 1
    assert styles[0].shadow_observation_count == 3
    assert styles[0].status == MemoryStatus.CANDIDATE
    assert styles[0].payload["structured_content"]["style_preferences"] == {
        "conclusion_first": True,
        "verbosity": "concise",
        "list_format": True,
    }
