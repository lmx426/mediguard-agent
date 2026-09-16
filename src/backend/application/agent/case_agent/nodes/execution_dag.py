"""Execution DAG nodes for Caser-controlled capability dispatch."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.case_agent.artifacts import (
    append_capability_result,
    dependency_artifacts,
    write_step_artifact,
)
from src.backend.application.agent.case_agent.tools.registry import CaseAgentToolResult
from src.backend.domain.case_agent.entities import CaseAgentSource

from ._errors import state_error


def execution_dag_builder_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Build an executable dependency graph from validated plan items."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="execution_dag_builder",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        steps: list[dict[str, Any]] = []
        for index, item in enumerate(state.get("execution_plan", []) or [], start=1):
            if not isinstance(item, dict):
                continue
            step_no = int(item.get("step") or index)
            step_id = str(item.get("step_id") or f"step_{step_no}")
            depends_on = [
                _normalize_dependency(dep)
                for dep in item.get("depends_on", [])
                if dep not in (None, "")
            ]
            steps.append(
                {
                    "step_id": step_id,
                    "step": step_no,
                    "layer": str(item.get("layer") or "L1").upper(),
                    "capability": str(item.get("capability") or item.get("name") or ""),
                    "name": str(item.get("name") or item.get("capability") or ""),
                    "arguments": dict(item.get("arguments") or {}),
                    "reason": str(item.get("reason") or "")[:300],
                    "depends_on": depends_on,
                    "status": "pending",
                    "artifact_ref": None,
                }
            )
        steps.sort(key=lambda item: item["step"])
        state["execution_dag"] = {
            "status": "ready",
            "steps": steps,
            "completed_step_ids": [],
            "failed_step_ids": [],
        }
        state.setdefault("artifact_store", {"artifacts": {}, "by_layer": {}})
        state.setdefault("artifact_refs", [])
        service._repository.append_event(
            run.run_id,
            "execution_dag_built",
            "Execution DAG built",
            {
                "step_count": len(steps),
                "steps": [
                    {
                        "step_id": step["step_id"],
                        "layer": step["layer"],
                        "capability": step["capability"],
                        "depends_on": step["depends_on"],
                    }
                    for step in steps
                ],
            },
        )
        state["next_action"] = "dag_executor"
        return state
    except Exception as exc:
        return state_error(state, exc)


def dag_executor_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Schedule the next ready DAG step or continue to answer assembly."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="dag_executor",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        dag = state.get("execution_dag")
        if not isinstance(dag, dict):
            dag = {"status": "ready", "steps": [], "completed_step_ids": []}
            state["execution_dag"] = dag
        steps = [
            item for item in dag.get("steps", [])
            if isinstance(item, dict)
        ]
        completed = {
            str(item.get("step_id"))
            for item in steps
            if item.get("status") == "completed"
        }
        pending = [item for item in steps if item.get("status") == "pending"]
        if not pending:
            dag["status"] = "completed"
            dag["completed_step_ids"] = list(completed)
            state.pop("current_dag_step", None)
            state["next_action"] = "build_answer_context"
            return state

        ready = [
            item for item in pending
            if all(str(dep) in completed for dep in item.get("depends_on", []))
        ]
        if ready:
            ready.sort(key=lambda item: int(item.get("step") or 0))
            step = ready[0]
            state["current_dag_step"] = dict(step)
            dag["status"] = "running"
            service._repository.append_event(
                run.run_id,
                "dag_step_ready",
                "Execution DAG ready step selected",
                {
                    "step_id": step.get("step_id"),
                    "layer": step.get("layer"),
                    "capability": step.get("capability"),
                },
            )
            state["next_action"] = "dispatch_step"
            return state

        dag["status"] = "blocked"
        state["error_code"] = "ExecutionDagBlocked"
        state["error_message"] = "No ready DAG step and graph is not complete."
        state["next_action"] = "fail_closed"
        return state
    except Exception as exc:
        return state_error(state, exc)


def dispatch_step_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Route one ready DAG step by L1/L2/L3 layer."""

    try:
        run = state["run"]
        step = _current_step(state)
        _mark_step_status(state, step["step_id"], "running")
        service._repository.update_run(
            run.run_id,
            current_node="dispatch_step",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        layer = str(step.get("layer") or "L1").upper()
        if layer == "L1":
            state["next_action"] = "execute_l1_step"
        elif layer == "L2":
            state["next_action"] = "execute_l2_step"
        elif layer == "L3":
            state["next_action"] = "materialize_expert_task"
        else:
            raise ValueError(f"unsupported DAG layer: {layer}")
        return state
    except Exception as exc:
        return state_error(state, exc)


def execute_l1_step_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Execute a read-only Caser View Store query step."""

    try:
        run = state["run"]
        step = _current_step(state)
        capability = str(step.get("capability") or "")
        arguments = dict(step.get("arguments") or {})
        result = service._run_tool(
            run.run_id,
            run.case_id,
            f"dag:{step['step_id']}:{capability}",
            capability,
            arguments,
        )
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        _complete_step_with_result(state, step, capability, result)
        state["next_action"] = "dag_executor"
        return state
    except Exception as exc:
        return state_error(state, exc)


def execute_l2_step_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Run controlled derivation over dependency artifacts."""

    try:
        run = state["run"]
        step = _current_step(state)
        capability = str(step.get("capability") or "")
        deps = dependency_artifacts(state, step)
        service._repository.append_event(
            run.run_id,
            "l2_derivation_running",
            "L2 derivation running",
            {
                "step_id": step.get("step_id"),
                "capability": capability,
                "dependency_count": len(deps),
            },
        )
        result = _derive_l2_result(step, deps)
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        service._repository.append_event(
            run.run_id,
            "l2_derivation_complete",
            "L2 derivation completed",
            {
                "step_id": step.get("step_id"),
                "capability": capability,
                "dependency_count": len(deps),
                "source_refs": result.source_refs[:8],
            },
        )
        _complete_step_with_result(state, step, capability, result)
        state["next_action"] = "dag_executor"
        return state
    except Exception as exc:
        return state_error(state, exc)


def materialize_expert_task_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create an isolated ExpertAnalysisTask from DAG step and artifacts."""

    try:
        run = state["run"]
        step = _current_step(state)
        args = dict(step.get("arguments") or {})
        deps = dependency_artifacts(state, step)
        if "source_refs" not in args:
            args["source_refs"] = _source_refs_from_artifacts(deps)
        if "fact_bundle" not in args:
            args["fact_bundle"] = _fact_bundle_from_artifacts(deps)
        task = service._materialize_expert_task(
            run_id=run.run_id,
            case_id=run.case_id,
            tool_name=str(step.get("capability") or "ask_policy_expert"),
            expert_task_type="policy_analysis",
            state=state,
            arguments=args,
        )
        state["current_expert_task"] = task
        service._repository.append_event(
            run.run_id,
            "expert_task_materialized",
            "ExpertAnalysisTask materialized",
            {
                "step_id": step.get("step_id"),
                "task_id": task.task_id,
                "context_ref_count": len(task.context_refs),
                "fact_count": len(task.fact_bundle),
                "allowed_tools": task.allowed_tools,
            },
        )
        state["next_action"] = "call_expert_analysis"
        return state
    except Exception as exc:
        return state_error(state, exc)


def call_expert_analysis_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the L3 Expert Analysis compiled subgraph and write ART3."""

    try:
        step = _current_step(state)
        task = state.get("current_expert_task")
        service._repository.append_event(
            state["run"].run_id,
            "expert_analysis_calling",
            "正在调用 L3 Expert Analysis",
            {
                "step_id": step.get("step_id"),
                "capability": str(step.get("capability") or "ask_policy_expert"),
            },
        )
        result = service._call_expert_analysis_task(
            run_id=state["run"].run_id,
            expert_task_type="policy_analysis",
            state=state,
            task=task,
            tool_name=str(step.get("capability") or "ask_policy_expert"),
        )
        _write_case_context_observation_artifact(state, step, result)
        tool_result = service._expert_result_to_tool_result(
            case_id=state["run"].case_id,
            capability=str(step.get("capability") or "ask_policy_expert"),
            result=result,
        )
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        _complete_step_with_result(
            state,
            step,
            str(step.get("capability") or "ask_policy_expert"),
            tool_result,
        )
        state["next_action"] = "dag_executor"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _current_step(state: dict[str, Any]) -> dict[str, Any]:
    step = state.get("current_dag_step")
    if not isinstance(step, dict):
        raise ValueError("current_dag_step is missing")
    if not step.get("step_id"):
        raise ValueError("current_dag_step.step_id is missing")
    return step


def _complete_step_with_result(
    state: dict[str, Any],
    step: dict[str, Any],
    capability: str,
    result: CaseAgentToolResult,
) -> None:
    artifact_ref = write_step_artifact(state, step=step, result=result)
    append_capability_result(state, capability=capability, result=result)
    _mark_step_status(state, step["step_id"], "completed", artifact_ref=artifact_ref)
    state.pop("current_expert_task", None)
    state.pop("current_dag_step", None)


def _write_case_context_observation_artifact(
    state: dict[str, Any],
    step: dict[str, Any],
    result: Any,
) -> None:
    safe_summary = getattr(result, "safe_summary", {})
    if not isinstance(safe_summary, dict):
        return
    observations = [
        item for item in safe_summary.get("case_context_observations", [])
        if isinstance(item, dict)
    ]
    if not observations:
        return
    source_refs: list[str] = []
    for item in observations:
        source_refs.extend(
            ref for ref in item.get("source_refs", [])
            if isinstance(ref, str) and ref
        )
    source_refs = list(dict.fromkeys(source_refs))
    source_by_ref = {
        source.source_ref: source
        for source in state.get("available_sources", [])
        if isinstance(source, CaseAgentSource)
    }
    sources = [source_by_ref[ref] for ref in source_refs if ref in source_by_ref]
    observation_result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "section_key": "case_context_observation",
            "capability": "case_context_observation",
            "payload": {
                "observations": observations[:8],
                "case_facts_used": (
                    result.payload.get("case_facts_used", [])
                    if isinstance(getattr(result, "payload", None), dict)
                    else []
                ),
                "expert_task_id": getattr(result, "task_id", None),
            },
        },
        source_refs=source_refs,
        sources=sources,
        status="success",
    )
    write_step_artifact(
        state,
        step={
            **step,
            "step_id": f"{step.get('step_id')}_case_context",
            "capability": "case_context_observation",
        },
        result=observation_result,
        artifact_type="case_context_observation_ref",
    )


def _mark_step_status(
    state: dict[str, Any],
    step_id: str,
    status: str,
    *,
    artifact_ref: str | None = None,
) -> None:
    dag = state.get("execution_dag")
    if not isinstance(dag, dict):
        return
    for step in dag.get("steps", []):
        if isinstance(step, dict) and str(step.get("step_id")) == str(step_id):
            step["status"] = status
            if artifact_ref is not None:
                step["artifact_ref"] = artifact_ref
            break
    if status == "completed":
        completed = dag.setdefault("completed_step_ids", [])
        if step_id not in completed:
            completed.append(step_id)


def _normalize_dependency(value: Any) -> str:
    text = str(value)
    if text.isdigit():
        return f"step_{text}"
    return text


def _derive_l2_result(
    step: dict[str, Any],
    deps: list[dict[str, Any]],
) -> CaseAgentToolResult:
    source_refs = _source_refs_from_artifacts(deps)
    sources = _sources_from_artifacts(deps)
    capability = str(step.get("capability") or "")
    payload = {
        "status": "ok" if deps else "not_ready",
        "section_key": "runtime_derivation",
        "capability": capability,
        "payload": {
            "derived_facts": _derived_facts_from_artifacts(deps),
            "dependency_artifact_refs": [
                item.get("artifact_ref") for item in deps
                if isinstance(item.get("artifact_ref"), str)
            ],
            "mode": capability,
        },
    }
    return CaseAgentToolResult(
        payload=payload,
        source_refs=source_refs,
        sources=sources,
        status="success" if deps else "unavailable",
        error_code=None if deps else "dependency_artifacts_missing",
    )


def _derived_facts_from_artifacts(deps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for artifact in deps[:8]:
        payload = artifact.get("payload")
        if not isinstance(payload, dict):
            continue
        section_key = payload.get("section_key") or artifact.get("capability")
        section_payload = payload.get("payload")
        facts.append(
            {
                "label": str(section_key or "dependency"),
                "status": artifact.get("status"),
                "source_refs": artifact.get("source_refs", [])[:8],
                "summary": _payload_summary(section_payload if section_payload is not None else payload),
            }
        )
    return facts


def _payload_summary(value: Any) -> Any:
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in list(value.items())[:12]:
            if item in (None, "", [], {}):
                continue
            if isinstance(item, list):
                summary[str(key)] = {"count": len(item), "sample": item[:3]}
            elif isinstance(item, dict):
                summary[str(key)] = {"keys": list(item.keys())[:8]}
            else:
                summary[str(key)] = item
        return summary
    if isinstance(value, list):
        return {"count": len(value), "sample": value[:3]}
    return value


def _source_refs_from_artifacts(deps: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for artifact in deps:
        refs.extend(
            ref for ref in artifact.get("source_refs", [])
            if isinstance(ref, str) and ref
        )
    return list(dict.fromkeys(refs))


def _sources_from_artifacts(deps: list[dict[str, Any]]) -> list[CaseAgentSource]:
    sources: dict[str, CaseAgentSource] = {}
    for artifact in deps:
        for item in artifact.get("sources", []):
            if not isinstance(item, dict):
                continue
            source = CaseAgentSource.model_validate(item)
            sources[source.source_ref] = source
    return list(sources.values())


def _fact_bundle_from_artifacts(deps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for artifact in deps[:8]:
        payload = artifact.get("payload")
        if not isinstance(payload, dict):
            continue
        section_key = payload.get("section_key") or artifact.get("capability")
        facts.append(
            {
                "label": str(section_key or "artifact"),
                "value": _payload_summary(payload.get("payload") or payload),
                "source_refs": artifact.get("source_refs", [])[:8],
                "artifact_ref": artifact.get("artifact_ref"),
            }
        )
    return facts[:16]
