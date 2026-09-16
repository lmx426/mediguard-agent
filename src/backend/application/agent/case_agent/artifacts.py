"""In-run artifact handoff helpers for the Case Agent DAG.

Artifacts are immutable handoff records inside one Caser run. They keep the
parent graph context clean while preserving source refs for answer generation
and checkpoint persistence.
"""

from __future__ import annotations

import json
from threading import RLock
from typing import Any

from src.backend.application.agent.case_agent.tools.registry import CaseAgentToolResult
from src.backend.domain.case_agent.entities import CaseAgentSource


ARTIFACT_TYPE_BY_LAYER = {
    "L1": "l1_result_ref",
    "L2": "l2_result_ref",
    "L3": "l3_result_ref",
}


class InRunArtifactBackend:
    """Process-local backend for full artifact payloads during one Caser run."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._artifacts_by_run: dict[str, dict[str, dict[str, Any]]] = {}

    def write(self, run_key: str, artifact: dict[str, Any]) -> None:
        artifact_ref = str(artifact.get("artifact_ref") or "")
        if not artifact_ref:
            return
        with self._lock:
            self._artifacts_by_run.setdefault(run_key, {})[artifact_ref] = artifact

    def read(self, run_key: str, artifact_ref: str) -> dict[str, Any] | None:
        with self._lock:
            artifact = self._artifacts_by_run.get(run_key, {}).get(artifact_ref)
            return dict(artifact) if isinstance(artifact, dict) else None

    def clear_run(self, run_key: str) -> None:
        with self._lock:
            self._artifacts_by_run.pop(run_key, None)


_ARTIFACT_BACKEND = InRunArtifactBackend()


def ensure_artifact_store(state: dict[str, Any]) -> dict[str, Any]:
    store = state.get("artifact_store")
    if not isinstance(store, dict):
        store = {}
    artifacts = store.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    by_layer = store.get("by_layer")
    if not isinstance(by_layer, dict):
        by_layer = {"L1": [], "L2": [], "L3": []}
    store["artifacts"] = artifacts
    store["by_layer"] = by_layer
    state["artifact_store"] = store
    return store


def artifact_type_for_layer(layer: str) -> str:
    return ARTIFACT_TYPE_BY_LAYER.get(str(layer or "").upper(), "step_result_ref")


def clear_runtime_artifacts(run_or_state: Any) -> None:
    """Clear full payload artifacts after the run no longer needs handoff data."""

    _ARTIFACT_BACKEND.clear_run(_run_key(run_or_state))


def write_step_artifact(
    state: dict[str, Any],
    *,
    step: dict[str, Any],
    result: CaseAgentToolResult,
    artifact_type: str | None = None,
) -> str:
    store = ensure_artifact_store(state)
    layer = str(step.get("layer") or "L1").upper()
    step_id = str(step.get("step_id") or f"step_{step.get('step') or 0}")
    capability = str(step.get("capability") or step.get("name") or "")
    ref_type = artifact_type or artifact_type_for_layer(layer)
    artifact_ref = f"artifact:{ref_type}:{step_id}:{capability}"
    source_refs = [
        ref for ref in result.source_refs
        if isinstance(ref, str) and ref
    ]
    artifact = {
        "artifact_ref": artifact_ref,
        "artifact_type": ref_type,
        "layer": layer,
        "step_id": step_id,
        "step": step.get("step"),
        "capability": capability,
        "status": result.status,
        "error_code": result.error_code,
        "payload": result.payload,
        "source_refs": source_refs,
        "sources": [
            source.model_dump(mode="json")
            for source in result.sources
            if isinstance(source, CaseAgentSource)
        ],
        "latency_ms": result.latency_ms,
    }
    _ARTIFACT_BACKEND.write(_run_key(state), artifact)
    store["artifacts"][artifact_ref] = _artifact_index(artifact)
    store.setdefault("by_layer", {}).setdefault(layer, [])
    if artifact_ref not in store["by_layer"][layer]:
        store["by_layer"][layer].append(artifact_ref)
    artifact_refs = state.setdefault("artifact_refs", [])
    if artifact_ref not in artifact_refs:
        artifact_refs.append(artifact_ref)
    return artifact_ref


def append_capability_result(
    state: dict[str, Any],
    *,
    capability: str,
    result: CaseAgentToolResult,
) -> None:
    results = [
        item for item in state.get("capability_results", [])
        if isinstance(item, tuple) and len(item) == 2
    ]
    results.append((capability, _lightweight_result(result)))
    state["capability_results"] = results

    source_by_ref: dict[str, CaseAgentSource] = {}
    for source in state.get("available_sources", []):
        if isinstance(source, CaseAgentSource):
            source_by_ref[source.source_ref] = source
    for source in result.sources:
        if isinstance(source, CaseAgentSource):
            source_by_ref[source.source_ref] = source
    state["available_sources"] = list(source_by_ref.values())


def artifact_results(state: dict[str, Any]) -> list[tuple[str, CaseAgentToolResult]]:
    store = ensure_artifact_store(state)
    results: list[tuple[str, CaseAgentToolResult]] = []
    for ref in state.get("artifact_refs", []):
        artifact = _resolve_artifact(state, str(ref))
        if not isinstance(artifact, dict):
            continue
        sources = [
            CaseAgentSource.model_validate(item)
            for item in artifact.get("sources", [])
            if isinstance(item, dict)
        ]
        result = CaseAgentToolResult(
            payload=artifact.get("payload")
            if isinstance(artifact.get("payload"), dict)
            else {},
            source_refs=[
                item for item in artifact.get("source_refs", [])
                if isinstance(item, str)
            ],
            sources=sources,
            status=str(artifact.get("status") or "success"),
            error_code=artifact.get("error_code")
            if isinstance(artifact.get("error_code"), str)
            else None,
            latency_ms=int(artifact.get("latency_ms") or 0),
        )
        results.append((str(artifact.get("capability") or ""), result))
    return results


def dependency_artifacts(
    state: dict[str, Any],
    step: dict[str, Any],
) -> list[dict[str, Any]]:
    store = ensure_artifact_store(state)
    artifacts = store.get("artifacts", {})
    wanted = {
        str(item) for item in step.get("depends_on", [])
        if item not in (None, "")
    }
    if not wanted:
        step_no = int(step.get("step") or 0)
        wanted = {
            str(item.get("step_id"))
            for item in state.get("execution_dag", {}).get("steps", [])
            if isinstance(item, dict)
            and str(item.get("status")) == "completed"
            and int(item.get("step") or 0) < step_no
        }
    result: list[dict[str, Any]] = []
    for ref, index in artifacts.items():
        if not isinstance(index, dict):
            continue
        if str(index.get("step_id")) in wanted or str(index.get("step")) in wanted:
            artifact = _resolve_artifact(state, str(ref))
            if isinstance(artifact, dict):
                result.append(artifact)
    return result


def all_artifact_sources(state: dict[str, Any]) -> list[CaseAgentSource]:
    source_by_ref: dict[str, CaseAgentSource] = {}
    for _, result in artifact_results(state):
        for source in result.sources:
            source_by_ref[source.source_ref] = source
    return list(source_by_ref.values())


def _run_key(run_or_state: Any) -> str:
    if isinstance(run_or_state, dict):
        run_id = run_or_state.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
        run = run_or_state.get("run")
    else:
        run = run_or_state
    run_id = getattr(run, "run_id", None)
    if isinstance(run_id, str) and run_id:
        return run_id
    return "__local__"


def _resolve_artifact(state: dict[str, Any], artifact_ref: str) -> dict[str, Any] | None:
    artifact = _ARTIFACT_BACKEND.read(_run_key(state), artifact_ref)
    if isinstance(artifact, dict):
        return artifact
    store = ensure_artifact_store(state)
    legacy = store.get("artifacts", {}).get(artifact_ref)
    if isinstance(legacy, dict) and isinstance(legacy.get("payload"), dict):
        return legacy
    return None


def _artifact_index(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = artifact.get("payload")
    source_refs = [
        ref for ref in artifact.get("source_refs", [])
        if isinstance(ref, str) and ref
    ]
    return {
        "artifact_ref": artifact.get("artifact_ref"),
        "artifact_type": artifact.get("artifact_type"),
        "layer": artifact.get("layer"),
        "step_id": artifact.get("step_id"),
        "step": artifact.get("step"),
        "capability": artifact.get("capability"),
        "status": artifact.get("status"),
        "error_code": artifact.get("error_code"),
        "source_refs": source_refs,
        "source_count": len(source_refs),
        "payload_summary": _payload_summary(payload),
        "latency_ms": artifact.get("latency_ms"),
    }


def _lightweight_result(result: CaseAgentToolResult) -> CaseAgentToolResult:
    return CaseAgentToolResult(
        payload=_payload_summary(result.payload),
        source_refs=[
            ref for ref in result.source_refs
            if isinstance(ref, str) and ref
        ],
        sources=[],
        status=result.status,
        error_code=result.error_code,
        latency_ms=result.latency_ms,
    )


def _payload_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "payload_omitted": True,
            "payload_type": type(payload).__name__,
        }
    section_payload = payload.get("payload")
    summary: dict[str, Any] = {
        "status": payload.get("status"),
        "section_key": payload.get("section_key"),
        "capability": payload.get("capability"),
        "payload_omitted": True,
        "payload_bytes": _json_size(payload),
    }
    if isinstance(section_payload, dict):
        summary["payload_keys"] = list(section_payload.keys())[:24]
    else:
        summary["payload_keys"] = list(payload.keys())[:24]
    return {
        key: value for key, value in summary.items()
        if value not in (None, "", [], {})
    }


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))
