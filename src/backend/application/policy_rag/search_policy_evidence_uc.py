"""Use case for source-grounded policy evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from src.backend.application.policy_rag.branch_rrf_fusion import (
    fuse_branch_rankings,
)
from src.backend.application.policy_rag.group_planner import (
    RetrievalGroup,
    RetrievalPlan,
    plan_policy_retrieval,
)
from src.backend.application.policy_rag.schemas import (
    DEFAULT_BRANCH_FUSION_LIMIT,
    DEFAULT_BRANCH_POOL_SIZE,
    DEFAULT_MULTI_GROUP_FETCH_K,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_LIMITS,
    FILTER_STRATEGY_DUAL_RRF,
    MODE,
    RETRIEVAL_MODE_FAISS_ONLY,
    RETRIEVAL_MODE_GROUP_AWARE,
    RETRIEVAL_MODE_GROUP_AWARE_STRICT_V3,
    RETRIEVAL_MODE_GROUP_AWARE_WITH_RERANK,
    RETRIEVAL_MODE_WITH_RERANK,
    RETRIEVAL_STRATEGY_DENSE,
    RETRIEVAL_STRATEGY_HYBRID,
    PolicyRagRequestError,
    PolicyRagSearchRequest,
    RetrievedPolicyCandidate,
    invalid_request_response,
    normalize_retrieval_strategy,
    retrieval_profile_for_strategy,
)
from src.backend.domain.policy_rag_taxonomy import CONTENT_TYPES


class PolicyRetriever(Protocol):
    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        """Return filtered retrieval candidates sorted by initial score."""


class PolicyReranker(Protocol):
    @property
    def available(self) -> bool:
        """Whether reranking can be applied."""

    @property
    def unavailable_reason(self) -> str | None:
        """Why the reranker is unavailable, if known."""

    @property
    def model_name(self) -> str:
        """Reranker model name."""

    def rerank(
        self,
        *,
        question: str,
        candidates: list[RetrievedPolicyCandidate],
    ) -> list[RetrievedPolicyCandidate]:
        """Return candidates with rerank scores sorted best-first."""


@dataclass(frozen=True, slots=True)
class RetrievalGroupBundle:
    group: RetrievalGroup
    candidates: list[RetrievedPolicyCandidate]


class SearchPolicyEvidenceUseCase:
    """Search policy evidence without making audit decisions."""

    def __init__(
        self,
        *,
        retriever: PolicyRetriever,
        reranker: PolicyReranker | None = None,
        retrieval_strategy: object | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._retrieval_strategy = normalize_retrieval_strategy(
            retrieval_strategy or _retrieval_strategy_for_retriever(retriever)
        )
        self._retrieval_profile = retrieval_profile_for_strategy(
            self._retrieval_strategy
        )

    def search(self, request_payload: dict[str, object]) -> dict[str, object]:
        try:
            request = PolicyRagSearchRequest.from_mapping(request_payload)
        except PolicyRagRequestError as exc:
            return invalid_request_response(exc)

        warnings: list[str] = []
        requested_strict_filters = request.filters.to_dict()
        strict_filters = dict(requested_strict_filters)
        plan = plan_policy_retrieval(
            question=request.question,
            filters=strict_filters,
        )
        strict_bundles = _retrieve_group_bundles(
            retriever=self._retriever,
            plan=plan,
            top_k=request.top_k,
            fetch_k=request.fetch_k,
        )
        strict_candidates = _branch_candidates(
            strict_bundles,
            limit=min(request.fetch_k, DEFAULT_BRANCH_POOL_SIZE),
        )

        dual_branch = (
            request.filter_strategy == FILTER_STRATEGY_DUAL_RRF
            and request.recall_filters is not None
        )
        requested_recall_filters = (
            request.recall_filters.to_dict()
            if request.recall_filters is not None
            else None
        )
        recall_filters = (
            dict(requested_recall_filters)
            if requested_recall_filters is not None
            else None
        )
        recall_plan: RetrievalPlan | None = None
        recall_candidates: list[RetrievedPolicyCandidate] = []
        if dual_branch and recall_filters is not None:
            recall_plan = _plan_recall_branch(
                strict_plan=plan,
                recall_filters=recall_filters,
            )
            recall_bundles = _retrieve_group_bundles(
                retriever=self._retriever,
                plan=recall_plan,
                top_k=request.top_k,
                fetch_k=request.fetch_k,
            )
            recall_candidates = _branch_candidates(
                recall_bundles,
                limit=min(request.fetch_k, DEFAULT_BRANCH_POOL_SIZE),
            )

        allowed_jurisdictions = set(
            (recall_filters or {}).get("jurisdiction") or []
        )
        strict_candidates = _apply_final_candidate_gate(
            strict_candidates,
            allowed_jurisdictions=allowed_jurisdictions,
            allowed_content_types=set(strict_filters.get("content_type") or []),
            expected_cite=(
                strict_filters.get("can_cite_as_policy_basis")
                if isinstance(strict_filters.get("can_cite_as_policy_basis"), bool)
                else None
            ),
        )
        recall_candidates = _apply_final_candidate_gate(
            recall_candidates,
            allowed_jurisdictions=allowed_jurisdictions,
            allowed_content_types=set((recall_filters or {}).get("content_type") or []),
            expected_cite=(
                (recall_filters or {}).get("can_cite_as_policy_basis")
                if isinstance(
                    (recall_filters or {}).get("can_cite_as_policy_basis"),
                    bool,
                )
                else None
            ),
        )

        branch_counts = _branch_contribution_counts(
            strict_candidates=strict_candidates,
            recall_candidates=recall_candidates,
        )
        candidates = (
            fuse_branch_rankings(
                strict_candidates=strict_candidates,
                recall_candidates=recall_candidates,
                limit=DEFAULT_BRANCH_FUSION_LIMIT,
            )
            if dual_branch
            else strict_candidates
        )

        core_group_order = _core_group_order(plan)
        effective_top_k = (
            _adaptive_final_top_k(len(core_group_order))
            if request.adaptive_top_k
            else request.top_k
        )

        rerank_requested = request.rerank
        rerank_applied = False
        pre_rerank_candidate_count = len(candidates)
        rerank_candidate_limit = _rerank_candidate_limit(request.fetch_k)
        rerank_input_count = 0
        rerank_unavailable_reason: str | None = None
        rerank_condition_met, rerank_trigger_reason = _should_rerank(
            candidates=candidates,
            plan=plan,
            top_k=effective_top_k,
            dual_branch=dual_branch,
        )
        rerank_triggered = rerank_requested and rerank_condition_met
        if rerank_triggered:
            if self._reranker is not None and self._reranker.available:
                try:
                    rerank_candidates = _select_group_aware_candidates(
                        candidates,
                        top_k=min(rerank_candidate_limit, len(candidates)),
                        source_cap=max(1, rerank_candidate_limit),
                        group_order=core_group_order,
                        rerank_applied=False,
                        per_group_min=(2 if len(core_group_order) > 1 else 1),
                    )
                    rerank_input_count = len(rerank_candidates)
                    candidates = self._reranker.rerank(
                        question=request.question,
                        candidates=rerank_candidates,
                    )
                    rerank_applied = True
                except Exception as exc:
                    rerank_unavailable_reason = (
                        f"reranker_failed: {type(exc).__name__}: {exc}"
                    )
                    warnings.append(
                        "reranker failed; results are ranked by retrieval score"
                    )
            else:
                rerank_unavailable_reason = (
                    self._reranker.unavailable_reason
                    if self._reranker is not None
                    else "reranker_not_configured"
                )
                warnings.append(
                    "reranker unavailable; results are ranked by retrieval score"
                )
        elif rerank_requested:
            rerank_unavailable_reason = "conditional_rerank_not_triggered"

        source_cap = _source_cap(
            top_k=effective_top_k,
            multi_group=len(core_group_order) > 1,
        )
        selected = _select_group_aware_candidates(
            candidates,
            top_k=effective_top_k,
            source_cap=source_cap,
            group_order=(
                core_group_order
                if request.adaptive_top_k
                else [group.name for group in plan.groups]
            ),
            rerank_applied=rerank_applied,
            per_group_min=(
                2
                if request.adaptive_top_k and len(core_group_order) > 1
                else 1
            ),
        )
        if not selected:
            warnings.append(
                "filters returned no results; check spelling or corpus coverage"
            )

        retrieval_mode = _retrieval_mode(
            retriever=self._retriever,
            multi_group=len(core_group_order) > 1,
            rerank_applied=rerank_applied,
        )
        if dual_branch:
            retrieval_mode = f"dual_rrf_{retrieval_mode}"
        diagnostics = {
            "filter_strategy": request.filter_strategy,
            "allow_broad_filters": request.allow_broad_filters,
            "strict_candidate_count": len(strict_candidates),
            "recall_candidate_count": len(recall_candidates),
            "deduplicated_candidate_count": branch_counts["union_count"],
            "strict_only_count": branch_counts["strict_only_count"],
            "recall_only_count": branch_counts["recall_only_count"],
            "branch_overlap_count": branch_counts["overlap_count"],
            "pre_rerank_candidate_count": pre_rerank_candidate_count,
            "rerank_candidate_limit": rerank_candidate_limit,
            "rerank_input_count": rerank_input_count,
            "final_context_count": len(selected),
            "evidence_group_count": len(core_group_order),
            "effective_top_k": effective_top_k,
            "branch_pool_limit": min(request.fetch_k, DEFAULT_BRANCH_POOL_SIZE),
            "branch_fusion_limit": DEFAULT_BRANCH_FUSION_LIMIT,
            "metadata_version": _metadata_version(selected),
            "ignored_filter_fields": [],
        }
        return {
            "status": "ok",
            "mode": MODE,
            "retrieval_strategy": self._retrieval_strategy,
            "retrieval_profile": self._retrieval_profile,
            "retrieval_mode": retrieval_mode,
            "question": request.question,
            "requested_filters": requested_strict_filters,
            "filters_used": strict_filters,
            "strict_filters_used": strict_filters,
            "requested_recall_filters": requested_recall_filters,
            "recall_filters_used": recall_filters,
            "filter_strategy": request.filter_strategy,
            "retrieval_plan": {
                "profile": plan.profile,
                "strategy": _plan_strategy_label(
                    plan_strategy=plan.strategy,
                    retrieval_strategy=self._retrieval_strategy,
                ),
                "groups": [
                    {
                        "name": group.name,
                        "filters": group.filters,
                        "priority": group.priority,
                    }
                    for group in plan.groups
                ],
                "source_cap": source_cap,
                "recall_groups": [
                    {
                        "name": group.name,
                        "filters": group.filters,
                        "priority": group.priority,
                    }
                    for group in (recall_plan.groups if recall_plan else ())
                ],
            },
            "top_k": effective_top_k,
            "requested_top_k": request.top_k,
            "adaptive_top_k": request.adaptive_top_k,
            "fetch_k": request.fetch_k,
            "rerank": {
                "requested": rerank_requested,
                "conditional": True,
                "condition_met": rerank_condition_met,
                "triggered": rerank_triggered,
                "trigger_reason": rerank_trigger_reason,
                "candidate_limit": rerank_candidate_limit,
                "input_count": rerank_input_count,
                "scope": "single_pass" if rerank_applied else "none",
                "applied": rerank_applied,
                "model": self._reranker.model_name if self._reranker else None,
                "unavailable_reason": rerank_unavailable_reason,
            },
            "result_count": len(selected),
            "retrieval_diagnostics": diagnostics,
            "evidence": [
                _evidence_payload(
                    candidate,
                    rank=index,
                    rerank_applied=rerank_applied,
                )
                for index, candidate in enumerate(selected, start=1)
            ],
            "warnings": warnings,
            "limits": DEFAULT_LIMITS,
        }


def _dedup_candidates(
    candidates: list[RetrievedPolicyCandidate],
) -> list[RetrievedPolicyCandidate]:
    output: list[RetrievedPolicyCandidate] = []
    seen_keys: dict[tuple[str, object], int] = {}
    for candidate in candidates:
        source_chunk = _candidate_key(candidate)
        existing_index = seen_keys.get(source_chunk)
        if existing_index is not None:
            existing = output[existing_index]
            merged_groups = tuple(
                dict.fromkeys(existing.retrieval_groups + candidate.retrieval_groups)
            )
            if candidate.faiss_score > existing.faiss_score:
                output[existing_index] = replace(
                    candidate,
                    retrieval_groups=merged_groups,
                )
            else:
                output[existing_index] = replace(
                    existing,
                    retrieval_groups=merged_groups,
                )
            continue
        seen_keys[source_chunk] = len(output)
        output.append(candidate)
    return output


def _branch_candidates(
    bundles: list[RetrievalGroupBundle],
    *,
    limit: int,
) -> list[RetrievedPolicyCandidate]:
    candidates = _dedup_candidates(
        [
            candidate
            for bundle in bundles
            for candidate in _dedup_candidates(bundle.candidates)
        ]
    )
    return _select_group_aware_candidates(
        candidates,
        top_k=max(1, int(limit)),
        source_cap=max(1, int(limit)),
        group_order=[bundle.group.name for bundle in bundles],
        rerank_applied=False,
        per_group_min=1,
    )


def _plan_recall_branch(
    *,
    strict_plan: RetrievalPlan,
    recall_filters: dict[str, object],
) -> RetrievalPlan:
    groups = tuple(
        replace(group, filters=dict(recall_filters))
        for group in strict_plan.groups
    )
    return RetrievalPlan(
        groups=groups,
        strategy=f"{strict_plan.strategy}_recall_safe",
        profile=strict_plan.profile,
    )


def _apply_final_candidate_gate(
    candidates: list[RetrievedPolicyCandidate],
    *,
    allowed_jurisdictions: set[str],
    allowed_content_types: set[str],
    expected_cite: bool | None,
) -> list[RetrievedPolicyCandidate]:
    output: list[RetrievedPolicyCandidate] = []
    for candidate in candidates:
        metadata = candidate.metadata
        content_type = str(metadata.get("content_type") or "").strip()
        policy_domain = str(metadata.get("policy_domain") or "").strip()
        if content_type and content_type not in CONTENT_TYPES:
            continue
        if allowed_content_types and content_type not in allowed_content_types:
            continue
        if policy_domain == "graph_hints":
            continue
        jurisdiction = str(metadata.get("jurisdiction") or "").strip()
        if (
            allowed_jurisdictions
            and jurisdiction
            and jurisdiction not in allowed_jurisdictions
        ):
            continue
        cite_value = metadata.get("can_cite_as_policy_basis")
        if expected_cite is not None and cite_value is not expected_cite:
            continue
        output.append(candidate)
    return output


def _branch_contribution_counts(
    *,
    strict_candidates: list[RetrievedPolicyCandidate],
    recall_candidates: list[RetrievedPolicyCandidate],
) -> dict[str, int]:
    strict_keys = {_candidate_key(item) for item in strict_candidates}
    recall_keys = {_candidate_key(item) for item in recall_candidates}
    return {
        "strict_only_count": len(strict_keys - recall_keys),
        "recall_only_count": len(recall_keys - strict_keys),
        "overlap_count": len(strict_keys & recall_keys),
        "union_count": len(strict_keys | recall_keys),
    }


def _core_group_order(plan: RetrievalPlan) -> list[str]:
    core = [
        group.name
        for group in plan.groups
        if group.name != "base" and not group.name.startswith("clause:")
    ]
    return list(dict.fromkeys(core)) or ["general_policy"]


def _adaptive_final_top_k(group_count: int) -> int:
    if group_count <= 1:
        return 6
    if group_count == 2:
        return 8
    if group_count == 3:
        return 10
    return 12


def _metadata_version(candidates: list[RetrievedPolicyCandidate]) -> str:
    versions = [
        str(item.metadata.get("metadata_version") or "").strip()
        for item in candidates
        if str(item.metadata.get("metadata_version") or "").strip()
    ]
    return versions[0] if versions else "legacy_v1"


def _retrieve_group_bundles(
    *,
    retriever: PolicyRetriever,
    plan: RetrievalPlan,
    top_k: int,
    fetch_k: int,
) -> list[RetrievalGroupBundle]:
    """Run one bounded retrieval pass per evidence group."""

    group_fetch_k = _group_fetch_k(plan=plan, top_k=top_k, fetch_k=fetch_k)
    bundles: list[RetrievalGroupBundle] = []
    for group in plan.groups:
        group_candidates = retriever.retrieve(
            question=group.question,
            filters=group.filters,
            fetch_k=group_fetch_k,
        )
        bundles.append(
            RetrievalGroupBundle(
                group=group,
                candidates=[
                    replace(candidate, retrieval_groups=(group.name,))
                    for candidate in group_candidates
                ],
            )
        )
    return bundles


def _group_fetch_k(
    *,
    plan: RetrievalPlan,
    top_k: int,
    fetch_k: int,
) -> int:
    if len(plan.groups) <= 1:
        return fetch_k
    if plan.profile == "strict_v3":
        return min(max(top_k + 3, DEFAULT_MULTI_GROUP_FETCH_K), fetch_k)
    return min(max(top_k * 3, 12), fetch_k)


def _rerank_candidate_limit(fetch_k: int) -> int:
    """Bound CrossEncoder work while leaving enough room for final coverage."""

    return 16 if int(fetch_k) >= 40 else 12


def _should_rerank(
    *,
    candidates: list[RetrievedPolicyCandidate],
    plan: RetrievalPlan,
    top_k: int,
    dual_branch: bool = False,
) -> tuple[bool, str]:
    if dual_branch and candidates:
        return True, "dual_branch_fusion"
    if len(plan.groups) > 1:
        return True, "multiple_evidence_groups"
    if len(candidates) <= top_k:
        return False, "candidate_count_not_above_top_k"
    ranked = sorted(candidates, key=lambda item: item.faiss_score, reverse=True)
    top_sources = {_source_key(item) for item in ranked[:top_k]}
    all_sources = {_source_key(item) for item in ranked}
    if len(top_sources) == 1 and len(all_sources) > 1:
        return True, "single_source_crowding"
    kth_index = min(max(top_k - 1, 0), len(ranked) - 1)
    margin = ranked[0].faiss_score - ranked[kth_index].faiss_score
    if margin < 0.08:
        return True, "low_dense_score_margin"
    return False, "single_group_confident_dense_results"


def _select_group_aware_candidates(
    candidates: list[RetrievedPolicyCandidate],
    *,
    top_k: int,
    source_cap: int,
    group_order: list[str],
    rerank_applied: bool,
    per_group_min: int = 1,
) -> list[RetrievedPolicyCandidate]:
    """Prefer one candidate per evidence group, then fill by score."""

    ranked = sorted(
        candidates,
        key=lambda item: item.score(rerank_applied=rerank_applied),
        reverse=True,
    )
    selected: list[RetrievedPolicyCandidate] = []
    source_counts: dict[str, int] = {}
    selected_ids: set[str] = set()

    for _ in range(max(1, int(per_group_min))):
        for group_name in group_order:
            candidate = next(
                (
                    item
                    for item in ranked
                    if group_name in item.retrieval_groups
                    and item.node_id not in selected_ids
                    and source_counts.get(_source_key(item), 0) < source_cap
                ),
                None,
            )
            if candidate is not None:
                _append_selected(candidate, selected, selected_ids, source_counts)
                if len(selected) >= top_k:
                    return selected

    for candidate in ranked:
        if candidate.node_id in selected_ids:
            continue
        if source_counts.get(_source_key(candidate), 0) >= source_cap:
            continue
        _append_selected(candidate, selected, selected_ids, source_counts)
        if len(selected) >= top_k:
            break
    return selected


def _source_cap(*, top_k: int, multi_group: bool) -> int:
    if multi_group:
        return max(2, min(3, top_k // 2 or 1))
    return top_k


def _plan_strategy_label(
    *,
    plan_strategy: str,
    retrieval_strategy: str,
) -> str:
    if plan_strategy.endswith("_dense") and retrieval_strategy != "dense":
        return f"{plan_strategy[:-len('_dense')]}_{retrieval_strategy}"
    return plan_strategy


def _retrieval_strategy_for_retriever(retriever: PolicyRetriever) -> str:
    mode = str(getattr(retriever, "retrieval_mode_name", "") or "")
    if "hybrid" in mode or "rrf" in mode:
        return RETRIEVAL_STRATEGY_HYBRID
    if mode:
        return RETRIEVAL_STRATEGY_DENSE
    return DEFAULT_RETRIEVAL_STRATEGY


def _append_selected(
    candidate: RetrievedPolicyCandidate,
    selected: list[RetrievedPolicyCandidate],
    selected_ids: set[str],
    source_counts: dict[str, int],
) -> None:
    selected.append(candidate)
    selected_ids.add(candidate.node_id)
    source = _source_key(candidate)
    source_counts[source] = source_counts.get(source, 0) + 1


def _retrieval_mode(
    *,
    retriever: PolicyRetriever,
    multi_group: bool,
    rerank_applied: bool,
) -> str:
    base_mode = str(
        getattr(retriever, "retrieval_mode_name", RETRIEVAL_MODE_FAISS_ONLY)
    )
    if base_mode == RETRIEVAL_MODE_FAISS_ONLY:
        if rerank_applied and multi_group:
            return RETRIEVAL_MODE_GROUP_AWARE_WITH_RERANK
        if rerank_applied:
            return RETRIEVAL_MODE_WITH_RERANK
        if multi_group:
            return RETRIEVAL_MODE_GROUP_AWARE_STRICT_V3
        return RETRIEVAL_MODE_FAISS_ONLY

    mode = base_mode
    if multi_group and not mode.startswith("group_aware_"):
        mode = f"group_aware_strict_v3_{mode}"
    if rerank_applied:
        mode = f"{mode}_with_conditional_bge_reranker_v2_m3"
    return mode


def _evidence_payload(
    candidate: RetrievedPolicyCandidate,
    *,
    rank: int,
    rerank_applied: bool,
) -> dict[str, object]:
    metadata = candidate.metadata
    score_type = (
        "rerank_score"
        if rerank_applied
        else str(metadata.get("score_type") or "faiss_score")
    )
    return {
        "rank": rank,
        "score": candidate.score(rerank_applied=rerank_applied),
        "score_type": score_type,
        "faiss_score": candidate.faiss_score,
        "rerank_score": candidate.rerank_score,
        "dense_score": metadata.get("dense_score"),
        "lexical_score": metadata.get("lexical_score"),
        "hybrid_score": metadata.get("hybrid_score"),
        "retrieval_strategy": metadata.get("retrieval_strategy"),
        "retrieval_branches": metadata.get("retrieval_branches", ["strict"]),
        "branch_ranks": metadata.get("branch_ranks"),
        "branch_rrf_score": metadata.get("branch_rrf_score"),
        "node_id": candidate.node_id,
        "text": candidate.text,
        "retrieval_groups": list(candidate.retrieval_groups),
        "title": metadata.get("title"),
        "source_url": metadata.get("source_url"),
        "source_id": metadata.get("source_id"),
        "jurisdiction": metadata.get("jurisdiction"),
        "policy_domain": metadata.get("policy_domain"),
        "content_type": metadata.get("content_type"),
        "doc_type": metadata.get("doc_type"),
        "doc_id": metadata.get("doc_id"),
        "section_heading": metadata.get("section_heading"),
        "chunk_index": metadata.get("chunk_index"),
        "evidence_role": metadata.get("evidence_role"),
        "can_cite_as_policy_basis": metadata.get("can_cite_as_policy_basis"),
    }


def _source_key(candidate: RetrievedPolicyCandidate) -> str:
    return str(
        candidate.metadata.get("source_id")
        or candidate.metadata.get("doc_id")
        or candidate.node_id.split("::", 1)[0]
    )


def _candidate_key(candidate: RetrievedPolicyCandidate) -> tuple[str, object]:
    chunk_index = candidate.metadata.get("chunk_index")
    return (
        _source_key(candidate),
        chunk_index if chunk_index is not None else candidate.node_id,
    )
