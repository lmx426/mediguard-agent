"""bge-reranker-v2-m3 adapter for policy evidence reranking."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.device import resolve_policy_model_device
from src.backend.infrastructure.policy_rag.paths import DEFAULT_MODEL_CACHE


DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MAX_LENGTH = 1024
DEFAULT_RERANKER_BATCH_SIZE = 16


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class BgePolicyReranker:
    """Reranks dense retrieval candidates with a cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        cache_folder: Path = DEFAULT_MODEL_CACHE,
        device: str | None = None,
        allow_unavailable: bool = True,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._cache_folder = cache_folder
        self._device = resolve_policy_model_device(device)
        self._model: CrossEncoder | None = None
        self._unavailable_reason: str | None = None
        self._cache_folder.mkdir(parents=True, exist_ok=True)
        try:
            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=self._device,
                cache_folder=str(cache_folder),
                trust_remote_code=False,
                local_files_only=_env_bool("MEDIGUARD_POLICY_RAG_LOCAL_FILES_ONLY", True),
            )
        except Exception as exc:
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            if not allow_unavailable:
                raise

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        *,
        question: str,
        candidates: list[RetrievedPolicyCandidate],
    ) -> list[RetrievedPolicyCandidate]:
        if not candidates:
            return []
        if self._model is None:
            raise RuntimeError(self._unavailable_reason or "reranker_unavailable")

        pairs = [(question, candidate.text) for candidate in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1).tolist()
        reranked = [
            RetrievedPolicyCandidate(
                node_id=candidate.node_id,
                text=candidate.text,
                metadata=dict(candidate.metadata),
                faiss_score=candidate.faiss_score,
                rerank_score=float(score),
                retrieval_groups=candidate.retrieval_groups,
            )
            for candidate, score in zip(candidates, score_values, strict=True)
        ]
        return sorted(
            reranked,
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )
