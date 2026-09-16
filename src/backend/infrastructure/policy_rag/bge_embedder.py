"""bge-m3 query embedding adapter for policy RAG retrieval."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.backend.infrastructure.policy_rag.device import resolve_policy_model_device
from src.backend.infrastructure.policy_rag.paths import DEFAULT_MODEL_CACHE


DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MAX_LENGTH = 2048


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class BgeQueryEmbedder:
    """Embeds user policy questions with the same model used for the FAISS index."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
        cache_folder: Path = DEFAULT_MODEL_CACHE,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._cache_folder = cache_folder
        self._device = resolve_policy_model_device(device)
        self._cache_folder.mkdir(parents=True, exist_ok=True)
        self._model = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=max_length,
            normalize=True,
            cache_folder=str(cache_folder),
            device=self._device,
            trust_remote_code=False,
            show_progress_bar=False,
            local_files_only=_env_bool("MEDIGUARD_POLICY_RAG_LOCAL_FILES_ONLY", True),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def cache_folder(self) -> Path:
        return self._cache_folder

    @property
    def device(self) -> str | None:
        return self._device

    def embed_query(self, question: str) -> np.ndarray:
        embedding = np.asarray(
            self._model.get_query_embedding(question),
            dtype=np.float32,
        )
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D query embedding, got {embedding.shape}")
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
