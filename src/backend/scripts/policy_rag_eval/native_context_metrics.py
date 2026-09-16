"""Native Ragas context metrics for offline Policy RAG retrieval evaluation."""

from __future__ import annotations

import asyncio
import importlib
import math
import sys
from types import ModuleType
from typing import Any

from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    first_env,
    load_env_file,
)


class NativeContextMetricsScorer:
    """Thin adapter around Ragas' reference-based context metrics."""

    def __init__(
        self,
        *,
        llm: Any,
        context_precision: Any,
        context_recall: Any,
        judge_model: str,
        ragas_version: str,
    ) -> None:
        self._context_precision = context_precision
        self._context_recall = context_recall
        self._llm = llm
        self.judge_model = judge_model
        self.ragas_version = ragas_version

    @classmethod
    def from_environment(
        cls,
        *,
        env_file: Any = DEFAULT_ENV_FILE,
        judge_model: str = "deepseek-chat",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> "NativeContextMetricsScorer":
        load_env_file(env_file)
        api_key = first_env(
            "MEDIGUARD_RAGAS_EVAL_API_KEY",
            "MEDIGUARD_RAGAS_JUDGE_API_KEY",
            "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY",
            "MEDIGUARD_DEEPSEEK_API_KEY",
        )
        if not api_key:
            raise RuntimeError(
                "DeepSeek API key is missing. Set it in .env.local as "
                "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY."
            )

        model = str(judge_model or "deepseek-chat").strip() or "deepseek-chat"
        validate_judge_model(model)
        ensure_ragas_import_compatibility()
        ragas_version = installed_ragas_version()
        require_ragas_04(ragas_version)

        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall

        base_url = (
            first_env(
                "MEDIGUARD_RAGAS_EVAL_BASE_URL",
                "MEDIGUARD_RAGAS_JUDGE_BASE_URL",
                "MEDIGUARD_POLICY_RAG_EVAL_BASE_URL",
            )
            or "https://api.deepseek.com"
        )
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=1,
        )
        ragas_llm = llm_factory(
            model,
            provider="openai",
            client=client,
            temperature=temperature,
        )
        return cls(
            llm=ragas_llm,
            context_precision=ContextPrecision(llm=ragas_llm),
            context_recall=ContextRecall(llm=ragas_llm),
            judge_model=model,
            ragas_version=ragas_version,
        )

    async def score(
        self,
        *,
        user_input: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> dict[str, dict[str, Any]]:
        return {
            "context_precision": await self._score_metric(
                self._context_precision,
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            ),
            "context_recall": await self._score_metric(
                self._context_recall,
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            ),
        }

    async def _score_metric(
        self,
        metric: Any,
        *,
        user_input: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> dict[str, Any]:
        if not retrieved_contexts:
            return {"status": "not_evaluable", "reason": "no_contexts", "value": None}
        try:
            result = await metric.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            value = getattr(result, "value", None)
            if value is None or not math.isfinite(float(value)):
                raise ValueError("Ragas returned a non-finite metric value")
            return {"status": "scored", "value": float(value)}
        except Exception as exc:
            return {
                "status": "metric_error",
                "reason": "native_ragas_call_failed",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc)[:500],
                "value": None,
            }

    def score_sync(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return asyncio.run(self.score(**kwargs))


def validate_judge_model(model: str) -> None:
    lowered = str(model).lower()
    risky_markers = ("reasoner", "thinking", "deepseek-r1", "deepseek-v4")
    if any(marker in lowered for marker in risky_markers):
        raise RuntimeError(
            f"Native Ragas requires a non-thinking DeepSeek judge; received {model!r}. "
            "Use --judge-model deepseek-chat."
        )


def require_ragas_04(version: str) -> None:
    parts = str(version).split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if (major, minor) < (0, 4):
        raise RuntimeError(
            f"Native context metrics require ragas>=0.4; installed {version}."
        )


def installed_ragas_version() -> str:
    ensure_ragas_import_compatibility()
    import ragas

    return str(ragas.__version__)


def ensure_ragas_import_compatibility() -> None:
    """Bridge Ragas 0.4.3's optional legacy Vertex AI import when absent."""

    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    compatibility_module = ModuleType(module_name)

    class ChatVertexAI:  # noqa: N801 - matches the upstream class name
        pass

    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module
