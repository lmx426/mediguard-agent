"""Runtime intent example matcher for the Case Agent perception layer."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


MatcherBackend = Literal["auto", "bge", "ngram"]


@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    ability_layer: str
    capability_hint: str | None
    answer_shape: str
    evidence_need: str
    status: str


@dataclass(frozen=True)
class IntentExample:
    intent_id: str
    text: str
    normalized_text: str
    spec: IntentSpec


def load_intent_seed_examples(*, examples_path: Path, taxonomy_path: Path) -> list[IntentExample]:
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    seed = json.loads(examples_path.read_text(encoding="utf-8"))
    specs = {
        str(item.get("intent_id") or ""): IntentSpec(
            intent_id=str(item.get("intent_id") or ""),
            ability_layer=str(item.get("ability_layer") or ""),
            capability_hint=(
                str(item.get("capability_hint"))
                if item.get("capability_hint") is not None
                else None
            ),
            answer_shape=str(item.get("answer_shape") or "overview"),
            evidence_need=str(item.get("evidence_need") or "none"),
            status=str(item.get("status") or ""),
        )
        for item in taxonomy.get("intents", [])
        if isinstance(item, dict) and item.get("intent_id")
    }
    examples: list[IntentExample] = []
    for group in seed.get("example_groups", []):
        if not isinstance(group, dict):
            continue
        intent_id = str(group.get("intent_id") or "")
        spec = specs.get(intent_id)
        if spec is None:
            continue
        for text in group.get("examples", []):
            normalized = normalize_intent_text(str(text or ""))
            if not normalized:
                continue
            examples.append(
                IntentExample(
                    intent_id=intent_id,
                    text=str(text),
                    normalized_text=normalized,
                    spec=spec,
                )
            )
    return examples


class IntentExampleMatcher:
    """Small example retriever with optional local BGE embeddings.

    The matcher is deliberately bounded to perception work: it returns intent
    candidates and routing hints, never an ExecutionPlan.
    """

    def __init__(
        self,
        *,
        examples_path: Path,
        taxonomy_path: Path,
        index_dir: Path | None = None,
        encoder_model: str = "BAAI/bge-small-zh-v1.5",
        model_cache_dir: Path | None = None,
        backend: MatcherBackend = "auto",
        high_confidence_top1: float = 0.78,
        high_confidence_margin: float = 0.08,
        medium_confidence_top1: float = 0.70,
    ) -> None:
        self._examples_path = examples_path
        self._taxonomy_path = taxonomy_path
        self._index_dir = index_dir
        self._encoder_model = encoder_model
        self._model_cache_dir = model_cache_dir
        self._requested_backend = backend
        self._high_confidence_top1 = high_confidence_top1
        self._high_confidence_margin = high_confidence_margin
        self._medium_confidence_top1 = medium_confidence_top1
        self._lock = threading.Lock()
        self._prewarm_lock = threading.Lock()
        self._active_backend = "uninitialized"
        self._encoder: Any | None = None
        self._example_embeddings: Any | None = None
        self._example_ngrams: list[dict[str, float]] = []
        self._index_manifest: dict[str, Any] = {}
        self._prewarm_metrics: dict[str, Any] = {
            "ready": False,
            "total_ms": 0.0,
            "probe_encode_ms": 0.0,
        }
        self._examples = self._load_examples()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "examples_path": str(self._examples_path),
            "taxonomy_path": str(self._taxonomy_path),
            "index_dir": str(self._index_dir) if self._index_dir is not None else "",
            "encoder_model": self._encoder_model,
            "model_cache_dir": str(self._model_cache_dir) if self._model_cache_dir is not None else "",
            "requested_backend": self._requested_backend,
            "active_backend": self._active_backend,
            "example_count": len(self._examples),
            "index_manifest": self._index_manifest,
            "prewarm": dict(self._prewarm_metrics),
        }

    def prewarm(self) -> dict[str, Any]:
        """Load the index and execute one real query encode before readiness."""

        with self._prewarm_lock:
            if self._prewarm_metrics["ready"]:
                return self.metadata
            started = time.perf_counter()
            self._ensure_index()
            probe_started = time.perf_counter()
            scores = self._score(normalize_intent_text("查询当前案件基本信息"))
            probe_encode_ms = (time.perf_counter() - probe_started) * 1000
            if len(scores) != len(self._examples):
                raise ValueError("intent prewarm score count does not match loaded examples")
            self._prewarm_metrics = {
                "ready": True,
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                "probe_encode_ms": round(probe_encode_ms, 3),
                "probe_score_count": len(scores),
            }
            return self.metadata

    def match(self, text: str) -> dict[str, Any]:
        normalized = normalize_intent_text(text)
        if not normalized:
            return self._fallback("empty_input", [])
        self._ensure_index()
        scores = self._score(normalized)
        ranked_examples = self._rank_examples(scores)
        ranked_intents = self._rank_intents(ranked_examples)
        if not ranked_intents:
            return self._fallback("no_candidates", [])

        top1 = ranked_intents[0]
        top2_score = ranked_intents[1]["score"] if len(ranked_intents) > 1 else 0.0
        margin = float(top1["score"]) - float(top2_score)
        ranked_capabilities = self._rank_capabilities(ranked_intents)
        top_capability = ranked_capabilities[0] if ranked_capabilities else None
        capability_score = float(top_capability["score"]) if top_capability else 0.0
        second_capability_score = (
            float(ranked_capabilities[1]["score"])
            if len(ranked_capabilities) > 1
            else 0.0
        )
        capability_margin = capability_score - second_capability_score
        blocked_reason = self._blocked_accept_reason(normalized, top1, margin)
        spec: IntentSpec = top1["spec"]
        safe_direct_l1 = bool(
            spec.ability_layer == "L1"
            and spec.status == "available"
            and spec.capability_hint
            and top_capability
            and top_capability["capability"] == spec.capability_hint
        )
        decision = "fallback"
        reason = "below_medium_threshold"
        if blocked_reason:
            decision = "candidate_only"
            reason = blocked_reason
        elif float(top1["score"]) >= self._high_confidence_top1 and (
            margin >= self._high_confidence_margin
            or (
                safe_direct_l1
                and capability_margin >= self._high_confidence_margin
            )
        ):
            decision = "accept"
            reason = (
                "safe_read_only_l1_capability"
                if safe_direct_l1 and margin < self._high_confidence_margin
                else "high_confidence"
            )
        elif float(top1["score"]) >= self._medium_confidence_top1:
            decision = "candidate_only"
            reason = "medium_confidence"

        return {
            "source": "intent_example_biencoder",
            "decision": decision,
            "reason": reason,
            "business_intent": self._coarse_business_intent(top1["spec"]),
            "matched_intent_id": top1["intent_id"],
            "ability_layer": top1["spec"].ability_layer,
            "target_layer_hint": self._target_layer_hint(top1["spec"]),
            "capability_hint": top1["spec"].capability_hint,
            "answer_shape": _normalize_answer_shape(top1["spec"].answer_shape),
            "evidence_need": top1["spec"].evidence_need,
            "confidence": round(float(top1["score"]), 4),
            "margin": round(margin, 4),
            "capability_confidence": round(capability_score, 4),
            "capability_margin": round(capability_margin, 4),
            "backend": self._active_backend,
            "matched_examples": ranked_examples[:5],
            "top_candidates": [self._public_candidate(candidate) for candidate in ranked_intents[:3]],
            "capability_candidates": ranked_capabilities[:3],
        }

    def _load_examples(self) -> list[IntentExample]:
        return load_intent_seed_examples(
            examples_path=self._examples_path,
            taxonomy_path=self._taxonomy_path,
        )

    def _ensure_index(self) -> None:
        if self._active_backend not in {"uninitialized", "failed"}:
            return
        with self._lock:
            if self._active_backend not in {"uninitialized", "failed"}:
                return
            if self._requested_backend in {"auto", "bge"}:
                try:
                    if self._runtime_index_available():
                        self._load_runtime_vector_index()
                        self._active_backend = "bge_index"
                    else:
                        self._load_bge_index()
                        self._active_backend = "bge"
                    return
                except Exception:
                    if self._requested_backend == "bge":
                        raise
                    self._active_backend = "failed"
            self._load_ngram_index()
            self._active_backend = "ngram"

    def _runtime_index_available(self) -> bool:
        if self._index_dir is None:
            return False
        return (
            (self._index_dir / "examples.jsonl").is_file()
            and (self._index_dir / "embeddings.npy").is_file()
            and (self._index_dir / "manifest.json").is_file()
        )

    def _load_runtime_vector_index(self) -> None:
        import numpy as np

        if self._index_dir is None:
            raise FileNotFoundError("intent example index_dir is not configured")
        manifest_path = self._index_dir / "manifest.json"
        examples_path = self._index_dir / "examples.jsonl"
        embeddings_path = self._index_dir / "embeddings.npy"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        examples = _load_index_examples(examples_path)
        embeddings = np.load(embeddings_path)
        if embeddings.ndim != 2:
            raise ValueError("intent example embeddings must be a 2D matrix")
        if int(embeddings.shape[0]) != len(examples):
            raise ValueError("intent example embeddings count does not match examples.jsonl")
        model_name = str(manifest.get("encoder_model") or self._encoder_model)
        self._encoder = self._load_query_encoder(model_name)
        self._example_embeddings = embeddings
        self._examples = examples
        self._encoder_model = model_name
        self._index_manifest = manifest

    def _load_bge_index(self) -> None:
        encoder = self._load_query_encoder(self._encoder_model)
        texts = [example.normalized_text for example in self._examples]
        embeddings = encoder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._encoder = encoder
        self._example_embeddings = embeddings

    def _load_query_encoder(self, model_name: str) -> Any:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"local_files_only": True}
        if self._model_cache_dir is not None:
            kwargs["cache_folder"] = str(self._model_cache_dir)
        return SentenceTransformer(model_name, **kwargs)

    def _load_ngram_index(self) -> None:
        self._example_ngrams = [_ngram_vector(example.normalized_text) for example in self._examples]

    def _score(self, normalized: str) -> list[float]:
        if self._active_backend in {"bge", "bge_index"} and self._encoder is not None and self._example_embeddings is not None:
            query_embedding = self._encoder.encode(
                [normalized],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            scores = self._example_embeddings @ query_embedding
            return [float(score) for score in scores.tolist()]
        query_vector = _ngram_vector(normalized)
        return [_cosine_dict(query_vector, example_vector) for example_vector in self._example_ngrams]

    def _rank_examples(self, scores: list[float]) -> list[dict[str, Any]]:
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        result: list[dict[str, Any]] = []
        for index, score in ranked[:8]:
            example = self._examples[index]
            result.append(
                {
                    "intent_id": example.intent_id,
                    "text": example.text,
                    "score": round(float(score), 4),
                    "ability_layer": example.spec.ability_layer,
                    "capability_hint": example.spec.capability_hint,
                }
            )
        return result

    def _rank_intents(self, ranked_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_intent: dict[str, dict[str, Any]] = {}
        for item in ranked_examples:
            intent_id = str(item.get("intent_id") or "")
            score = float(item.get("score") or 0.0)
            current = by_intent.get(intent_id)
            if current is not None and float(current.get("score") or 0.0) >= score:
                continue
            example = next((candidate for candidate in self._examples if candidate.intent_id == intent_id), None)
            if example is None:
                continue
            by_intent[intent_id] = {
                "intent_id": intent_id,
                "score": score,
                "spec": example.spec,
                "matched_example": item.get("text") or "",
            }
        return sorted(by_intent.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)

    @staticmethod
    def _rank_capabilities(ranked_intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate semantically adjacent intents at the executable capability boundary."""

        by_capability: dict[str, dict[str, Any]] = {}
        for item in ranked_intents:
            spec = item.get("spec")
            if not isinstance(spec, IntentSpec) or not spec.capability_hint:
                continue
            capability = spec.capability_hint
            score = float(item.get("score") or 0.0)
            current = by_capability.get(capability)
            if current is None:
                by_capability[capability] = {
                    "capability": capability,
                    "score": round(score, 4),
                    "intent_ids": [spec.intent_id],
                    "ability_layer": spec.ability_layer,
                }
                continue
            current["intent_ids"].append(spec.intent_id)
            current["score"] = round(max(float(current["score"]), score), 4)
        return sorted(
            by_capability.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

    def _blocked_accept_reason(self, normalized: str, top1: dict[str, Any], margin: float) -> str:
        spec: IntentSpec = top1["spec"]
        if spec.status == "blocked_by_policy" or spec.ability_layer == "GOVERNANCE":
            return "governance_intent_requires_guardrail"
        if _looks_like_strong_followup(normalized):
            return "followup_requires_context"
        if margin < self._high_confidence_margin:
            return ""
        return ""

    @staticmethod
    def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        spec = candidate.get("spec")
        if not isinstance(spec, IntentSpec):
            return {
                "intent_id": str(candidate.get("intent_id") or ""),
                "score": round(float(candidate.get("score") or 0.0), 4),
                "matched_example": str(candidate.get("matched_example") or ""),
            }
        return {
            "intent_id": spec.intent_id,
            "score": round(float(candidate.get("score") or 0.0), 4),
            "matched_example": str(candidate.get("matched_example") or ""),
            "ability_layer": spec.ability_layer,
            "capability_hint": spec.capability_hint,
            "answer_shape": _normalize_answer_shape(spec.answer_shape),
            "evidence_need": spec.evidence_need,
        }

    def _fallback(self, reason: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "source": "intent_example_biencoder",
            "decision": "fallback",
            "reason": reason,
            "business_intent": "general_help",
            "matched_intent_id": "",
            "ability_layer": "none",
            "target_layer_hint": "none",
            "capability_hint": None,
            "answer_shape": "overview",
            "evidence_need": "none",
            "confidence": 0.0,
            "margin": 0.0,
            "backend": self._active_backend,
            "matched_examples": [],
            "top_candidates": candidates,
        }

    @staticmethod
    def _coarse_business_intent(spec: IntentSpec) -> str:
        if spec.ability_layer == "L3":
            return "expert_task"
        if spec.ability_layer in {"L1", "L2"}:
            return "case_task"
        if spec.status == "blocked_by_policy":
            return "clarification_required"
        return "general_help"

    @staticmethod
    def _target_layer_hint(spec: IntentSpec) -> str:
        if spec.ability_layer in {"L1", "L2", "L3"}:
            return spec.ability_layer
        return "none"


def _load_index_examples(path: Path) -> list[IntentExample]:
    examples: list[IntentExample] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            spec = IntentSpec(
                intent_id=str(payload.get("intent_id") or ""),
                ability_layer=str(payload.get("ability_layer") or ""),
                capability_hint=(
                    str(payload.get("capability_hint"))
                    if payload.get("capability_hint") is not None
                    else None
                ),
                answer_shape=str(payload.get("answer_shape") or "overview"),
                evidence_need=str(payload.get("evidence_need") or "none"),
                status=str(payload.get("status") or ""),
            )
            text = str(payload.get("text") or "")
            normalized = normalize_intent_text(str(payload.get("normalized_text") or text))
            if not text or not normalized or not spec.intent_id:
                continue
            examples.append(
                IntentExample(
                    intent_id=spec.intent_id,
                    text=text,
                    normalized_text=normalized,
                    spec=spec,
                )
            )
    return examples


def normalize_intent_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。！？、；：,.!?;:\"'`~()\[\]{}<>《》【】（）]", "", normalized)
    return normalized.strip()


def _normalize_answer_shape(value: str) -> str:
    mapping = {
        "summary": "overview",
        "comparison": "analysis",
        "explanation": "analysis",
        "policy_analysis": "analysis",
        "source_detail": "detail",
        "clarification": "overview",
        "unavailable": "overview",
    }
    answer_shape = mapping.get(str(value or ""), str(value or "overview"))
    if answer_shape in {"single_field", "field_group", "list", "detail", "overview", "analysis"}:
        return answer_shape
    return "overview"


def _ngram_vector(text: str) -> dict[str, float]:
    vector: dict[str, float] = {}
    chars = list(text)
    for n in (1, 2, 3):
        if len(chars) < n:
            continue
        for index in range(0, len(chars) - n + 1):
            gram = "".join(chars[index : index + n])
            vector[gram] = vector.get(gram, 0.0) + 1.0
    return vector


def _cosine_dict(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _looks_like_strong_followup(normalized: str) -> bool:
    if len(normalized) > 12:
        return False
    followup_markers = ("这个", "它", "上一", "刚才", "第一个", "第二个", "为什么", "怎么会")
    explicit_markers = ("风险", "材料", "政策", "规则", "就诊", "费用", "药", "审核")
    return any(marker in normalized for marker in followup_markers) and not any(
        marker in normalized for marker in explicit_markers
    )
