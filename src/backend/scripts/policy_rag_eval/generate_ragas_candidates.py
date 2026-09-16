"""Generate Policy RAG v2 candidate questions with DeepSeek.

The output is RAGAS-compatible candidate data. It does not become a formal
goldset until node mapping and audit have passed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_ENV_FILE,
    DEFAULT_NODES_PATH,
    DEFAULT_TIKTOKEN_CACHE,
    DEFAULT_V2_CANDIDATES_PATH,
    SCHEMA_CANDIDATE_V2,
    best_evidence_text,
    citeable_nodes,
    extract_anchor_terms,
    is_generic_answer_point,
    is_title_style_question,
    load_deepseek_config,
    load_policy_nodes,
    parse_json_object,
    ragas_document_for_node,
    stable_id,
    text_contains,
    versioned_eval_path,
    write_jsonl,
)


DEFAULT_SEED = 20260819
BUNDLE_TYPE_RATIOS = [
    ("single_clause", 0.30),
    ("table_lookup", 0.25),
    ("same_doc_multi_clause", 0.15),
    ("cross_doc_policy_combo", 0.225),
    ("case_like_policy_query", 0.075),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek/RAGAS-compatible Policy RAG v2 candidates."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--version-suffix", default=None)
    parser.add_argument(
        "--candidates-jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument("--target-candidates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.candidates_jsonl is None:
        args.candidates_jsonl = (
            versioned_eval_path("candidates", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_CANDIDATES_PATH
        )
    return args


def main() -> None:
    args = parse_args()
    nodes = load_policy_nodes(args.nodes_jsonl)
    source_nodes = citeable_nodes(nodes)
    bundles = build_context_bundles(
        source_nodes,
        target_count=args.target_candidates,
        max_context_chars=args.max_context_chars,
        seed=args.seed,
    )
    if args.dry_run:
        candidates = [
            deterministic_candidate(bundle, index)
            for index, bundle in enumerate(bundles, start=1)
        ][: args.target_candidates]
        model = "dry-run"
    else:
        config = load_deepseek_config(args.env_file)
        candidates = generate_with_deepseek(
            bundles,
            config=config,
            batch_size=args.batch_size,
            target_candidates=args.target_candidates,
            temperature=args.temperature,
        )
        model = config["model"]

    write_jsonl(args.candidates_jsonl, candidates)
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_count": len(candidates),
                "candidate_path": str(args.candidates_jsonl),
                "generator": "deepseek_structured" if not args.dry_run else "dry_run",
                "model": model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_context_bundles(
    nodes: list[Any],
    *,
    target_count: int,
    max_context_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if not nodes:
        return []

    by_domain: dict[str, list[Any]] = {}
    by_doc: dict[str, list[Any]] = {}
    for node in nodes:
        domain = str(node.metadata.get("policy_domain") or "unknown")
        by_domain.setdefault(domain, []).append(node)
        doc_id = str(node.metadata.get("doc_id") or node.metadata.get("source_id") or "")
        if doc_id:
            by_doc.setdefault(doc_id, []).append(node)
    for bucket in by_domain.values():
        rng.shuffle(bucket)
    for bucket in by_doc.values():
        rng.shuffle(bucket)

    table_nodes = [
        node for node in nodes if str(node.metadata.get("content_type") or "") == "table_row"
    ]
    policy_text_nodes = [
        node for node in nodes if str(node.metadata.get("content_type") or "") != "table_row"
    ]
    same_doc_buckets = [bucket for bucket in by_doc.values() if len(bucket) >= 2]
    domains = sorted(by_domain)
    counts = bundle_type_counts(target_count)
    bundles: list[dict[str, Any]] = []
    for bundle_type, count in counts.items():
        for _ in range(count):
            index = len(bundles) + 1
            picked = pick_nodes_for_bundle(
                bundle_type,
                nodes=nodes,
                table_nodes=table_nodes,
                policy_text_nodes=policy_text_nodes,
                same_doc_buckets=same_doc_buckets,
                by_domain=by_domain,
                domains=domains,
                rng=rng,
            )
            contexts = [
                ragas_document_for_node(
                    node,
                    max_context_chars=max_context_chars,
                    context_id=f"ctx_{index:05d}_{position}",
                )
                for position, node in enumerate(picked, start=1)
            ]
            bundles.append(
                {
                    "bundle_id": f"bundle_{bundle_type}_{index:05d}",
                    "bundle_type": bundle_type,
                    "contexts": contexts,
                    "expected_context_count": len(contexts),
                }
            )
    rng.shuffle(bundles)
    for index, bundle in enumerate(bundles, start=1):
        bundle["bundle_id"] = f"bundle_{bundle['bundle_type']}_{index:05d}"
        for position, context in enumerate(bundle["contexts"], start=1):
            context["context_id"] = f"ctx_{index:05d}_{position}"
    return bundles[:target_count]


def bundle_type_counts(target_count: int) -> dict[str, int]:
    raw_counts: list[tuple[str, float, int]] = []
    assigned = 0
    for bundle_type, ratio in BUNDLE_TYPE_RATIOS:
        exact = target_count * ratio
        count = int(exact)
        raw_counts.append((bundle_type, exact - count, count))
        assigned += count
    remainder = max(0, target_count - assigned)
    raw_counts.sort(key=lambda item: item[1], reverse=True)
    counts = {bundle_type: count for bundle_type, _, count in raw_counts}
    for bundle_type, _, _ in raw_counts[:remainder]:
        counts[bundle_type] += 1
    return {bundle_type: counts.get(bundle_type, 0) for bundle_type, _ in BUNDLE_TYPE_RATIOS}


def pick_nodes_for_bundle(
    bundle_type: str,
    *,
    nodes: list[Any],
    table_nodes: list[Any],
    policy_text_nodes: list[Any],
    same_doc_buckets: list[list[Any]],
    by_domain: dict[str, list[Any]],
    domains: list[str],
    rng: random.Random,
) -> list[Any]:
    if bundle_type == "table_lookup":
        return [rng.choice(table_nodes or nodes)]
    if bundle_type == "same_doc_multi_clause" and same_doc_buckets:
        bucket = rng.choice(same_doc_buckets)
        return distinct_nodes(rng.sample(bucket, k=min(len(bucket), rng.randint(2, 3))))
    if bundle_type in {"cross_doc_policy_combo", "case_like_policy_query"} and len(domains) >= 2:
        context_count = rng.randint(2, 3 if bundle_type == "cross_doc_policy_combo" else 4)
        chosen_domains = rng.sample(domains, k=min(len(domains), context_count))
        picked = [rng.choice(by_domain[domain]) for domain in chosen_domains]
        while len(picked) < context_count:
            picked.append(rng.choice(nodes))
        return distinct_nodes(picked)[:context_count]
    return [rng.choice(policy_text_nodes or nodes)]


def distinct_nodes(nodes: list[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for node in nodes:
        unique[str(node.node_id)] = node
    return list(unique.values())


def generate_with_deepseek(
    bundles: list[dict[str, Any]],
    *,
    config: dict[str, str],
    batch_size: int,
    target_candidates: int,
    temperature: float,
) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The openai package is required for DeepSeek calls.") from exc

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=90,
        max_retries=1,
    )
    candidates: list[dict[str, Any]] = []
    for batch_index in range(0, len(bundles), batch_size):
        if len(candidates) >= target_candidates:
            break
        batch = bundles[batch_index : batch_index + batch_size]
        payload = {
            "bundles": [
                {
                    "bundle_id": bundle["bundle_id"],
                    "contexts": [
                        {
                            "context_id": context["context_id"],
                            "metadata": context["metadata"],
                            "page_content": context["page_content"],
                        }
                        for context in bundle["contexts"]
                    ],
                    "bundle_type": bundle.get("bundle_type"),
                    "expected_context_count": bundle.get("expected_context_count"),
                }
                for bundle in batch
            ]
        }
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            stream=False,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_object(content)
        batch_candidates = normalize_generated_candidates(
            parsed.get("candidates"),
            batch=batch,
            start_index=len(candidates) + 1,
            model=config["model"],
        )
        candidates.extend(batch_candidates)
        print(
            f"[v2-candidates] batch={batch_index // batch_size + 1} "
            f"candidates={len(candidates)}/{target_candidates}",
            flush=True,
        )
        time.sleep(0.2)
    return candidates[:target_candidates]


def system_prompt() -> str:
    return (
        "你是医保审核政策 RAG goldset 候选题生成器。只能根据用户提供的 contexts 生成候选题，"
        "输出必须是 JSON object，格式为 {\"candidates\": [...]}，每个 bundle 生成 1 个候选题。"
        "问题必须像审核员在 Case Agent 里自然提问，不要暴露资料标题或来源名称；"
        "严禁以“根据《”“依据《”“按照《”“参照《”开头，严禁在 question 中出现书名号。"
        "不要问“这份通知规定了什么”这类文献阅读题，要问审核动作会遇到的政策核验问题。"
        "reference_answer 只能使用 contexts 中可见事实，不能引入外部知识，不能下拒付/处罚/欺诈结论。"
        "reference_contexts 必须引用输入 context_id，并从原文摘录 evidence_text，不能改写 evidence_text。"
        "answer_points 必须从 reference_answer 拆出 1-4 个自然事实点；事实点数量按答案内容决定，"
        "不要固定 3 个，也不要输出“该问题需要根据候选上下文中的政策事实作答”“根据上下文可知”等空话。"
        "每个 answer_point 必须包含 statement 和 context_ids；context_ids 要指向真正支撑该事实点的 context。"
        "单 context bundle 可以生成 1-2 个事实点；多 context bundle 如果问题需要组合回答，应尽量覆盖多个 context。"
        "proposed_filters 只允许包含 jurisdiction、policy_domain、content_type。"
        "question_type 从 single_hop、multi_hop、table_lookup、policy_reasoning 中选择。"
    )


def normalize_generated_candidates(
    raw_candidates: object,
    *,
    batch: list[dict[str, Any]],
    start_index: int,
    model: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_candidates, list):
        raise ValueError("DeepSeek response missing candidates list")
    by_bundle = {bundle["bundle_id"]: bundle for bundle in batch}
    outputs: list[dict[str, Any]] = []
    for offset, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            continue
        bundle_id = str(raw.get("bundle_id") or "").strip()
        bundle = by_bundle.get(bundle_id) or batch[min(offset, len(batch) - 1)]
        context_by_id = {
            str(context["context_id"]): context for context in bundle["contexts"]
        }
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("reference_answer") or "").strip()
        if not question or not answer or is_title_style_question(question):
            continue

        refs = normalize_reference_contexts(raw, context_by_id)
        answer_points = normalize_answer_points(raw.get("answer_points"), refs, answer)
        if not answer_points:
            continue
        filters = normalize_filters(raw.get("proposed_filters"), refs)
        candidate_index = start_index + len(outputs)
        candidate_id = f"cand_{candidate_index:06d}"
        outputs.append(
            {
                "schema_version": SCHEMA_CANDIDATE_V2,
                "candidate_id": candidate_id,
                "source_bundle_id": bundle["bundle_id"],
                "bundle_type": str(bundle.get("bundle_type") or "single_clause"),
                "source_context_count": len(bundle.get("contexts") or []),
                "question": question,
                "question_type": normalize_question_type(
                    raw.get("question_type"),
                    bundle_type=str(bundle.get("bundle_type") or ""),
                ),
                "reference_answer": answer,
                "answer_points": answer_points,
                "reference_contexts": refs,
                "proposed_filters": filters,
                "generated_by": "deepseek_structured",
                "generation_model": model,
                "ragas_compatibility": {
                    "document_shape": "langchain_document_compatible",
                    "uses_native_ragas_testset_generator": False,
                    "native_ragas_blocker": "ragas.testset imports tiktoken o200k_base when cache is missing",
                    "candidate_style": "ragas_testset_compatible_deepseek_structured",
                },
            }
        )
    return outputs


def normalize_reference_contexts(
    raw: dict[str, Any],
    context_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_refs = raw.get("reference_contexts")
    refs: list[dict[str, Any]] = []
    if isinstance(raw_refs, list):
        for item in raw_refs:
            if not isinstance(item, dict):
                continue
            context_id = str(item.get("context_id") or "").strip()
            source = context_by_id.get(context_id)
            if source is None:
                continue
            text = str(item.get("evidence_text") or item.get("text") or "").strip()
            if not text:
                text = str(source.get("page_content") or "")[:360].strip()
            elif not text_contains(str(source.get("page_content") or ""), text):
                text = best_evidence_text(text, str(source.get("page_content") or ""))
            refs.append(
                {
                    "context_id": context_id,
                    "text": text,
                    "metadata": dict(source.get("metadata") or {}),
                    "anchor_terms": extract_anchor_terms(text),
                }
            )
    if refs:
        return refs
    first_context = next(iter(context_by_id.values()))
    text = str(first_context.get("page_content") or "")[:360].strip()
    return [
        {
            "context_id": str(first_context["context_id"]),
            "text": text,
            "metadata": dict(first_context.get("metadata") or {}),
            "anchor_terms": extract_anchor_terms(text),
        }
    ]


def normalize_answer_points(
    raw_points: object,
    refs: list[dict[str, Any]],
    reference_answer: str,
) -> list[dict[str, Any]]:
    default_context_ids = [str(ref["context_id"]) for ref in refs]
    points: list[dict[str, Any]] = []
    if isinstance(raw_points, list):
        for item in raw_points[:4]:
            if isinstance(item, dict):
                statement = str(item.get("statement") or item.get("text") or "").strip()
                context_ids = [
                    str(value)
                    for value in item.get("context_ids", [])
                    if str(value) in default_context_ids
                ]
            else:
                statement = str(item).strip()
                context_ids = []
            if not statement or is_generic_answer_point(statement):
                continue
            points.append(
                {
                    "point_id": f"p{len(points) + 1}",
                    "statement": statement,
                    "required": True,
                    "context_ids": context_ids or default_context_ids,
                }
            )
    if points:
        return points
    for statement in split_answer_to_points(reference_answer):
        if is_generic_answer_point(statement):
            continue
        points.append(
            {
                "point_id": f"p{len(points) + 1}",
                "statement": statement,
                "required": True,
                "context_ids": default_context_ids,
            }
        )
        if len(points) >= 4:
            break
    return points


def split_answer_to_points(answer: str) -> list[str]:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return []
    pieces = [
        item.strip(" ；;。")
        for item in re.split(r"[。；;]\s*", cleaned)
        if item.strip(" ；;。")
    ]
    if not pieces:
        pieces = [cleaned]
    return [re.sub(r"^[（(]?\d+[）).、]\s*", "", item).strip() for item in pieces]


def normalize_filters(
    raw_filters: object,
    refs: list[dict[str, Any]],
) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    if isinstance(raw_filters, dict):
        for key in ("jurisdiction", "policy_domain", "content_type"):
            values = raw_filters.get(key)
            if isinstance(values, list):
                cleaned = sorted({str(value) for value in values if str(value)})
                if cleaned:
                    filters[key] = cleaned
    if filters:
        return filters
    pseudo_refs = [
        {
            "node_id": ref.get("metadata", {}).get("node_id"),
            "metadata": ref.get("metadata") or {},
        }
        for ref in refs
    ]
    return {
        key: sorted(
            {
                str(ref["metadata"].get(key) or "")
                for ref in pseudo_refs
                if str(ref["metadata"].get(key) or "")
            }
        )
        for key in ("jurisdiction", "policy_domain", "content_type")
        if any(str(ref["metadata"].get(key) or "") for ref in pseudo_refs)
    }


def normalize_question_type(value: object, *, bundle_type: str = "") -> str:
    allowed = {"single_hop", "multi_hop", "table_lookup", "policy_reasoning"}
    text = str(value or "").strip()
    if text in allowed:
        return text
    if bundle_type == "table_lookup":
        return "table_lookup"
    if bundle_type in {"cross_doc_policy_combo", "case_like_policy_query"}:
        return "multi_hop"
    if bundle_type == "same_doc_multi_clause":
        return "policy_reasoning"
    return "single_hop"


def deterministic_candidate(bundle: dict[str, Any], index: int) -> dict[str, Any]:
    context = bundle["contexts"][0]
    metadata = context["metadata"]
    text = str(context["page_content"] or "").strip()
    anchors = extract_anchor_terms(text)
    topic = "、".join(anchors[:2]) or str(metadata.get("title") or "该政策")
    question = f"{topic}在医保审核中应如何核验？"
    point = f"需要核验与{topic}相关的政策正文或表格字段。"
    refs = [
        {
            "context_id": context["context_id"],
            "text": text[:360],
            "metadata": metadata,
            "anchor_terms": anchors,
        }
    ]
    candidate_id = f"cand_{index:06d}"
    return {
        "schema_version": SCHEMA_CANDIDATE_V2,
        "candidate_id": candidate_id,
        "source_bundle_id": bundle["bundle_id"],
        "bundle_type": str(bundle.get("bundle_type") or "single_clause"),
        "source_context_count": len(bundle.get("contexts") or []),
        "question": question,
        "question_type": normalize_question_type(None, bundle_type=str(bundle.get("bundle_type") or "")),
        "reference_answer": point,
        "answer_points": [
            {
                "point_id": "p1",
                "statement": point,
                "required": True,
                "context_ids": [context["context_id"]],
            }
        ],
        "reference_contexts": refs,
        "proposed_filters": {
            key: [str(metadata[key])]
            for key in ("jurisdiction", "policy_domain", "content_type")
            if metadata.get(key)
        },
        "generated_by": "dry_run",
        "generation_model": "dry-run",
        "ragas_compatibility": {
            "document_shape": "langchain_document_compatible",
            "uses_native_ragas_testset_generator": False,
        },
        "debug_id": stable_id("dry", candidate_id, question),
    }


if __name__ == "__main__":
    try:
        DEFAULT_TIKTOKEN_CACHE.mkdir(parents=True, exist_ok=True)
        main()
    except Exception as exc:
        print(f"[v2-candidates] failed: {exc}", file=sys.stderr)
        raise
