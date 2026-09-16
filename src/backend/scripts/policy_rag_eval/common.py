"""Shared helpers for Policy RAG v2 goldset construction."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_MODEL_CACHE,
    DEFAULT_NODES_PATH,
)


DEFAULT_EVAL_ROOT = DEFAULT_CORPUS_ROOT / "eval"
DEFAULT_REPORT_ROOT = DEFAULT_CORPUS_ROOT / "reports"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.local"
DEFAULT_TIKTOKEN_CACHE = PROJECT_ROOT / "runtime" / "tiktoken_cache"

DEFAULT_V2_CANDIDATES_PATH = (
    DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_candidates.jsonl"
)
DEFAULT_V2_MAPPED_PATH = DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_mapped.jsonl"
DEFAULT_V2_DRAFT_PATH = DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_draft.jsonl"
DEFAULT_V2_AUDITED_PATH = DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_audited.jsonl"
DEFAULT_V2_AUDIT_REPORT_PATH = (
    DEFAULT_REPORT_ROOT / "policy_rag_eval_set_v2_audit_report.md"
)

SCHEMA_CANDIDATE_V2 = "policy_rag_eval_candidate_v2"
SCHEMA_MAPPED_V2 = "policy_rag_eval_candidate_mapped_v2"
SCHEMA_EVAL_V2 = "policy_rag_eval_v2"
SCHEMA_EVAL_V2_2 = "policy_rag_eval_v2_2"

VERSIONED_JSONL_ARTIFACTS = {
    "candidates",
    "mapped",
    "draft",
    "audited",
}

GENERIC_ANSWER_POINT_PATTERNS = [
    "该问题需要根据候选上下文中的政策事实作答",
    "根据上下文可知",
    "根据材料可知",
    "按相关规定执行",
    "详见上述政策",
    "需要结合上下文",
    "需要根据上下文",
    "候选上下文",
    "政策事实作答",
]


KNOWN_ANCHOR_TERMS = [
    "异地就医",
    "跨省",
    "备案",
    "补办备案",
    "急诊",
    "急诊抢救",
    "急诊留观",
    "直接结算",
    "手工报销",
    "零星报销",
    "门诊",
    "门急诊",
    "普通门诊",
    "退休人员",
    "在职职工",
    "起付标准",
    "支付比例",
    "最高支付限额",
    "封顶线",
    "药品目录",
    "甲类",
    "乙类",
    "限定支付范围",
    "诊疗项目",
    "医疗服务项目",
    "CT",
    "胸部CT",
    "计价单位",
    "价格",
    "医用耗材",
    "定点医疗机构",
    "定点零售药店",
    "基金支付",
    "基金监管",
    "重复收费",
    "分解收费",
    "超标准收费",
    "处方",
    "费用清单",
    "收费票据",
    "病历",
    "诊疗证明",
    "慢病",
    "门诊慢特病",
    "长期处方",
    "续方",
]

TABLE_ENTITY_LABEL_PATTERNS = [
    "drug_name",
    "medicine_name",
    "item_name",
    "project_name",
    "service_name",
    "institution_name",
    "pharmacy_name",
    "hospital_name",
    "institution_code",
    "item_code",
    "project_code",
    "medical_service_code",
    "药品名称",
    "药品名",
    "通用名",
    "项目名称",
    "医疗服务项目",
    "医疗服务项目名称",
    "诊疗项目",
    "服务项目",
    "耗材名称",
    "医用耗材",
    "机构名称",
    "医院名称",
    "药店名称",
    "定点医疗机构",
    "定点零售药店",
    "机构编码",
    "项目编码",
    "编码",
]

TABLE_ENTITY_GENERIC_TERMS = {
    "资料标题",
    "地区",
    "政策领域",
    "来源ID",
    "官方来源",
    "source_url",
    "dataset",
    "page",
    "fetched_at",
    "医疗服务项目",
    "床位费等医疗服务项目及价格表",
    "北京医疗服务价格查询",
    "北京定点医疗机构全量分页抓取结果",
}

TABLE_QUERY_GENERIC_PATTERNS = [
    "某北京定点医疗机构",
    "某定点医疗机构",
    "某医疗机构",
    "某北京定点零售药店",
    "某定点零售药店",
    "某药店",
    "某药品",
    "某项目",
    "相关医疗服务项目",
    "床位费相关医疗服务项目",
]

CODE_ENTITY_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]{3,}\b")
LOOSE_CODE_ENTITY_RE = re.compile(r"\b(?:[A-Za-z]{1,}[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*|\d{6,}[A-Za-z]?)\b")
SHORT_MEDICAL_ENTITY_SUFFIXES = (
    "片",
    "丸",
    "散",
    "针",
    "膏",
    "胶囊",
    "颗粒",
    "滴眼剂",
    "注射剂",
    "口服液",
    "起搏器",
    "吻合器",
)
STRICT_TABLE_LOOKUP_DOMAINS = {
    "drug_catalog",
    "medical_service_price",
    "designated_institution",
    "designated_pharmacy",
    "shanghai_payment_scope",
}


@dataclass(frozen=True, slots=True)
class PolicyEvalNode:
    node_id: str
    text: str
    metadata: dict[str, Any]


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs without printing secrets."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if line.lower().startswith("$env:"):
            line = line[5:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_deepseek_config(env_file: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    load_env_file(env_file)
    api_key = first_env(
        "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY",
        "MEDIGUARD_DEEPSEEK_API_KEY",
        "MEDIGUARD_REVIEW_ADVISOR_DEEPSEEK_API_KEY",
        "MEDIGUARD_CASE_AGENT_DEEPSEEK_API_KEY",
    )
    if not api_key:
        raise RuntimeError(
            "DeepSeek API key is missing. Put it in .env.local as "
            "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY or "
            "MEDIGUARD_DEEPSEEK_API_KEY."
        )
    return {
        "api_key": api_key,
        "base_url": first_env(
            "MEDIGUARD_POLICY_RAG_EVAL_BASE_URL",
            "MEDIGUARD_LLM_BASE_URL",
            "MEDIGUARD_REVIEW_ADVISOR_BASE_URL",
            "MEDIGUARD_CASE_AGENT_BASE_URL",
        )
        or "https://api.deepseek.com",
        "model": first_env(
            "MEDIGUARD_POLICY_RAG_EVAL_MODEL",
            "MEDIGUARD_LLM_MODEL",
            "MEDIGUARD_REVIEW_ADVISOR_MODEL",
            "MEDIGUARD_CASE_AGENT_MODEL",
        )
        or "deepseek-v4-flash",
    }


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def normalize_version_suffix(version_suffix: str | None) -> str:
    suffix = str(version_suffix or "v2").strip().lower()
    suffix = re.sub(r"[^a-z0-9_]+", "_", suffix).strip("_")
    return suffix or "v2"


def versioned_eval_path(artifact: str, version_suffix: str | None) -> Path:
    suffix = normalize_version_suffix(version_suffix)
    if artifact in VERSIONED_JSONL_ARTIFACTS:
        return DEFAULT_EVAL_ROOT / f"policy_rag_eval_set_{suffix}_{artifact}.jsonl"
    if artifact == "audit_report":
        return DEFAULT_REPORT_ROOT / f"policy_rag_eval_set_{suffix}_audit_report.md"
    raise ValueError(f"Unknown eval artifact: {artifact}")


def load_policy_nodes(path: Path = DEFAULT_NODES_PATH) -> dict[str, PolicyEvalNode]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    nodes: dict[str, PolicyEvalNode] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            node_id = str(payload.get("id_") or payload.get("node_id") or "").strip()
            text = str(payload.get("text") or "")
            metadata = payload.get("metadata") or {}
            if not node_id or not text.strip() or not isinstance(metadata, dict):
                raise ValueError(f"Invalid policy node at {path}:{line_no}")
            nodes[node_id] = PolicyEvalNode(
                node_id=node_id,
                text=text,
                metadata=dict(metadata),
            )
    if not nodes:
        raise RuntimeError(f"No policy nodes loaded from {path}")
    return nodes


def is_citeable_node(node: PolicyEvalNode) -> bool:
    metadata = node.metadata
    return (
        bool(node.node_id)
        and bool(node.text.strip())
        and bool(metadata.get("source_id"))
        and bool(metadata.get("source_url"))
        and metadata.get("can_cite_as_policy_basis") is True
    )


def citeable_nodes(nodes: dict[str, PolicyEvalNode]) -> list[PolicyEvalNode]:
    return [node for node in nodes.values() if is_citeable_node(node)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            file_obj.write("\n")


def stable_id(prefix: str, *parts: object, width: int = 12) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:width]
    return f"{prefix}_{digest}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def text_contains(container: str, fragment: str) -> bool:
    normalized_fragment = normalize_text(fragment)
    if not normalized_fragment:
        return False
    return normalized_fragment in normalize_text(container)


def is_title_style_question(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    if "《" in text and "》" in text:
        return True
    return bool(re.match(r"^(根据|依据|按照|参照|基于)\s*《", text))


def is_generic_answer_point(statement: str) -> bool:
    text = normalize_text(statement)
    if len(text) < 8:
        return True
    return any(normalize_text(pattern) in text for pattern in GENERIC_ANSWER_POINT_PATTERNS)


def is_table_query_missing_concrete_entity(question: str) -> bool:
    normalized_question = normalize_text(question)
    return any(
        normalize_text(pattern) in normalized_question
        for pattern in TABLE_QUERY_GENERIC_PATTERNS
    )


def is_table_like_node(node: PolicyEvalNode) -> bool:
    metadata = node.metadata
    content_type = str(metadata.get("content_type") or "")
    policy_domain = str(metadata.get("policy_domain") or "")
    doc_type = str(metadata.get("doc_type") or "")
    return (
        (content_type == "table_row" and policy_domain in STRICT_TABLE_LOOKUP_DOMAINS)
        or doc_type
        in {
            "price_table_csv",
            "drug_catalog_rows",
            "designated_institution_rows",
            "designated_pharmacy_rows",
        }
    )


def extract_table_search_entities(
    *,
    question: str,
    evidence_text: str,
    node: PolicyEvalNode,
    anchor_terms: Iterable[str] | None = None,
    limit: int = 12,
) -> list[str]:
    """Extract row-level entities that should appear in a table lookup question."""

    candidates: list[str] = []
    for text in (node.text, evidence_text):
        candidates.extend(_extract_labeled_table_entities(text))
        candidates.extend(CODE_ENTITY_RE.findall(str(text or "")))
        candidates.extend(LOOSE_CODE_ENTITY_RE.findall(str(text or "")))

    for term in anchor_terms or []:
        candidates.extend(_split_table_entity_fragment(str(term or "")))
    candidates.extend(_extract_unlabeled_table_entities(evidence_text))

    question_text = str(question or "")
    filtered: list[str] = []
    for raw_candidate in candidates:
        for candidate in _split_table_entity_fragment(raw_candidate):
            candidate = _clean_table_entity(candidate)
            if not _is_valid_table_entity(candidate):
                continue
            if candidate in filtered:
                continue
            filtered.append(candidate)

    filtered.sort(
        key=lambda item: (
            0 if table_entity_in_question(item, question_text) else 1,
            -len(normalize_text(item)),
            item,
        )
    )
    return filtered[:limit]


def table_entity_in_question(entity: str, question: str) -> bool:
    normalized_entity = normalize_text(entity)
    normalized_question = normalize_text(question)
    if not normalized_entity or not normalized_question:
        return False
    if normalized_entity in normalized_question:
        return True
    if normalized_question in normalized_entity and len(normalized_question) >= 6:
        return True
    if CODE_ENTITY_RE.fullmatch(entity.strip()):
        return normalized_entity in normalized_question
    if "磁共振" in normalized_entity and "MRI" in question.upper():
        return True
    if normalized_entity.upper() in {"CT", "MRI", "MRCP", "MRM", "MRU"}:
        return normalized_entity.upper() in question.upper()

    # A generated evidence fragment can merge a row value with the next field
    # (for example "脑起搏器及配件患者控制器").  Allow a strong prefix match.
    for size in (10, 8, 6):
        if len(normalized_entity) >= size and normalized_entity[:size] in normalized_question:
            return True
        if len(normalized_entity) >= size and normalized_entity[-size:] in normalized_question:
            return True
    return False


def _extract_labeled_table_entities(text: str) -> list[str]:
    entities: list[str] = []
    labeled: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if ":" not in line and "：" not in line:
            continue
        key, value = re.split(r"[:：]", line, maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        normalized_key = key.strip().lower()
        labeled[normalized_key] = value
        if any(pattern.lower() in key.lower() for pattern in TABLE_ENTITY_LABEL_PATTERNS):
            entities.append(value)
    drug_name = labeled.get("drug_name") or labeled.get("药品名称") or labeled.get("药品名")
    dosage_form = labeled.get("dosage_form") or labeled.get("剂型")
    if drug_name and dosage_form:
        entities.append(f"{drug_name}{dosage_form}")
    return entities


def _extract_unlabeled_table_entities(text: str) -> list[str]:
    entities: list[str] = []
    for token in re.split(r"[\s,，;；|]+", str(text or "")):
        cleaned = _clean_table_entity(token)
        if _is_valid_table_entity(cleaned):
            entities.append(cleaned)
    return entities


def _split_table_entity_fragment(value: str) -> list[str]:
    cleaned = _clean_table_entity(value)
    if not cleaned:
        return []
    pieces = [cleaned]
    for splitter in ("(", "（"):
        if splitter in cleaned:
            prefix = cleaned.split(splitter, 1)[0]
            if prefix:
                pieces.append(prefix)
    for splitter in ("患者", "自付", "限", "价格", "单价", "编码", "医保", "支付", "甲", "乙"):
        if splitter in cleaned:
            prefix = cleaned.split(splitter, 1)[0]
            if prefix:
                pieces.append(prefix)
    return pieces


def _clean_table_entity(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^[\s:：,，;；]+|[\s:：,，;；]+$", "", text)
    if CODE_ENTITY_RE.fullmatch(text) or LOOSE_CODE_ENTITY_RE.fullmatch(text):
        return text
    text = re.sub(r"(原件|复印件|纸质|电子|必要|非必要|查看)+$", "", text)
    text = re.sub(r"\d+(\.\d+)?%?$", "", text).strip()
    return text


def _is_valid_table_entity(value: str) -> bool:
    normalized = normalize_text(value)
    if "_" in normalized:
        return False
    if len(normalized) < 4 and not re.search(r"[A-Za-z0-9]", normalized):
        if not normalized.endswith(SHORT_MEDICAL_ENTITY_SUFFIXES):
            return False
    if normalized.lower().startswith(("catalog", "policy", "ragtable", "sourceid")):
        return False
    if normalized in {normalize_text(term) for term in TABLE_ENTITY_GENERIC_TERMS}:
        return False
    if normalized.startswith(("http", "www")):
        return False
    if re.fullmatch(r"\d+(\.\d+)?%?", normalized):
        return False
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}.*", normalized):
        return False
    return True


def statement_supported_by_evidence(
    statement: str,
    evidence_texts: Iterable[str],
) -> bool:
    if is_generic_answer_point(statement):
        return False
    evidence = "\n".join(str(text or "") for text in evidence_texts if str(text or "").strip())
    if not evidence.strip():
        return False
    if text_contains(evidence, statement):
        return True

    statement_terms = set(extract_anchor_terms(statement, limit=8))
    evidence_terms = set(extract_anchor_terms(evidence, limit=16))
    shared_terms = statement_terms & evidence_terms
    if statement_terms and len(shared_terms) >= min(2, len(statement_terms)):
        return True

    statement_ngrams = set(chinese_ngrams(normalize_text(statement), size=2))
    evidence_ngrams = set(chinese_ngrams(normalize_text(evidence), size=2))
    if not statement_ngrams:
        return False
    overlap = len(statement_ngrams & evidence_ngrams)
    coverage = overlap / len(statement_ngrams)
    return coverage >= 0.35 or (overlap >= 12 and coverage >= 0.2)


def split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    buffer = ""
    for char in str(text or ""):
        buffer += char
        if char in "。；;！？!?\n":
            stripped = buffer.strip()
            if stripped:
                pieces.append(stripped)
            buffer = ""
    tail = buffer.strip()
    if tail:
        pieces.append(tail)
    return pieces or [str(text or "").strip()]


def best_evidence_text(reference_text: str, node_text: str, *, max_chars: int = 420) -> str:
    """Return text that is actually present in node_text."""

    reference_text = str(reference_text or "").strip()
    node_text = str(node_text or "").strip()
    if not node_text:
        return ""
    if reference_text and reference_text in node_text:
        return reference_text[:max_chars]

    normalized_reference = normalize_text(reference_text)
    if normalized_reference and normalized_reference in normalize_text(node_text):
        for sentence in split_sentences(node_text):
            if text_contains(sentence, reference_text):
                return sentence[:max_chars]

    reference_terms = set(extract_anchor_terms(reference_text))
    reference_bigrams = set(chinese_ngrams(normalize_text(reference_text), size=2))
    best_score = -1.0
    best_sentence = ""
    for sentence in split_sentences(node_text):
        normalized_sentence = normalize_text(sentence)
        if not normalized_sentence:
            continue
        sentence_terms = set(extract_anchor_terms(sentence))
        sentence_bigrams = set(chinese_ngrams(normalized_sentence, size=2))
        term_score = len(reference_terms & sentence_terms) * 4
        overlap = len(reference_bigrams & sentence_bigrams)
        length_penalty = min(len(normalized_sentence) / 400, 1.5)
        score = term_score + overlap / 20 - length_penalty
        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()
    return (best_sentence or node_text)[:max_chars]


def chinese_ngrams(text: str, *, size: int) -> list[str]:
    if not text or len(text) < size:
        return []
    return [text[index : index + size] for index in range(len(text) - size + 1)]


def extract_anchor_terms(*texts: str, limit: int = 6) -> list[str]:
    joined = "\n".join(str(text or "") for text in texts)
    anchors = [term for term in KNOWN_ANCHOR_TERMS if term in joined]
    if len(anchors) < 2:
        for match in re.finditer(r"[\u4e00-\u9fff]{2,8}", joined):
            token = match.group(0)
            if token not in anchors:
                anchors.append(token)
            if len(anchors) >= limit:
                break
    return anchors[:limit]


def node_document_role(node: PolicyEvalNode) -> str:
    metadata = node.metadata
    content_type = str(metadata.get("content_type") or "")
    policy_domain = str(metadata.get("policy_domain") or "")
    doc_type = str(metadata.get("doc_type") or "")
    if content_type == "table_row":
        if policy_domain == "drug_catalog":
            return "drug_catalog_row"
        if policy_domain == "medical_service_price":
            return "medical_service_price_row"
        if policy_domain == "designated_institution":
            return "designated_institution_row"
        return "table_row"
    if doc_type in {"faq", "service_guide", "policy_interpretation", "policy"}:
        return doc_type
    return "policy_evidence"


def ragas_document_for_node(
    node: PolicyEvalNode,
    *,
    max_context_chars: int,
    context_id: str,
) -> dict[str, Any]:
    metadata = node.metadata
    return {
        "context_id": context_id,
        "page_content": node.text[:max_context_chars],
        "metadata": {
            "node_id": node.node_id,
            "source_id": str(metadata.get("source_id") or ""),
            "doc_id": str(metadata.get("doc_id") or ""),
            "source_url": str(metadata.get("source_url") or ""),
            "title": str(metadata.get("title") or ""),
            "jurisdiction": str(metadata.get("jurisdiction") or ""),
            "policy_domain": str(metadata.get("policy_domain") or ""),
            "doc_type": str(metadata.get("doc_type") or ""),
            "content_type": str(metadata.get("content_type") or ""),
            "document_role": node_document_role(node),
            "can_cite_as_policy_basis": bool(
                metadata.get("can_cite_as_policy_basis")
            ),
        },
    }


def balanced_sample_nodes(
    nodes: list[PolicyEvalNode],
    *,
    count: int,
    seed: int,
) -> list[PolicyEvalNode]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    by_domain: dict[str, list[PolicyEvalNode]] = defaultdict(list)
    for node in nodes:
        domain = str(node.metadata.get("policy_domain") or "unknown")
        by_domain[domain].append(node)
    for bucket in by_domain.values():
        bucket.sort(key=lambda item: item.node_id)

    domains = sorted(by_domain)
    rng.shuffle(domains)
    selected: list[PolicyEvalNode] = []
    position = 0
    while len(selected) < min(count, len(nodes)):
        added = False
        for domain in domains:
            bucket = by_domain[domain]
            if position < len(bucket):
                selected.append(bucket[position])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        position += 1
    if len(selected) < count:
        leftovers = [node for node in nodes if node not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])
    return selected[:count]


def parse_json_object(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last >= first:
        text = text[first : last + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def derive_filters_from_refs(refs: list[dict[str, Any]], nodes: dict[str, PolicyEvalNode]) -> dict[str, list[str]]:
    jurisdictions: set[str] = set()
    domains: set[str] = set()
    content_types: set[str] = set()
    for ref in refs:
        node = nodes.get(str(ref.get("node_id") or ""))
        if node is None:
            continue
        metadata = node.metadata
        if metadata.get("jurisdiction"):
            jurisdictions.add(str(metadata["jurisdiction"]))
        if metadata.get("policy_domain"):
            domains.add(str(metadata["policy_domain"]))
        if metadata.get("content_type"):
            content_types.add(str(metadata["content_type"]))
    filters: dict[str, list[str]] = {}
    if jurisdictions:
        filters["jurisdiction"] = sorted(jurisdictions)
    if domains:
        filters["policy_domain"] = sorted(domains)
    if content_types:
        filters["content_type"] = sorted(content_types)
    return filters
