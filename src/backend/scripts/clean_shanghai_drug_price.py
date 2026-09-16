"""Clean Shanghai drug product price raw JSONL into reference rows.

This stops before chunking, TextNode conversion, embedding, or FAISS rebuild.
The clean rows are prepared to later become row-level RAG chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "rag_ready"
    / "reference"
    / "drug_product_price_reference"
)
CLEAN_SCHEMA_VERSION = "shanghai_drug_product_price_reference_clean_v0.1"
SOURCE_PLATFORM = "国家医保开放平台上海专区"
SOURCE_URL = "https://bjxt.smiic.net.cn/ypcx/?sessionid=#/pages/drugs-query/search"


def parse_args() -> argparse.Namespace:
    snapshot_date = date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / f"shanghai_drug_price_raw_{snapshot_date}.jsonl",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / f"shanghai_drug_price_clean_{snapshot_date}.jsonl",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REFERENCE_DIR / f"shanghai_drug_price_clean_{snapshot_date}_report.json",
    )
    parser.add_argument("--allow-no-price", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = load_jsonl(args.raw_input)
    clean_rows, report = clean_records(raw_rows, allow_no_price=args.allow_no_price)
    write_jsonl(args.clean_output, clean_rows)
    report.update(
        {
            "status": "ok",
            "raw_input": str(args.raw_input),
            "clean_output": str(args.clean_output),
            "report_json": str(args.report_json),
        }
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clean_records(
    records: list[dict[str, Any]],
    *,
    allow_no_price: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_by_key: dict[str, dict[str, Any]] = {}
    issue_counts: Counter[str] = Counter()
    source_keywords: Counter[str] = Counter()
    drug_types: Counter[str] = Counter()

    for record in records:
        raw_row = record.get("raw_row") or {}
        if not isinstance(raw_row, dict):
            issue_counts["missing_raw_row"] += 1
            continue
        clean = normalize_record(record, raw_row)
        if not clean["drug_name"]:
            issue_counts["missing_drug_name"] += 1
            continue
        if not allow_no_price and not any(
            clean[key] is not None
            for key in ("avg_price", "price_range_min", "price_range_max")
        ):
            issue_counts["missing_price"] += 1
            continue
        dedupe_key = dedupe_key_for(clean)
        existing = output_by_key.get(dedupe_key)
        if existing is None or completeness_score(clean) > completeness_score(existing):
            output_by_key[dedupe_key] = clean
        else:
            issue_counts["duplicate_lower_quality"] += 1
        source_keywords[clean["query_keyword"]] += 1
        drug_types[clean["drug_type"] or "unknown"] += 1

    clean_rows = sorted(
        output_by_key.values(),
        key=lambda item: (
            item["drug_name"],
            item["manufacturer"] or "",
            item["specification"] or "",
            item["package_quantity"] or "",
        ),
    )
    report = {
        "input_count": len(records),
        "clean_count": len(clean_rows),
        "dropped_count": len(records) - len(clean_rows),
        "issue_counts": dict(issue_counts),
        "source_keyword_count": len(source_keywords),
        "top_query_keywords": source_keywords.most_common(20),
        "drug_type_counts": dict(drug_types),
        "policy_domain": "drug_product_price_reference",
        "jurisdiction": "shanghai",
        "can_cite_as_policy_basis": False,
    }
    return clean_rows, report


def normalize_record(record: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    drug_name = clean_text(raw_row.get("drugName"))
    registered_name = clean_text(raw_row.get("regName"))
    product_name = clean_text(raw_row.get("drugProdName"))
    manufacturer = clean_text(raw_row.get("prodentpName"))
    specification = clean_text(raw_row.get("drugSpec"))
    dosage_form = clean_text(raw_row.get("drugDosformName"))
    package_unit = clean_text(raw_row.get("pacUnt"))
    package_quantity = clean_text(raw_row.get("pacCnt"))
    drug_code = clean_text(raw_row.get("drugCode"))
    drug_type = clean_text(raw_row.get("drugType"))

    core = {
        "jurisdiction": "shanghai",
        "policy_domain": "drug_product_price_reference",
        "content_type": "table_row",
        "legal_weight": "reference",
        "rag_ingestion_role": "drug_price_reference",
        "can_cite_as_policy_basis": False,
        "source_platform": SOURCE_PLATFORM,
        "source_url": clean_text(record.get("source_url")) or SOURCE_URL,
        "source_service_code": clean_text(record.get("source_service_code")) or "YBDRUG001",
        "query_keyword": clean_text(record.get("query_keyword")),
        "fetched_at": clean_text(record.get("fetched_at")),
        "price_period_label": clean_text(record.get("price_period_label")) or "上周",
        "price_period_start": clean_text(record.get("price_period_start")),
        "price_period_end": clean_text(record.get("price_period_end")),
        "price_region": "上海",
        "drug_code": drug_code,
        "drug_name": drug_name,
        "registered_name": registered_name,
        "trade_name": product_name,
        "manufacturer": manufacturer,
        "specification": specification,
        "dosage_form": dosage_form,
        "package_unit": package_unit,
        "package_quantity": package_quantity,
        "drug_type": drug_type,
        "avg_price": parse_number(raw_row.get("drugAvgPrice")),
        "price_range_min": parse_number(raw_row.get("drugMinPrice")),
        "price_range_max": parse_number(raw_row.get("drugMaxPrice")),
        "price_unit": "元",
        "sales_count": parse_number(raw_row.get("drugCnt")),
        "pharmacy_count": parse_number(raw_row.get("phacCnt")),
        "herbal_effect_category": clean_text(raw_row.get("ecyType")),
        "herbal_common_usage": clean_text(raw_row.get("cnvlUsed")),
    }
    record_id = stable_record_id(core)
    return {
        "schema_version": CLEAN_SCHEMA_VERSION,
        "record_id": record_id,
        **core,
    }


def stable_record_id(record: dict[str, Any]) -> str:
    seed = "|".join(
        str(record.get(key) or "")
        for key in (
            "price_period_start",
            "price_period_end",
            "drug_code",
            "drug_name",
            "registered_name",
            "trade_name",
            "manufacturer",
            "specification",
            "dosage_form",
            "package_unit",
            "package_quantity",
        )
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"drug_price_shanghai::{record.get('price_period_end') or 'unknown'}::{digest}"


def dedupe_key_for(record: dict[str, Any]) -> str:
    keys = (
        "price_period_start",
        "price_period_end",
        "drug_code",
        "drug_name",
        "registered_name",
        "trade_name",
        "manufacturer",
        "specification",
        "dosage_form",
        "package_unit",
        "package_quantity",
    )
    return "|".join(str(record.get(key) or "") for key in keys)


def completeness_score(record: dict[str, Any]) -> int:
    return sum(1 for value in record.values() if value not in (None, ""))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


if __name__ == "__main__":
    main()
