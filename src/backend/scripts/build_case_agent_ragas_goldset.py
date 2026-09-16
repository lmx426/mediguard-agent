"""Build the minimal Case Agent Native Ragas goldset.

The source dataset contains audit and evidence metadata, but the Ragas
generation metrics only need the question and the reviewed required answer.
This script deliberately emits exactly three fields per row:
``query_id``, ``user_input`` and ``reference``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "eval"
    / "policy_rag_eval_set_v2_2_required_optional_audited.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "eval"
    / "policy_case_agent_ragas_goldset_197.jsonl"
)
EXPECTED_COUNT = 197
EXPECTED_FIELDS = {"query_id", "user_input", "reference"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the three-field Case Agent Native Ragas goldset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def build_goldset(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_COUNT} source rows, received {len(rows)}."
        )

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        query_id = str(row.get("query_id") or "").strip()
        user_input = str(row.get("question") or "").strip()
        reference = str(row.get("reference_answer_required") or "").strip()
        if not query_id:
            raise ValueError(f"Row {index} has an empty query_id.")
        if query_id in seen:
            raise ValueError(f"Duplicate query_id: {query_id}")
        if not user_input:
            raise ValueError(f"Row {query_id} has an empty question.")
        if not reference:
            raise ValueError(f"Row {query_id} has an empty reference_answer_required.")
        seen.add(query_id)
        output.append(
            {
                "query_id": query_id,
                "user_input": user_input,
                "reference": reference,
            }
        )

    if len(output) != EXPECTED_COUNT:
        raise ValueError(f"Goldset count changed unexpectedly: {len(output)}")
    if any(set(row) != EXPECTED_FIELDS for row in output):
        raise ValueError("Goldset rows must contain exactly query_id, user_input, reference.")
    return output


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    goldset = build_goldset(read_jsonl(args.source))
    write_jsonl(args.output, goldset)
    print(
        json.dumps(
            {
                "status": "ok",
                "count": len(goldset),
                "source": str(args.source),
                "output": str(args.output),
                "fields": sorted(EXPECTED_FIELDS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
