"""Build the Case Agent intent example vector index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.agent.case_agent.intent_examples import (  # noqa: E402
    IntentExample,
    load_intent_seed_examples,
)


DEFAULT_EXAMPLES_PATH = (
    PROJECT_ROOT / "docs" / "02-planning" / "intent_biencoder_prep" / "intent_examples.seed.json"
)
DEFAULT_TAXONOMY_PATH = (
    PROJECT_ROOT / "docs" / "02-planning" / "intent_biencoder_prep" / "intent_taxonomy.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runtime" / "intent_example_index" / "caser_intent_examples_v0.2"
)
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "runtime" / "hf_cache"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Case Agent intent example vector index."
    )
    parser.add_argument("--examples-path", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples_path = args.examples_path.resolve()
    taxonomy_path = args.taxonomy_path.resolve()
    output_dir = args.output_dir.resolve()
    cache_folder = args.cache_folder.resolve()
    examples = load_intent_seed_examples(
        examples_path=examples_path,
        taxonomy_path=taxonomy_path,
    )
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("No intent examples were loaded.")
    _prepare_output_dir(output_dir, force=bool(args.force))

    texts = [example.normalized_text for example in examples]
    encoder = _load_encoder(args.model_name, cache_folder)
    embeddings = encoder.encode(
        texts,
        batch_size=max(1, int(args.batch_size)),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or int(embeddings.shape[0]) != len(examples):
        raise SystemExit("Embedding matrix shape does not match loaded examples.")

    examples_output = output_dir / "examples.jsonl"
    embeddings_output = output_dir / "embeddings.npy"
    manifest_output = output_dir / "manifest.json"
    _write_examples_jsonl(examples_output, examples)
    np.save(embeddings_output, embeddings)
    manifest = _build_manifest(
        examples=examples,
        examples_path=examples_path,
        taxonomy_path=taxonomy_path,
        output_dir=output_dir,
        model_name=str(args.model_name),
        cache_folder=cache_folder,
        embeddings=embeddings,
    )
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _load_encoder(model_name: str, cache_folder: Path) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        cache_folder=str(cache_folder),
        local_files_only=True,
    )


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_files = ("examples.jsonl", "embeddings.npy", "manifest.json")
    existing = [output_dir / name for name in managed_files if (output_dir / name).exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"Intent example index already exists: {names}. Use --force to rebuild.")
    for path in existing:
        path.unlink()


def _write_examples_jsonl(path: Path, examples: list[IntentExample]) -> None:
    counters: dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for example in examples:
            counters[example.intent_id] = counters.get(example.intent_id, 0) + 1
            record = {
                "example_id": f"{example.intent_id}_{counters[example.intent_id]:04d}",
                "intent_id": example.intent_id,
                "text": example.text,
                "normalized_text": example.normalized_text,
                "ability_layer": example.spec.ability_layer,
                "capability_hint": example.spec.capability_hint,
                "answer_shape": example.spec.answer_shape,
                "evidence_need": example.spec.evidence_need,
                "status": example.spec.status,
            }
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_manifest(
    *,
    examples: list[IntentExample],
    examples_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    model_name: str,
    cache_folder: Path,
    embeddings: np.ndarray,
) -> dict[str, Any]:
    seed = json.loads(examples_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return {
        "artifact_version": "caser_intent_example_index_v0.1",
        "source_examples_version": seed.get("version"),
        "source_taxonomy_version": taxonomy.get("version"),
        "source_examples_path": str(examples_path),
        "source_taxonomy_path": str(taxonomy_path),
        "source_examples_sha256": _sha256_file(examples_path),
        "source_taxonomy_sha256": _sha256_file(taxonomy_path),
        "encoder_model": model_name,
        "cache_folder": str(cache_folder),
        "backend": "bge_index",
        "example_count": len(examples),
        "embedding_shape": list(embeddings.shape),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "files": {
            "examples": "examples.jsonl",
            "embeddings": "embeddings.npy",
            "manifest": "manifest.json",
        },
        "thresholds": taxonomy.get("default_thresholds") or {},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
