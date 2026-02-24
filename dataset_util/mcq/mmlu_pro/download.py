#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "MMMU/MMMU_Pro"
SUBSET = "vision"
SPLIT = "test"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "mcq" / "mmlu_pro"
OUTPUT_COLUMNS = ["id", "image", "question", "options", "answer"]
COLUMN_CREATION = {
    "id": "Copied from source column 'id'.",
    "image": "Saved from source column 'image' into images/<safe_id>.<ext>; stored as relative path.",
    "question": "Set to empty string for every sampled row.",
    "options": "Parsed from source column 'options' (list or stringified list) and kept only when length is exactly 10.",
    "answer": "Copied from source column 'answer'.",
}


def _parse_options(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _score(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _safe_name(sample_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._-")
    if cleaned:
        return cleaned
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]


def _save_image(image_value: Any, images_dir: Path, sample_id: str) -> str:
    safe_id = _safe_name(sample_id)

    if hasattr(image_value, "save"):
        filename = f"{safe_id}.png"
        image_path = images_dir / filename
        image_value.save(image_path)
        return f"images/{filename}"

    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_source_path = image_value.get("path")

        suffix = ".png"
        if image_source_path:
            parsed_suffix = Path(str(image_source_path)).suffix.lower()
            if parsed_suffix and len(parsed_suffix) <= 8:
                suffix = parsed_suffix

        filename = f"{safe_id}{suffix}"
        image_path = images_dir / filename

        if isinstance(image_bytes, (bytes, bytearray)):
            image_path.write_bytes(bytes(image_bytes))
            return f"images/{filename}"

        if isinstance(image_source_path, str) and Path(image_source_path).is_file():
            shutil.copy2(image_source_path, image_path)
            return f"images/{filename}"

        if image_source_path:
            return str(image_source_path)

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample MMMU_Pro vision/test rows with exactly 10 options."
    )
    parser.add_argument("-n", "--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    expected_columns: tuple[str, ...] | None = None
    source_columns_mismatch_rows = 0
    first_source_column_mismatch_row: int | None = None
    candidates: list[tuple[int, str, Any, Any, list[Any]]] = []

    dataset = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
    for example in dataset:
        total_rows += 1
        row_columns = tuple(example.keys())
        if expected_columns is None:
            expected_columns = row_columns
        elif row_columns != expected_columns:
            source_columns_mismatch_rows += 1
            if first_source_column_mismatch_row is None:
                first_source_column_mismatch_row = total_rows - 1

        options = _parse_options(example.get("options"))
        if options is None or len(options) != 10:
            continue
        sample_id_raw = example.get("id")
        sample_id = str(sample_id_raw)
        candidates.append(
            (
                _score(sample_id, args.seed),
                sample_id,
                sample_id_raw,
                example.get("answer"),
                options,
            )
        )

    if expected_columns is None:
        raise RuntimeError("Dataset split is empty.")

    if not candidates:
        raise RuntimeError("No rows found with exactly 10 options.")

    selected_count = min(args.num_samples, len(candidates))
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:selected_count]
    selected_ids = {item[1] for item in selected}

    image_by_id: dict[str, str] = {}
    dataset_for_images = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
    for example in dataset_for_images:
        sample_id = str(example.get("id"))
        if sample_id not in selected_ids or sample_id in image_by_id:
            continue
        options = _parse_options(example.get("options"))
        if options is None or len(options) != 10:
            continue
        image_by_id[sample_id] = _save_image(example.get("image"), images_dir, sample_id)
        if len(image_by_id) == len(selected_ids):
            break

    missing_image_ids = [
        sample_id for _, sample_id, _, _, _ in selected if sample_id not in image_by_id
    ]
    if missing_image_ids:
        raise RuntimeError(
            f"Could not collect image values for {len(missing_image_ids)} sampled rows."
        )

    output_file = output_dir / "samples.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for _, sample_id, sample_id_raw, answer, options in selected:
            row = {
                "id": sample_id_raw,
                "image": image_by_id[sample_id],
                "question": "",
                "answer": answer,
                "options": options,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    metadata = {
        "dataset_id": DATASET_ID,
        "subset": SUBSET,
        "split": SPLIT,
        "num_rows_scanned": total_rows,
        "columns": OUTPUT_COLUMNS,
        "source_columns": list(expected_columns),
        "source_columns_consistent": source_columns_mismatch_rows == 0,
        "source_columns_mismatch_rows": source_columns_mismatch_rows,
        "first_source_column_mismatch_row": first_source_column_mismatch_row,
        "rows_meeting_filter": len(candidates),
        "requested_samples": args.num_samples,
        "saved_samples": selected_count,
        "seed": args.seed,
        "filter_criteria": {
            "options_count": 10,
        },
        "column_creation": COLUMN_CREATION,
        "output_file": str(output_file),
        "images_dir": str(images_dir),
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Total rows scanned: {total_rows}")
    print(f"Rows with exactly 10 options: {len(candidates)}")
    print(f"Requested samples: {args.num_samples}")
    print(f"Saved samples: {selected_count}")
    print(f"Saved file: {output_file}")
    print(f"Metadata file: {metadata_path}")
    print(f"Images dir: {images_dir}")


if __name__ == "__main__":
    main()
