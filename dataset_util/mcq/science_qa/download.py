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

DATASET_ID = "lmms-lab/ScienceQA"
SUBSET = "ScienceQA-IMG"
SPLIT = "test"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "mcq" / "science_qa"
OUTPUT_COLUMNS = ["id", "image", "question", "options", "answer"]
COLUMN_CREATION = {
    "id": "Assigned sequential integer from 1..n after deterministic sampling.",
    "image": "Saved from source column 'image' into images/<safe_id>.<ext>; stored as relative path.",
    "question": "Copied from source column 'question' (null replaced by empty string).",
    "options": "Parsed from source column 'choices' (list or stringified list), renamed to 'options', and kept only when length is exactly 4.",
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


def _score(row_index: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{row_index}".encode("utf-8")).digest()
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
        description="Sample ScienceQA-IMG/test rows with exactly 4 options."
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
    candidates: list[tuple[int, int, Any, Any, list[Any]]] = []

    dataset = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
    for row_index, example in enumerate(dataset):
        total_rows += 1
        row_columns = tuple(example.keys())
        if expected_columns is None:
            expected_columns = row_columns
        elif row_columns != expected_columns:
            source_columns_mismatch_rows += 1
            if first_source_column_mismatch_row is None:
                first_source_column_mismatch_row = row_index

        options = _parse_options(example.get("choices"))
        if options is None or len(options) != 4:
            continue
        candidates.append(
            (
                _score(row_index, args.seed),
                row_index,
                example.get("question"),
                example.get("answer"),
                options,
            )
        )

    if expected_columns is None:
        raise RuntimeError("Dataset split is empty.")

    if not candidates:
        raise RuntimeError("No rows found with exactly 4 options.")

    selected_count = min(args.num_samples, len(candidates))
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:selected_count]
    selected_row_indices = {item[1] for item in selected}

    image_by_row_index: dict[int, str] = {}
    dataset_for_images = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)
    for row_index, example in enumerate(dataset_for_images):
        if row_index not in selected_row_indices or row_index in image_by_row_index:
            continue
        options = _parse_options(example.get("choices"))
        if options is None or len(options) != 4:
            continue
        image_by_row_index[row_index] = _save_image(
            example.get("image"), images_dir, str(row_index)
        )
        if len(image_by_row_index) == len(selected_row_indices):
            break

    missing_image_rows = [
        row_index for _, row_index, _, _, _ in selected if row_index not in image_by_row_index
    ]
    if missing_image_rows:
        raise RuntimeError(
            f"Could not collect image values for {len(missing_image_rows)} sampled rows."
        )

    output_file = output_dir / "samples.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for output_id, (_, row_index, question, answer, options) in enumerate(selected, start=1):
            row = {
                "id": output_id,
                "image": image_by_row_index[row_index],
                "question": question if question is not None else "",
                "options": options,
                "answer": answer,
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
            "options_count": 4,
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
    print(f"Rows with exactly 4 options: {len(candidates)}")
    print(f"Requested samples: {args.num_samples}")
    print(f"Saved samples: {selected_count}")
    print(f"Saved file: {output_file}")
    print(f"Metadata file: {metadata_path}")
    print(f"Images dir: {images_dir}")


if __name__ == "__main__":
    main()
