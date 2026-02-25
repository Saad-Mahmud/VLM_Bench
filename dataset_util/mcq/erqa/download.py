#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "GeorgeBredis/ERQA"
SPLIT = "train"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "mcq" / "erqa"
OUTPUT_COLUMNS = ["id", "image", "question", "options", "answer"]
COLUMN_CREATION = {
    "id": "Assigned sequential integer from 1..n after deterministic sampling.",
    "image": "Taken from the single element of source column 'images' (row kept only when list length is 1), saved into images/<safe_id>.<ext>, and stored as relative path.",
    "question": "Parsed from source column 'question' by removing the choices block and trailing instructions.",
    "options": "Parsed from source column 'question' by extracting choices labeled A-D.",
    "answer": "Copied from source column 'answer'.",
}

CHOICE_BLOCK_RE = re.compile(r"\bchoices\s*:\s*", re.IGNORECASE)
CHOICE_LABEL_RE = re.compile(r"\b([A-D])\.\s*")


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _extract_single_image(images_value: Any) -> Any | None:
    if not isinstance(images_value, list):
        return None
    if len(images_value) != 1:
        return None
    return images_value[0]


def _parse_question_and_choices(text: Any) -> tuple[str, list[str]] | None:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    match = CHOICE_BLOCK_RE.search(raw)
    if not match:
        return None

    question_text = raw[: match.start()].strip()
    choices_text = raw[match.end() :].strip()

    if not choices_text:
        return None

    choices_text = re.split(r"(?i)\bplease answer\b", choices_text)[0].strip()

    matches = list(CHOICE_LABEL_RE.finditer(choices_text))
    if len(matches) < 4:
        return None

    labels = [m.group(1) for m in matches[:4]]
    if labels != ["A", "B", "C", "D"]:
        return None

    if len(matches) > 4:
        extra_labels = {m.group(1) for m in matches[4:]}
        if extra_labels:
            return None

    options: list[str] = []
    for idx, current in enumerate(matches[:4]):
        start = current.end()
        end = matches[idx + 1].start() if idx < 3 else len(choices_text)
        option_text = choices_text[start:end].strip()
        option_text = option_text.rstrip(" ;")
        options.append(option_text)

    if any(not opt for opt in options):
        return None

    return question_text, options


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

    if isinstance(image_value, str):
        return image_value

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample ERQA rows with exactly one image and parse choices A-D from the question."
        )
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
    mismatched_column_rows = 0
    first_mismatch: dict[str, Any] | None = None

    candidates: list[tuple[int, int, str, Any, list[str]]] = []

    dataset = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    for row_index, example in enumerate(dataset):
        total_rows += 1

        row_columns = tuple(example.keys())
        if expected_columns is None:
            expected_columns = row_columns
        elif row_columns != expected_columns:
            mismatched_column_rows += 1
            if first_mismatch is None:
                first_mismatch = {
                    "row_index": row_index,
                    "expected_columns": list(expected_columns),
                    "actual_columns": list(row_columns),
                }

        if not _is_non_empty(example.get("answer")):
            continue

        single_image = _extract_single_image(example.get("images"))
        if single_image is None:
            continue

        parsed = _parse_question_and_choices(example.get("question"))
        if parsed is None:
            continue

        question_text, options = parsed
        candidates.append(
            (
                _score(row_index, args.seed),
                row_index,
                question_text,
                example.get("answer"),
                options,
            )
        )

    if expected_columns is None:
        raise RuntimeError("Dataset split is empty.")

    if mismatched_column_rows > 0:
        raise RuntimeError(
            "Column mismatch detected while scanning dataset. "
            f"mismatched_rows={mismatched_column_rows}, first_mismatch={first_mismatch}"
        )

    if not candidates:
        raise RuntimeError(
            "No rows found meeting criteria: parsed 4 choices, non-empty answer, exactly one image."
        )

    selected_count = min(args.num_samples, len(candidates))
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:selected_count]
    selected_row_indices = {item[1] for item in selected}

    image_by_row_index: dict[int, str] = {}
    dataset_for_images = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    for row_index, example in enumerate(dataset_for_images):
        if row_index not in selected_row_indices or row_index in image_by_row_index:
            continue

        if not _is_non_empty(example.get("answer")):
            continue
        single_image = _extract_single_image(example.get("images"))
        if single_image is None:
            continue
        parsed = _parse_question_and_choices(example.get("question"))
        if parsed is None:
            continue

        image_by_row_index[row_index] = _save_image(single_image, images_dir, str(row_index))
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
        for output_id, (_, row_index, question_text, answer, options) in enumerate(
            selected, start=1
        ):
            row = {
                "id": output_id,
                "image": image_by_row_index[row_index],
                "question": question_text,
                "options": options,
                "answer": answer,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    metadata = {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "num_rows_scanned": total_rows,
        "columns": OUTPUT_COLUMNS,
        "source_columns": list(expected_columns),
        "source_columns_consistent": mismatched_column_rows == 0,
        "source_columns_mismatch_rows": mismatched_column_rows,
        "first_source_column_mismatch_row": first_mismatch,
        "rows_meeting_filter": len(candidates),
        "requested_samples": args.num_samples,
        "saved_samples": selected_count,
        "seed": args.seed,
        "filter_criteria": {
            "options_count": 4,
            "image_count": 1,
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
    print(f"Rows meeting filter: {len(candidates)}")
    print(f"Requested samples: {args.num_samples}")
    print(f"Saved samples: {selected_count}")
    print(f"Saved file: {output_file}")
    print(f"Metadata file: {metadata_path}")
    print(f"Images dir: {images_dir}")


if __name__ == "__main__":
    main()
