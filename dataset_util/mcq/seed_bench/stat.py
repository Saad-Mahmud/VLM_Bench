#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "lmms-lab/SEED-Bench"
SPLIT = "test"
CHOICE_KEYS = ("choice_a", "choice_b", "choice_c", "choice_d")


def _is_non_empty_choice(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _extract_image_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if value is None:
        return None
    return 1


def main() -> None:
    dataset = load_dataset(DATASET_ID, split=SPLIT, streaming=True)

    column_names = list(dataset.features.keys()) if getattr(dataset, "features", None) else None
    row_count = 0

    rows_with_all_4_choices = 0
    choice_presence_hist = Counter()
    missing_counts_by_choice = {key: 0 for key in CHOICE_KEYS}

    rows_with_image_count = 0
    image_count_hist = Counter()

    for example in dataset:
        row_count += 1
        if column_names is None:
            column_names = list(example.keys())

        present_choices = 0
        for key in CHOICE_KEYS:
            if _is_non_empty_choice(example.get(key)):
                present_choices += 1
            else:
                missing_counts_by_choice[key] += 1
        choice_presence_hist[present_choices] += 1
        if present_choices == len(CHOICE_KEYS):
            rows_with_all_4_choices += 1

        image_count = _extract_image_count(example.get("image"))
        if image_count is not None:
            rows_with_image_count += 1
            image_count_hist[image_count] += 1

    if column_names is None:
        column_names = []

    min_images = min(image_count_hist) if image_count_hist else None
    max_images = max(image_count_hist) if image_count_hist else None
    avg_images = (
        sum(count * freq for count, freq in image_count_hist.items()) / rows_with_image_count
        if rows_with_image_count
        else None
    )

    images_stats = {
        "rows_with_image_count": rows_with_image_count,
        "min": min_images,
        "max": max_images,
        "avg": round(avg_images, 4) if avg_images is not None else None,
        "histogram": {str(count): image_count_hist[count] for count in sorted(image_count_hist)},
    }

    report = {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "num_columns": len(column_names),
        "column_names": column_names,
        "num_rows": row_count,
        "choices_check": {
            "choice_keys": list(CHOICE_KEYS),
            "rows_with_all_4_choices": rows_with_all_4_choices,
            "all_rows_have_all_4_choices": rows_with_all_4_choices == row_count,
            "choice_presence_histogram": {
                str(count): choice_presence_hist[count] for count in sorted(choice_presence_hist)
            },
            "missing_counts_by_choice": missing_counts_by_choice,
        },
        "images_per_sample": images_stats,
        "images_per_action": images_stats,
    }

    out_path = Path(__file__).resolve().parent / "report.json"
    with out_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")

    print(f"Columns ({len(column_names)}): {column_names}")
    print(f"Rows: {row_count}")
    print(
        "Choices check: "
        f"rows_with_all_4_choices={rows_with_all_4_choices}, "
        f"all_rows_have_all_4_choices={rows_with_all_4_choices == row_count}"
    )
    if rows_with_image_count:
        print(
            "Images per sample: "
            f"min={min_images}, "
            f"max={max_images}, "
            f"avg={avg_images:.4f}, "
            f"histogram={report['images_per_sample']['histogram']}"
        )
    else:
        print("Images per sample: could not parse 'image' values.")
    print(f"JSON report saved: {out_path}")


if __name__ == "__main__":
    main()
