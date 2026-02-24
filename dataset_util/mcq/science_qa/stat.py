#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "lmms-lab/ScienceQA"
SUBSET = "ScienceQA-IMG"
SPLIT = "test"


def _extract_choice_count(example: dict[str, Any]) -> int | None:
    choices = example.get("choices")
    if isinstance(choices, list):
        return len(choices)
    return None


def main() -> None:
    dataset = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)

    column_names = list(dataset.features.keys()) if getattr(dataset, "features", None) else None
    row_count = 0
    choice_count_hist = Counter()
    rows_with_choices = 0

    for example in dataset:
        row_count += 1
        if column_names is None:
            column_names = list(example.keys())
        choice_count = _extract_choice_count(example)
        if choice_count is None:
            continue
        rows_with_choices += 1
        choice_count_hist[choice_count] += 1

    if column_names is None:
        column_names = []

    min_choices = min(choice_count_hist) if choice_count_hist else None
    max_choices = max(choice_count_hist) if choice_count_hist else None
    avg_choices = (
        sum(count * freq for count, freq in choice_count_hist.items()) / rows_with_choices
        if rows_with_choices
        else None
    )

    histogram = {str(count): choice_count_hist[count] for count in sorted(choice_count_hist)}
    report = {
        "dataset_id": DATASET_ID,
        "subset": SUBSET,
        "split": SPLIT,
        "num_columns": len(column_names),
        "column_names": column_names,
        "num_rows": row_count,
        "rows_with_choices": rows_with_choices,
        "rows_with_list_choices": rows_with_choices,
        "choices_per_query": {
            "min": min_choices,
            "max": max_choices,
            "avg": round(avg_choices, 4) if avg_choices is not None else None,
            "histogram": histogram,
        },
    }

    out_path = Path(__file__).resolve().parent / "report.json"
    with out_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")

    print(f"Columns ({len(column_names)}): {column_names}")
    print(f"Rows: {row_count}")
    if rows_with_choices:
        print(
            "Choices per query: "
            f"min={min_choices}, "
            f"max={max_choices}, "
            f"avg={avg_choices:.4f}, "
            f"histogram={histogram}"
        )
    else:
        print("Choices per query: could not parse 'choices' values.")
    print(f"JSON report saved: {out_path}")


if __name__ == "__main__":
    main()
