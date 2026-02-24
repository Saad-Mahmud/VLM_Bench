#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "MMMU/MMMU_Pro"
SUBSET = "vision"
SPLIT = "test"


def _extract_option_count(example: dict[str, Any]) -> int | None:
    options = example.get("options")
    if isinstance(options, list):
        return len(options)
    if isinstance(options, str):
        text = options.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, list):
            return len(parsed)
    return None


def main() -> None:
    dataset = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True)

    column_names = list(dataset.features.keys()) if getattr(dataset, "features", None) else None
    row_count = 0
    option_count_hist = Counter()
    rows_with_options = 0

    for example in dataset:
        row_count += 1
        if column_names is None:
            column_names = list(example.keys())
        option_count = _extract_option_count(example)
        if option_count is None:
            continue
        rows_with_options += 1
        option_count_hist[option_count] += 1

    if column_names is None:
        column_names = []

    min_options = min(option_count_hist) if option_count_hist else None
    max_options = max(option_count_hist) if option_count_hist else None
    avg_options = (
        sum(count * freq for count, freq in option_count_hist.items()) / rows_with_options
        if rows_with_options
        else None
    )

    histogram = {str(count): option_count_hist[count] for count in sorted(option_count_hist)}
    report = {
        "dataset_id": DATASET_ID,
        "subset": SUBSET,
        "split": SPLIT,
        "num_columns": len(column_names),
        "column_names": column_names,
        "num_rows": row_count,
        "rows_with_options": rows_with_options,
        "rows_with_list_options": rows_with_options,
        "options_per_query": {
            "min": min_options,
            "max": max_options,
            "avg": round(avg_options, 4) if avg_options is not None else None,
            "histogram": histogram,
        },
    }

    out_path = Path(__file__).resolve().parent / "report.json"
    with out_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")

    print(f"Columns ({len(column_names)}): {column_names}")
    print(f"Rows: {row_count}")
    if rows_with_options:
        print(
            "Options per query: "
            f"min={min_options}, "
            f"max={max_options}, "
            f"avg={avg_options:.4f}, "
            f"histogram={histogram}"
        )
    else:
        print("Options per query: could not parse 'options' values.")
    print(f"JSON report saved: {out_path}")


if __name__ == "__main__":
    main()
