#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_ID = "GeorgeBredis/ERQA"
SPLIT = "train"

CHOICE_BLOCK_RE = re.compile(r"\bchoices\s*:\s*", re.IGNORECASE)
CHOICE_LABEL_RE = re.compile(r"\b([A-D])\.\s*")


def _extract_image_count(images_value: Any) -> int | None:
    if not isinstance(images_value, list):
        return None
    return len(images_value)


def _parse_question_and_choices(text: Any) -> list[str] | None:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    match = CHOICE_BLOCK_RE.search(raw)
    if not match:
        return None

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

    return options


def main() -> None:
    dataset = load_dataset(DATASET_ID, split=SPLIT, streaming=True)

    column_names = list(dataset.features.keys()) if getattr(dataset, "features", None) else None
    row_count = 0
    image_count_hist = Counter()
    question_type_hist = Counter()
    answer_hist = Counter()
    options_count_hist = Counter()
    rows_with_parsed_choices = 0
    rows_with_one_image = 0
    rows_with_one_image_and_choices = 0

    for example in dataset:
        row_count += 1
        if column_names is None:
            column_names = list(example.keys())

        image_count = _extract_image_count(example.get("images"))
        if image_count is not None:
            image_count_hist[image_count] += 1
            if image_count == 1:
                rows_with_one_image += 1

        question_type = example.get("question_type")
        if isinstance(question_type, str) and question_type.strip():
            question_type_hist[question_type.strip()] += 1

        answer = example.get("answer")
        if answer is not None:
            answer_hist[str(answer).strip()] += 1

        options = _parse_question_and_choices(example.get("question"))
        if options is not None:
            rows_with_parsed_choices += 1
            options_count_hist[len(options)] += 1
            if image_count == 1:
                rows_with_one_image_and_choices += 1

    if column_names is None:
        column_names = []

    report = {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "num_columns": len(column_names),
        "column_names": column_names,
        "num_rows": row_count,
        "image_count_histogram": {
            str(count): image_count_hist[count] for count in sorted(image_count_hist)
        },
        "rows_with_one_image": rows_with_one_image,
        "rows_with_parsed_choices": rows_with_parsed_choices,
        "rows_with_one_image_and_choices": rows_with_one_image_and_choices,
        "choices_per_question_histogram": {
            str(count): options_count_hist[count] for count in sorted(options_count_hist)
        },
        "question_type_histogram": {
            key: question_type_hist[key]
            for key in sorted(question_type_hist, key=lambda k: (-question_type_hist[k], k))
        },
        "answer_histogram": {
            key: answer_hist[key]
            for key in sorted(answer_hist, key=lambda k: (-answer_hist[k], k))
        },
    }

    out_path = Path(__file__).resolve().parent / "report.json"
    with out_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")

    print(f"Columns ({len(column_names)}): {column_names}")
    print(f"Rows: {row_count}")
    print(f"Rows with exactly one image: {rows_with_one_image}")
    print(f"Rows with parsed choices: {rows_with_parsed_choices}")
    print(f"Rows with one image and parsed choices: {rows_with_one_image_and_choices}")
    print(f"JSON report saved: {out_path}")


if __name__ == "__main__":
    main()
