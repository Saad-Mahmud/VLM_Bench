#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_ROOT = PROJECT_ROOT / "dataset_util" / "mcq"
SUPPORTED_DATASETS = ("mmlu_pro", "mmlu_pro_easy", "science_qa", "seed_bench")


def _parse_options(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def load_template(dataset_name: str, template_root: Path = DEFAULT_TEMPLATE_ROOT) -> str:
    template_path = template_root / dataset_name / "prompt_template.txt"
    if not template_path.is_file():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def render_prompt(
    dataset_name: str,
    sample: dict[str, Any],
    template_text: str | None = None,
    template_root: Path = DEFAULT_TEMPLATE_ROOT,
) -> str:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    if template_text is None:
        template_text = load_template(dataset_name, template_root=template_root)

    question = sample.get("question")
    if question is None:
        question = ""
    else:
        question = str(question)

    fields: dict[str, Any] = {"question": question}

    if dataset_name in {"science_qa", "seed_bench"}:
        options = _parse_options(sample.get("options"))
        while len(options) < 4:
            options.append("")
        fields.update(
            {
                "A": str(options[0]),
                "B": str(options[1]),
                "C": str(options[2]),
                "D": str(options[3]),
            }
        )
    elif dataset_name == "mmlu_pro_easy":
        options = _parse_options(sample.get("options"))
        while len(options) < 10:
            options.append("")
        fields.update(
            {
                "A": str(options[0]),
                "B": str(options[1]),
                "C": str(options[2]),
                "D": str(options[3]),
                "E": str(options[4]),
                "F": str(options[5]),
                "G": str(options[6]),
                "H": str(options[7]),
                "I": str(options[8]),
                "J": str(options[9]),
            }
        )

    try:
        return template_text.format(**fields)
    except KeyError as exc:
        raise ValueError(
            f"Template for {dataset_name} has unknown placeholder: {exc}"
        ) from exc


def _read_jsonl_line(path: Path, line_index: int) -> dict[str, Any]:
    if line_index < 0:
        raise ValueError("--line-index must be >= 0")
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == line_index:
                return json.loads(line)
    raise IndexError(f"line_index={line_index} out of range for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply dataset prompt template to a raw sample row."
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument(
        "--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--raw-query-json",
        help="Raw sample row as a JSON object string.",
    )
    input_group.add_argument(
        "--samples-file",
        type=Path,
        help="Path to samples.jsonl file.",
    )
    parser.add_argument(
        "--line-index",
        type=int,
        default=0,
        help="Zero-based line index in --samples-file (default: 0).",
    )

    args = parser.parse_args()

    if args.raw_query_json is not None:
        sample = json.loads(args.raw_query_json)
    else:
        sample = _read_jsonl_line(args.samples_file, args.line_index)

    prompt = render_prompt(
        dataset_name=args.dataset,
        sample=sample,
        template_root=args.template_root,
    )
    print(prompt)


if __name__ == "__main__":
    main()
