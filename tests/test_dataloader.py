#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_util.mcq.mcq_dataloader import (
    DEFAULT_DATA_ROOT,
    DEFAULT_TEMPLATE_ROOT,
    MCQDataLoader,
    SUPPORTED_DATASETS,
)

DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "example.txt"


def dump_one_prompt_per_dataset(
    output_file: Path, data_root: Path, template_root: Path
) -> None:
    lines: list[str] = []

    for dataset_name in SUPPORTED_DATASETS:
        loader = MCQDataLoader(
            dataset_name=dataset_name,
            data_root=data_root,
            template_root=template_root,
            limit=1,
        )
        iterator = iter(loader)
        try:
            item = next(iterator)
        except StopIteration:
            lines.append(f"=== {dataset_name} ===")
            lines.append("No sample found.")
            lines.append("")
            continue

        lines.append(f"=== {dataset_name} ===")
        lines.append(f"id: {item['id']}")
        lines.append(f"image: {item['image']}")
        lines.append(f"answer: {item['answer']}")
        lines.append("prompt:")
        lines.append(item["prompt"].rstrip("\n"))
        lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump one templated prompt from each MCQ dataset into a text file."
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    args = parser.parse_args()

    dump_one_prompt_per_dataset(
        output_file=args.output_file,
        data_root=args.data_root,
        template_root=args.template_root,
    )
    print(f"Wrote examples to: {args.output_file}")


if __name__ == "__main__":
    main()
