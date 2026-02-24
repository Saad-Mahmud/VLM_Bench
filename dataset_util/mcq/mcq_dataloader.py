#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

try:
    from .prompt_template import (
        DEFAULT_TEMPLATE_ROOT,
        SUPPORTED_DATASETS,
        load_template,
        render_prompt,
    )
except ImportError:
    from prompt_template import (
        DEFAULT_TEMPLATE_ROOT,
        SUPPORTED_DATASETS,
        load_template,
        render_prompt,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "mcq"

# Some datasets are prompt/template variants over the same underlying samples/images.
DATASET_SOURCE_MAP: dict[str, str] = {
    "mmlu_pro_easy": "mmlu_pro",
}


class MCQDataLoader:
    def __init__(
        self,
        dataset_name: str,
        data_root: Path = DEFAULT_DATA_ROOT,
        template_root: Path = DEFAULT_TEMPLATE_ROOT,
        limit: int | None = None,
    ) -> None:
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        self.dataset_name = dataset_name
        self.source_dataset_name = DATASET_SOURCE_MAP.get(dataset_name, dataset_name)
        self.data_root = data_root
        self.dataset_dir = self.data_root / self.source_dataset_name
        self.template_root = template_root
        self.limit = limit
        self.samples_path = self.dataset_dir / "samples.jsonl"

        if not self.samples_path.is_file():
            raise FileNotFoundError(f"Samples file not found: {self.samples_path}")

        self.template_text = load_template(
            dataset_name=self.dataset_name,
            template_root=self.template_root,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yielded = 0
        with self.samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                if self.limit is not None and yielded >= self.limit:
                    break

                sample = json.loads(line)
                prompt = render_prompt(
                    dataset_name=self.dataset_name,
                    sample=sample,
                    template_text=self.template_text,
                    template_root=self.template_root,
                )

                yield {
                    "id": sample.get("id"),
                    "image": sample.get("image"),
                    "image_path": (self.dataset_dir / str(sample.get("image") or "")).resolve(),
                    "data_dir": self.dataset_dir,
                    "answer": sample.get("answer"),
                    "prompt": prompt,
                    "raw": sample,
                }
                yielded += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterate templated MCQ samples for a dataset."
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of samples to print (default: 3). Use -1 for no limit.",
    )
    args = parser.parse_args()

    limit = None if args.limit < 0 else args.limit
    loader = MCQDataLoader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        template_root=args.template_root,
        limit=limit,
    )

    for item in loader:
        print(
            json.dumps(
                {
                    "id": item["id"],
                    "image": item["image"],
                    "answer": item["answer"],
                    "prompt": item["prompt"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
