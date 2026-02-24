#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_util.mcq.mcq_dataloader import SUPPORTED_DATASETS  # noqa: E402
from vllm_servers.model_registry import list_model_aliases  # noqa: E402

# Matches: <model_alias>_<YYYYMMDDTHHMMSSZ>.json
FILENAME_RE = re.compile(r"^(?P<model>.+)_(?P<ts>\d{8}T\d{6}Z)\.json$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_accuracy(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in ("mean_accuracy", "accuracy"):
            value = metrics.get(key)
            if isinstance(value, (float, int)):
                return float(value)

        correct = metrics.get("correct")
        scored = metrics.get("scored")
        if isinstance(correct, int) and isinstance(scored, int) and scored > 0:
            return float(correct) / float(scored)
    return None


def _extract_counts(payload: dict[str, Any]) -> tuple[int, int] | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    correct = metrics.get("correct")
    scored = metrics.get("scored")
    if isinstance(correct, int) and isinstance(scored, int):
        return correct, scored
    return None


def _find_latest_per_model(dataset_dir: Path) -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in dataset_dir.glob("*.json"):
        match = FILENAME_RE.match(path.name)
        if match is None:
            continue
        model = match.group("model")
        ts = match.group("ts")
        prev = latest.get(model)
        if prev is None or ts > prev[0]:
            latest[model] = (ts, path)
    return {model: entry[1] for model, entry in latest.items()}


def _plot_accuracy_bar(
    *,
    title: str,
    rows: list[tuple[str, float]],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install it, e.g. `pip install matplotlib`."
        ) from exc

    if not rows:
        raise ValueError("No rows to plot.")

    rows_sorted = sorted(rows, key=lambda item: item[1], reverse=True)
    models = [m for m, _ in rows_sorted]
    accs = [a for _, a in rows_sorted]

    height = max(3.0, 0.45 * len(models))
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(models, accs, color="#2f6f9f")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, acc in zip(bars, accs):
        x = float(acc)
        y = float(bar.get_y() + bar.get_height() / 2)
        ax.text(min(x + 0.01, 0.99), y, f"{acc:.3f}", va="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-dataset and combined MCQ accuracy histograms from eval outputs.\n"
            "Uses the latest timestamped JSON per model for each dataset."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "mcq",
        help="Root folder containing per-dataset eval JSON files (default: results/mcq).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "figs" / "mcq",
        help="Output folder for figures (default: analysis/figs/mcq).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model aliases to include (default: all registered).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to include (default: all supported with available results).",
    )
    args = parser.parse_args()

    if not args.results_root.is_dir():
        raise SystemExit(f"Results root not found: {args.results_root}")

    requested_models = set(args.models or list_model_aliases())

    if args.datasets:
        datasets = [d.strip() for d in args.datasets if d.strip()]
    else:
        datasets = [d for d in SUPPORTED_DATASETS if (args.results_root / d).is_dir()]

    if not datasets:
        raise SystemExit(f"No dataset result folders found under: {args.results_root}")

    per_dataset_rows: dict[str, list[tuple[str, float]]] = {}
    combined_counts: dict[str, dict[str, int]] = {}

    for dataset in datasets:
        dataset_dir = args.results_root / dataset
        if not dataset_dir.is_dir():
            print(f"[WARN] dataset folder missing, skipping: {dataset_dir}")
            continue

        latest = _find_latest_per_model(dataset_dir)
        rows: list[tuple[str, float]] = []

        for model, path in sorted(latest.items()):
            if model not in requested_models:
                continue

            payload = _load_json(path)
            acc = _extract_accuracy(payload)
            if acc is None:
                print(f"[WARN] missing accuracy in {path}")
                continue
            rows.append((model, float(acc)))

            counts = _extract_counts(payload)
            if counts is not None:
                correct, scored = counts
                entry = combined_counts.setdefault(model, {"correct": 0, "scored": 0})
                entry["correct"] += int(correct)
                entry["scored"] += int(scored)

        if rows:
            per_dataset_rows[dataset] = rows
        else:
            print(f"[WARN] no usable results for dataset={dataset}")

    # Per-dataset plots
    for dataset, rows in per_dataset_rows.items():
        out_path = args.out_dir / f"{dataset}_accuracy_hist.png"
        _plot_accuracy_bar(
            title=f"{dataset} accuracy (latest per model)",
            rows=rows,
            out_path=out_path,
        )
        print(f"Wrote: {out_path}")

    # Combined plot (weighted by scored counts)
    combined_rows: list[tuple[str, float]] = []
    for model, counts in sorted(combined_counts.items()):
        if model not in requested_models:
            continue
        scored = int(counts.get("scored", 0))
        if scored <= 0:
            continue
        correct = int(counts.get("correct", 0))
        combined_rows.append((model, correct / scored))

    if combined_rows:
        out_path = args.out_dir / "combined_accuracy_hist.png"
        _plot_accuracy_bar(
            title="Combined accuracy across datasets (latest per model/dataset, weighted)",
            rows=combined_rows,
            out_path=out_path,
        )
        print(f"Wrote: {out_path}")
    else:
        print("[WARN] no combined results to plot.")


if __name__ == "__main__":
    main()
