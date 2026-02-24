#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_util.mcq.mcq_dataloader import SUPPORTED_DATASETS  # noqa: E402
from vllm_servers.model_registry import list_model_aliases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval/eval_mcq.py across multiple models and datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model aliases (default: all registered).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of dataset names (default: all supported).",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument(
        "--opt-mode",
        default="seq_opt",
        choices=["seq_opt", "parallel_opt", "parallele_opt"],
    )
    parser.add_argument("--cooldown", type=int, default=15)
    parser.add_argument("--health-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining pairs if one eval fails.",
    )
    args = parser.parse_args()

    models = args.models or list_model_aliases()
    datasets = args.datasets or list(SUPPORTED_DATASETS)

    eval_script = PROJECT_ROOT / "eval" / "eval_mcq.py"
    if not eval_script.is_file():
        raise SystemExit(f"Missing eval script: {eval_script}")

    failures: list[str] = []
    total = len(models) * len(datasets)
    idx = 0

    for model in models:
        for dataset in datasets:
            idx += 1
            cmd = [
                sys.executable,
                str(eval_script),
                "--model-alias",
                model,
                "--dataset",
                dataset,
                "--max-model-len",
                str(args.max_model_len),
                "--dtype",
                args.dtype,
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--opt-mode",
                args.opt_mode,
                "--cooldown",
                str(args.cooldown),
                "--health-timeout",
                str(args.health_timeout),
                "--request-timeout",
                str(args.request_timeout),
            ]

            print(f"\n[{idx}/{total}] model={model} dataset={dataset}")
            print(f"Command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                msg = f"{model}/{dataset}: exit_code={exc.returncode}"
                failures.append(msg)
                print(f"[FAIL] {msg}")
                if not args.continue_on_error:
                    raise SystemExit(1) from exc

    if failures:
        print("\nFailures:")
        for msg in failures:
            print(f"- {msg}")
        raise SystemExit(1)

    print("\nAll evaluations completed successfully.")


if __name__ == "__main__":
    main()

