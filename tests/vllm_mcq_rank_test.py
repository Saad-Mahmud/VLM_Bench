#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_util.mcq.mcq_dataloader import (
    DEFAULT_DATA_ROOT,
    DEFAULT_TEMPLATE_ROOT,
    MCQDataLoader,
    SUPPORTED_DATASETS,
)
from vllm_servers.model_registry import list_model_aliases


def _wait_for_health(url: str, process: subprocess.Popen[bytes], timeout_s: int) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    last_error = "health endpoint not reachable yet"

    while time.time() < deadline:
        if process.poll() is not None:
            return False, f"process exited early with code {process.returncode}"

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)
                if response.status == 200 and payload.get("status") == "ok":
                    return True, "ok"
                last_error = f"unexpected health response: {payload}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)

    return False, f"timed out waiting for health: {last_error}"


def _stop_process(process: subprocess.Popen[bytes], terminate_timeout_s: int) -> None:
    # vLLM launches worker subprocesses. Always signal the whole process group.
    pgid = None
    if process.pid > 0:
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = None

    if pgid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        try:
            process.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            if pgid is not None and hasattr(os, "killpg"):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def _wait_for_port_close(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.75)
        try:
            status = sock.connect_ex((host, port))
        finally:
            sock.close()
        if status != 0:
            return True
        time.sleep(1)
    return False


def _parse_options(raw_options: Any) -> list[str]:
    if isinstance(raw_options, list):
        return [str(x) for x in raw_options]
    if isinstance(raw_options, str):
        text = raw_options.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def _load_one_sample_per_dataset(
    data_root: Path,
    template_root: Path,
) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}

    for dataset_name in SUPPORTED_DATASETS:
        loader = MCQDataLoader(
            dataset_name=dataset_name,
            data_root=data_root,
            template_root=template_root,
            limit=1,
        )
        item = next(iter(loader), None)
        if item is None:
            raise RuntimeError(f"No sample available for dataset: {dataset_name}")

        options = _parse_options(item["raw"].get("options"))
        if not options:
            raise RuntimeError(f"No valid options found for dataset: {dataset_name}")

        image_path = item.get("image_path")
        if not isinstance(image_path, Path):
            image_rel = str(item["image"])
            image_path = data_root / dataset_name / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path not found for {dataset_name}: {image_path}")

        samples[dataset_name] = {
            "id": item["id"],
            "answer": item["answer"],
            "text": item["prompt"],
            "options": options,
            "image_path": image_path.resolve(),
        }

    return samples


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
        msg = f"HTTP {exc.code}"
        if response_body:
            msg = f"{msg}: {response_body}"
        raise RuntimeError(msg) from exc


def _validate_rank_response(response: dict[str, Any], expected_options: int) -> list[str]:
    errors: list[str] = []

    scores = response.get("scores")
    ranked = response.get("ranked")
    predicted = response.get("predicted")

    if not isinstance(scores, list):
        errors.append("scores is not a list")
        return errors
    if not isinstance(ranked, list):
        errors.append("ranked is not a list")
        return errors
    if not isinstance(predicted, dict):
        errors.append("predicted is not an object")
        return errors

    if len(scores) != expected_options:
        errors.append(f"scores length mismatch: got {len(scores)}, expected {expected_options}")
    if len(ranked) != expected_options:
        errors.append(f"ranked length mismatch: got {len(ranked)}, expected {expected_options}")

    logprobs: list[float] = []
    score_indices: list[int] = []
    for idx, row in enumerate(scores):
        if not isinstance(row, dict):
            errors.append(f"scores[{idx}] is not an object")
            continue
        if row.get("logprob") is None:
            errors.append(f"scores[{idx}].logprob is null")
            continue
        try:
            lp = float(row["logprob"])
        except (TypeError, ValueError):
            errors.append(f"scores[{idx}].logprob is not numeric")
            continue
        if not math.isfinite(lp):
            errors.append(f"scores[{idx}].logprob is not finite")
        logprobs.append(lp)
        try:
            score_indices.append(int(row.get("index")))
        except (TypeError, ValueError):
            errors.append(f"scores[{idx}].index is invalid")

    if logprobs:
        if all(abs(lp) <= 1e-9 for lp in logprobs):
            errors.append("all logprobs are zero or near-zero")
        if max(logprobs) - min(logprobs) <= 1e-9:
            errors.append("all logprobs are identical")

    ranked_logprobs: list[float] = []
    ranked_indices: list[int] = []
    for idx, row in enumerate(ranked):
        if not isinstance(row, dict):
            errors.append(f"ranked[{idx}] is not an object")
            continue
        try:
            ranked_logprobs.append(float(row.get("logprob")))
        except (TypeError, ValueError):
            errors.append(f"ranked[{idx}].logprob is invalid")
            continue
        try:
            ranked_indices.append(int(row.get("index")))
        except (TypeError, ValueError):
            errors.append(f"ranked[{idx}].index is invalid")

    for i in range(1, len(ranked_logprobs)):
        if ranked_logprobs[i - 1] + 1e-9 < ranked_logprobs[i]:
            errors.append("ranked is not sorted by descending logprob")
            break

    if score_indices and ranked_indices and sorted(score_indices) != sorted(ranked_indices):
        errors.append("ranked/scores contain different option indices")

    try:
        predicted_index = int(predicted.get("index"))
    except (TypeError, ValueError):
        errors.append("predicted.index is invalid")
        predicted_index = None

    if ranked and predicted_index is not None:
        top = ranked[0]
        try:
            top_index = int(top.get("index"))
        except (TypeError, ValueError):
            errors.append("ranked[0].index is invalid")
            top_index = None
        if top_index is not None and predicted_index != top_index:
            errors.append("predicted does not match top ranked item")

    return errors


def _run_one_model(
    alias: str,
    host: str,
    port: int,
    max_model_len: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int,
    opt_mode: str,
    health_timeout_s: int,
    request_timeout_s: int,
    terminate_timeout_s: int,
    drain_timeout_s: int,
    cooldown_s: int,
    logs_dir: Path,
    dataset_queries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{alias}.log"

    cmd = [
        sys.executable,
        "-m",
        "vllm_servers.vllm_mcq",
        "--model-alias",
        alias,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--opt-mode",
        opt_mode,
    ]

    print(f"\n=== Testing model: {alias} (port={port}) ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    model_result: dict[str, Any] = {
        "model": alias,
        "started": False,
        "start_error": None,
        "datasets": {},
    }

    with log_file.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            health_url = f"http://{host}:{port}/health"
            ok, detail = _wait_for_health(health_url, process, timeout_s=health_timeout_s)
            if not ok:
                model_result["start_error"] = detail
                print(f"[FAILED] {alias} failed to start: {detail}")
                return model_result

            model_result["started"] = True
            print(f"[OK] {alias} started, sending rank requests...")

            rank_url = f"http://{host}:{port}/rank"
            for dataset_name, query in dataset_queries.items():
                payload = {
                    "text": query["text"],
                    "options": query["options"],
                    "image_path": str(query["image_path"]),
                }
                dataset_result: dict[str, Any] = {
                    "passed": False,
                    "errors": [],
                    "id": query["id"],
                }
                try:
                    response = _post_json(rank_url, payload=payload, timeout_s=request_timeout_s)
                except Exception as exc:
                    err_msg = str(exc)
                    dataset_result["errors"].append(f"request failed: {err_msg}")
                    model_result["datasets"][dataset_name] = dataset_result
                    print(f"  [{dataset_name}] FAIL ({err_msg})")
                    continue

                errors = _validate_rank_response(
                    response=response,
                    expected_options=len(query["options"]),
                )
                dataset_result["errors"] = errors
                dataset_result["passed"] = len(errors) == 0
                model_result["datasets"][dataset_name] = dataset_result
                if dataset_result["passed"]:
                    print(f"  [{dataset_name}] PASS")
                else:
                    print(f"  [{dataset_name}] FAIL ({'; '.join(errors)})")

            return model_result
        finally:
            _stop_process(process, terminate_timeout_s=terminate_timeout_s)
            drained = _wait_for_port_close(host=host, port=port, timeout_s=drain_timeout_s)
            if not drained:
                print(
                    f"[WARN] {alias} port {port} still open after stop timeout "
                    f"({drain_timeout_s}s)."
                )
            if cooldown_s > 0:
                print(f"Cooling down for {cooldown_s} seconds...")
                time.sleep(cooldown_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start each vllm_mcq model alias, run one MCQ sample from each dataset "
            "through /rank, and perform basic log-likelihood ranking sanity checks."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=9109)
    parser.add_argument(
        "--model-aliases",
        nargs="+",
        default=None,
        help=(
            "Optional subset of model aliases to test. "
            "Example: --model-aliases ministral_3_3b qwen3_vl_2b"
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Forwarded to vllm_servers.vllm_mcq --dtype.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
        help="Forwarded to vllm_servers.vllm_mcq --gpu-memory-utilization.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
        help="Forwarded to vllm_servers.vllm_mcq --max-num-batched-tokens.",
    )
    parser.add_argument(
        "--opt-mode",
        default="seq_opt",
        choices=["seq_opt", "parallel_opt", "parallele_opt"],
        help="Forwarded to vllm_servers.vllm_mcq --opt-mode.",
    )
    parser.add_argument("--health-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--cooldown", type=int, default=15)
    parser.add_argument("--terminate-timeout", type=int, default=30)
    parser.add_argument(
        "--drain-timeout",
        type=int,
        default=45,
        help="Seconds to wait for server port to close after stop.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "logs" / "vllm_mcq_rank_test",
    )
    args = parser.parse_args()

    dataset_queries = _load_one_sample_per_dataset(
        data_root=args.data_root,
        template_root=args.template_root,
    )
    available_aliases = list_model_aliases()
    if args.model_aliases:
        requested = [alias.strip() for alias in args.model_aliases if alias.strip()]
        unknown = sorted(set(requested) - set(available_aliases))
        if unknown:
            raise SystemExit(
                "Unknown model alias(es): "
                f"{', '.join(unknown)}. Supported: {', '.join(available_aliases)}"
            )
        # Keep user-provided ordering and de-duplicate.
        seen: set[str] = set()
        aliases = []
        for alias in requested:
            if alias not in seen:
                seen.add(alias)
                aliases.append(alias)
    else:
        aliases = available_aliases
    results: list[dict[str, Any]] = []

    print("vllm_mcq_rank_test")
    print(f"Models to test ({len(aliases)}): {', '.join(aliases)}")
    print(f"Datasets to test ({len(SUPPORTED_DATASETS)}): {', '.join(SUPPORTED_DATASETS)}")

    for idx, alias in enumerate(aliases):
        port = args.base_port + idx
        result = _run_one_model(
            alias=alias,
            host=args.host,
            port=port,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            opt_mode=args.opt_mode,
            health_timeout_s=args.health_timeout,
            request_timeout_s=args.request_timeout,
            terminate_timeout_s=args.terminate_timeout,
            drain_timeout_s=args.drain_timeout,
            cooldown_s=args.cooldown,
            logs_dir=args.logs_dir,
            dataset_queries=dataset_queries,
        )
        results.append(result)

    total_pairs = len(aliases) * len(SUPPORTED_DATASETS)
    passed_pairs = 0
    started_models = 0
    fully_passing_models: list[str] = []

    for result in results:
        if result["started"]:
            started_models += 1
        per_model_pass = result["started"]
        for dataset_name in SUPPORTED_DATASETS:
            ds = result["datasets"].get(dataset_name)
            if ds and ds.get("passed"):
                passed_pairs += 1
            else:
                per_model_pass = False
        if per_model_pass:
            fully_passing_models.append(result["model"])

    all_pass = passed_pairs == total_pairs

    print("\n=== Summary ===")
    print(f"Started models: {started_models}/{len(aliases)}")
    print(f"Dataset checks passed: {passed_pairs}/{total_pairs}")
    print(f"Fully passing models: {fully_passing_models}")
    print(f"ALL PASS: {all_pass}")

    for result in results:
        model = result["model"]
        if not result["started"]:
            print(f"- {model}: START FAIL ({result['start_error']})")
            continue
        fail_msgs: list[str] = []
        for dataset_name in SUPPORTED_DATASETS:
            ds = result["datasets"].get(dataset_name)
            if not ds or not ds.get("passed"):
                errors = ds.get("errors", []) if ds else ["missing result"]
                fail_msgs.append(f"{dataset_name}: {errors}")
        if fail_msgs:
            print(f"- {model}: FAIL ({' | '.join(fail_msgs)})")
        else:
            print(f"- {model}: PASS")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
