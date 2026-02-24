#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_util.mcq.mcq_dataloader import (  # noqa: E402
    DATASET_SOURCE_MAP,
    DEFAULT_DATA_ROOT,
    DEFAULT_TEMPLATE_ROOT,
    MCQDataLoader,
    SUPPORTED_DATASETS,
)
from vllm_servers.model_registry import list_model_aliases  # noqa: E402


def _pick_free_port(host: str) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


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


def _index_to_label(index: int) -> str:
    # Excel-style labels: A..Z, AA..AZ, BA...
    value = index + 1
    chars: list[str] = []
    while value > 0:
        value, rem = divmod(value - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def _answer_to_label(answer: Any, num_options: int) -> str | None:
    if answer is None:
        return None

    if isinstance(answer, int):
        idx = answer
        if 0 <= idx < num_options:
            return _index_to_label(idx)
        return None

    if isinstance(answer, str):
        text = answer.strip()
        if not text:
            return None

        # Some datasets store numeric indices as strings.
        try:
            idx = int(text)
        except ValueError:
            idx = None

        if idx is not None and 0 <= idx < num_options:
            return _index_to_label(idx)

        # Assume it's already a label.
        return text.upper()

    return str(answer).strip().upper() or None


def _count_lines(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _format_seconds(value: float) -> str:
    if value < 0:
        value = 0.0
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(value), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an MCQ dataset using the vllm_mcq /rank endpoint for a single model.\n"
            "Writes a JSON report under results/mcq/<dataset>/ with model name + timestamp."
        )
    )
    parser.add_argument("--model-alias", required=True, choices=list_model_aliases())
    parser.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of samples (default: -1 means full dataset).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N samples (default: 10). Set <=0 to disable.",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for vLLM server (default: 0 means auto-pick a free port).",
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
    parser.add_argument("--health-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--terminate-timeout", type=int, default=30)
    parser.add_argument("--drain-timeout", type=int, default=45)
    parser.add_argument("--cooldown", type=int, default=15)

    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "mcq",
        help="Root folder for JSON results (default: results/mcq).",
    )
    args = parser.parse_args()

    port = args.port or _pick_free_port(args.host)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_dataset_name = DATASET_SOURCE_MAP.get(args.dataset, args.dataset)
    samples_path = args.data_root / source_dataset_name / "samples.jsonl"
    total_lines = _count_lines(samples_path)
    if total_lines is not None:
        planned_total = total_lines if args.limit < 0 else min(int(args.limit), int(total_lines))
    else:
        planned_total = None

    dataset_out_dir = args.results_root / args.dataset
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = dataset_out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    out_path = dataset_out_dir / f"{args.model_alias}_{timestamp}.json"
    log_path = logs_dir / f"{args.model_alias}_{timestamp}.log"

    cmd = [
        sys.executable,
        "-m",
        "vllm_servers.vllm_mcq",
        "--model-alias",
        args.model_alias,
        "--host",
        args.host,
        "--port",
        str(port),
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
    ]

    print(f"eval_mcq: model={args.model_alias} dataset={args.dataset} samples={args.limit}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_path}")
    print(f"Output JSON: {out_path}")

    started_at = datetime.now(timezone.utc)

    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        health_payload: dict[str, Any] | None = None
        items: list[dict[str, Any]] = []
        correct = 0
        failed = 0
        total = 0

        try:
            health_url = f"http://{args.host}:{port}/health"
            print("Waiting for /health ...")
            ok, detail = _wait_for_health(health_url, process, timeout_s=args.health_timeout)
            if not ok:
                raise SystemExit(f"Server failed health check: {detail}")

            try:
                with urllib.request.urlopen(health_url, timeout=10) as response:
                    health_payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                health_payload = None

            loader = MCQDataLoader(
                dataset_name=args.dataset,
                data_root=args.data_root,
                template_root=args.template_root,
                limit=None if args.limit < 0 else args.limit,
            )

            rank_url = f"http://{args.host}:{port}/rank"
            eval_started_s = time.time()
            for idx, sample in enumerate(loader):
                total += 1
                raw = sample.get("raw") or {}
                options = _parse_options(raw.get("options"))
                if not options:
                    items.append(
                        {
                            "id": sample.get("id"),
                            "error": "no options found",
                        }
                    )
                    failed += 1
                    continue

                image_path = sample.get("image_path")
                if not isinstance(image_path, Path):
                    image_rel = str(sample.get("image") or "").strip()
                    image_path = args.data_root / args.dataset / image_rel

                if not image_path.is_file():
                    items.append(
                        {
                            "id": sample.get("id"),
                            "error": f"image not found: {image_path}",
                        }
                    )
                    failed += 1
                    continue

                answer_raw = sample.get("answer")
                answer_label = _answer_to_label(answer_raw, num_options=len(options))

                payload = {
                    "text": str(sample.get("prompt") or ""),
                    "options": options,
                    "image_path": str(image_path),
                    "opt_mode": args.opt_mode,
                }

                try:
                    response = _post_json(rank_url, payload=payload, timeout_s=args.request_timeout)
                except Exception as exc:
                    items.append(
                        {
                            "id": sample.get("id"),
                            "answer": {"raw": answer_raw, "label": answer_label},
                            "error": f"request failed: {exc}",
                        }
                    )
                    failed += 1
                    continue

                predicted = response.get("predicted") or {}
                predicted_label = str(predicted.get("label") or "").strip().upper()
                predicted_index = predicted.get("index")
                predicted_option = predicted.get("option")

                ranked_in = response.get("ranked")
                ranked_out: list[dict[str, Any]] = []
                option_ranks: dict[str, int] = {}
                if isinstance(ranked_in, list):
                    for rank, row in enumerate(ranked_in, start=1):
                        if not isinstance(row, dict):
                            continue
                        label = str(row.get("label") or "").strip().upper()
                        if label:
                            option_ranks[label] = rank
                        ranked_out.append(
                            {
                                "rank": rank,
                                "index": row.get("index"),
                                "label": label,
                                "option": row.get("option"),
                                "logprob": row.get("logprob"),
                            }
                        )

                is_correct = bool(answer_label) and predicted_label == answer_label
                if is_correct:
                    correct += 1

                items.append(
                        {
                            "id": sample.get("id"),
                            "answer": {"raw": answer_raw, "label": answer_label},
                            "predicted": {
                                "label": predicted_label,
                            "index": predicted_index,
                            "option": predicted_option,
                        },
                        "correct": is_correct,
                        "top_option": predicted_label,
                        "option_ranks": option_ranks,
                        "ranked": ranked_out,
                        }
                    )

                progress_every = int(args.progress_every)
                processed = idx + 1
                should_print = (
                    progress_every > 0
                    and (processed % progress_every == 0)
                    or (planned_total is not None and processed == planned_total)
                )
                if should_print:
                    elapsed = time.time() - eval_started_s
                    scored_so_far = max(total - failed, 0)
                    acc_so_far = (correct / scored_so_far) if scored_so_far else 0.0
                    eta = None
                    if planned_total is not None and processed > 0:
                        eta = elapsed * (planned_total / processed - 1.0)
                    total_str = str(planned_total) if planned_total is not None else "?"
                    eta_str = _format_seconds(eta) if eta is not None else "?"
                    print(
                        f"[{args.dataset}] {processed}/{total_str} "
                        f"correct={correct} scored={scored_so_far} failed={failed} acc={acc_so_far:.4f} "
                        f"elapsed={_format_seconds(elapsed)} eta={eta_str}",
                        flush=True,
                    )

        finally:
            _stop_process(process, terminate_timeout_s=args.terminate_timeout)
            drained = _wait_for_port_close(host=args.host, port=port, timeout_s=args.drain_timeout)
            if not drained:
                print(f"[WARN] port {port} still open after stop timeout ({args.drain_timeout}s).")
            if args.cooldown > 0:
                print(f"Cooling down for {args.cooldown} seconds...")
                time.sleep(args.cooldown)

    finished_at = datetime.now(timezone.utc)
    duration_s = (finished_at - started_at).total_seconds()
    scored = max(total - failed, 0)
    accuracy = (correct / scored) if scored else 0.0

    result = {
        "task": "mcq_rank",
        "dataset": args.dataset,
        "model_alias": args.model_alias,
        "timestamp": timestamp,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": duration_s,
        "server": health_payload,
        "config": {
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "opt_mode": args.opt_mode,
        },
        "metrics": {
            "mean_accuracy": accuracy,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "failed": failed,
            "scored": scored,
        },
        "items": items,
    }

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Done. accuracy={accuracy:.4f} correct={correct} scored={scored} failed={failed} wrote={out_path}"
    )


if __name__ == "__main__":
    main()
