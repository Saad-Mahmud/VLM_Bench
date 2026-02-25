#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _run_one_model(
    alias: str,
    host: str,
    port: int,
    max_model_len: int,
    wait_after_start_s: int,
    health_timeout_s: int,
    cooldown_s: int,
    terminate_timeout_s: int,
    drain_timeout_s: int,
    logs_dir: Path,
) -> bool:
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
    ]

    print(f"\n=== Starting model: {alias} (port={port}) ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    with log_file.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        success = False
        try:
            health_url = f"http://{host}:{port}/health"
            ok, detail = _wait_for_health(health_url, process, timeout_s=health_timeout_s)
            if ok:
                success = True
                print(f"[SUCCESS] {alias} health check passed.")
                print(f"Holding server for {wait_after_start_s} seconds...")
                time.sleep(wait_after_start_s)
            else:
                print(f"[FAILED] {alias} did not start correctly: {detail}")
        finally:
            _stop_process(process, terminate_timeout_s=terminate_timeout_s)
            drained = _wait_for_port_close(host=host, port=port, timeout_s=drain_timeout_s)
            if not drained:
                print(
                    f"[WARN] {alias} port {port} still open after stop timeout "
                    f"({drain_timeout_s}s)."
                )
            print(f"Stopped {alias}. Cooling down for {cooldown_s} seconds...")
            time.sleep(cooldown_s)

    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start vllm_mcq server for each registered model alias, keep each running "
            "for 5 minutes after healthy, then stop and wait 1 minute."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=9009)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--health-timeout", type=int, default=300)
    parser.add_argument("--wait-after-start", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=30)
    parser.add_argument("--terminate-timeout", type=int, default=30)
    parser.add_argument(
        "--drain-timeout",
        type=int,
        default=45,
        help="Seconds to wait for server port to close after stop.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "logs" / "vllm_mcq_start_test",
    )
    args = parser.parse_args()

    aliases = list_model_aliases()
    total = len(aliases)
    success_count = 0
    started_aliases: list[str] = []

    print("vllm_mcq_start_test")
    print(f"Model aliases to test ({total}): {', '.join(aliases)}")

    for idx, alias in enumerate(aliases):
        port = args.base_port + idx
        ok = _run_one_model(
            alias=alias,
            host=args.host,
            port=port,
            max_model_len=args.max_model_len,
            wait_after_start_s=args.wait_after_start,
            health_timeout_s=args.health_timeout,
            cooldown_s=args.cooldown,
            terminate_timeout_s=args.terminate_timeout,
            drain_timeout_s=args.drain_timeout,
            logs_dir=args.logs_dir,
        )
        if ok:
            success_count += 1
            started_aliases.append(alias)

    print("\n=== Summary ===")
    print(f"Successfully started: {success_count}/{total}")
    print(f"Started aliases: {started_aliases}")


if __name__ == "__main__":
    main()
