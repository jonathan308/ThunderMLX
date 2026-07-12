#!/usr/bin/env python3
"""Repeat the real Claude Code tool matrix while auditing server health.

The live probe executes actual Bash/Read/Write/Edit loops through the Anthropic
gateway. This wrapper alternates models, records one compact JSONL row per run,
and stops at the first CLI failure, server failure, or leaked active request.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
LIVE_PROBE = ROOT / "probes" / "m3_claude_code_live_probe.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_health(url: str, timeout: int = 15) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def compact_health(health: dict[str, Any]) -> dict[str, Any]:
    active = health.get("active_request")
    return {
        "status": health.get("status"),
        "active_request": active.get("id") if isinstance(active, dict) else active,
        "requests_completed": health.get("requests_completed"),
        "requests_failed": health.get("requests_failed"),
        "last_error": health.get("last_error"),
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--health", default="http://127.0.0.1:8080/health")
    parser.add_argument(
        "--models",
        default="Minimax-M3-No-Think,Minimax-M3",
        help="Comma-separated model IDs to alternate.",
    )
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Optional maximum total probe runs; zero means duration only.",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--case", default="all")
    parser.add_argument("--extended", action="store_true", default=True)
    parser.add_argument("--no-extended", action="store_false", dest="extended")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        raise SystemExit("--models must contain at least one model ID")
    if args.duration_hours <= 0 and args.iterations <= 0:
        raise SystemExit("set a positive --duration-hours or --iterations")

    output = args.output or Path(tempfile.gettempdir()) / (
        f"thundermlx_claude_soak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + (args.duration_hours * 3600)
    initial = compact_health(read_health(args.health))
    baseline_failures = int(initial.get("requests_failed") or 0)
    append_row(
        output,
        {
            "event": "start",
            "at": now_iso(),
            "models": models,
            "duration_hours": args.duration_hours,
            "iterations": args.iterations,
            "health": initial,
        },
    )
    print(f"SOAK START output={output} health={initial}", flush=True)

    run_number = 0
    while time.monotonic() < deadline:
        if args.iterations and run_number >= args.iterations:
            break
        model = models[run_number % len(models)]
        run_number += 1
        command = [
            sys.executable,
            str(LIVE_PROBE),
            "--base",
            args.base,
            "--model",
            model,
            "--timeout",
            str(args.timeout),
            "--case",
            args.case,
        ]
        if args.extended:
            command.append("--extended")
        if args.keep_workdir:
            command.append("--keep-workdir")

        run_started = time.monotonic()
        proc = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout * 6,
            check=False,
        )
        elapsed = time.monotonic() - run_started
        health = compact_health(read_health(args.health))
        output_lines = [line.strip() for line in proc.stdout.splitlines()]
        passed = (
            proc.returncode == 0
            and "PASS" in output_lines
            and health.get("status") == "healthy"
            and health.get("active_request") is None
            and int(health.get("requests_failed") or 0) == baseline_failures
        )
        row = {
            "event": "probe",
            "at": now_iso(),
            "run": run_number,
            "model": model,
            "elapsed_seconds": round(elapsed, 3),
            "passed": passed,
            "returncode": proc.returncode,
            "health": health,
            "output_tail": proc.stdout[-4000:],
        }
        append_row(output, row)
        print(
            f"RUN {run_number} model={model} passed={passed} "
            f"elapsed={elapsed:.1f}s completed={health.get('requests_completed')} "
            f"failed={health.get('requests_failed')}",
            flush=True,
        )
        if not passed:
            print(proc.stdout[-4000:], flush=True)
            print(f"SOAK FAILED output={output}", flush=True)
            return 1
        time.sleep(max(0.0, args.pause_seconds))

    final = compact_health(read_health(args.health))
    elapsed = time.monotonic() - started
    append_row(
        output,
        {
            "event": "complete",
            "at": now_iso(),
            "runs": run_number,
            "elapsed_seconds": round(elapsed, 3),
            "health": final,
        },
    )
    print(
        f"SOAK PASS runs={run_number} elapsed={elapsed:.1f}s "
        f"completed={final.get('requests_completed')} failed={final.get('requests_failed')} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
