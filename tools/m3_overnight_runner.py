#!/usr/bin/env python3
"""Safe overnight validation runner for ThunderMLX.

This runner is intentionally conservative:
- it refuses to start when wired memory suggests orphaned Metal allocations;
- it uses the existing start/stop/probe scripts instead of a parallel launcher;
- it writes both raw logs and JSONL events for later comparison.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8080"
TOKENS_PER_SYNTH_RECORD = 17.85


def load_env():
    env = os.environ.copy()
    for name in (".env.local", "m3_cluster.env", ".env"):
        path = ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in env:
                env[key] = value
        break
    return env


def run_capture(cmd, *, timeout=30, env=None):
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def vm_stat_wired_gb(cmd, *, timeout=10):
    try:
        p = run_capture(cmd, timeout=timeout)
    except Exception as exc:
        return None, str(exc)
    if p.returncode != 0:
        return None, p.stderr.strip() or p.stdout.strip()
    m = re.search(r"Pages wired down:\s+([0-9]+)", p.stdout)
    if not m:
        return None, "Pages wired down not found"
    return round(int(m.group(1)) * 16384 / 1024**3, 1), None


def resolve_peer(env):
    return (
        env.get("M3_PEER")
        or env.get("M3_RANK1_DIRECT_SSH")
        or env.get("M3_RANK1_FALLBACK_SSH")
        or env.get("M3_TAILSCALE_PEER")
        or ""
    )


def wired_snapshot(env):
    local_gb, local_error = vm_stat_wired_gb(["vm_stat"])
    peer = resolve_peer(env)
    remote_gb = None
    remote_error = None
    if peer:
        remote_gb, remote_error = vm_stat_wired_gb(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", peer, "vm_stat"],
            timeout=15,
        )
    else:
        remote_error = "peer not configured"
    return {
        "rank0_wired_gb": local_gb,
        "rank0_error": local_error,
        "rank1_wired_gb": remote_gb,
        "rank1_error": remote_error,
        "peer": peer or None,
    }


def memory_is_clean(snapshot, rank0_limit, rank1_limit):
    r0 = snapshot.get("rank0_wired_gb")
    r1 = snapshot.get("rank1_wired_gb")
    if r0 is None or r1 is None:
        return False
    return r0 <= rank0_limit and r1 <= rank1_limit


def health(base=DEFAULT_BASE, timeout=5):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "offline", "error": str(exc)}


def compact_health(h):
    active = h.get("active_request") or {}
    last = h.get("last_request") or {}
    defaults = h.get("generation_defaults") or {}
    return {
        "status": h.get("status"),
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "active": bool(active),
        "active_elapsed_s": active.get("elapsed_s"),
        "active_prompt_tokens": active.get("prompt_tokens"),
        "active_prompt_tps": active.get("prompt_tps"),
        "active_tokens": active.get("tokens_emitted"),
        "active_decode_tps": active.get("decode_tps"),
        "active_seconds_since_progress": active.get("seconds_since_progress"),
        "last_full_prompt_tokens": last.get("full_prompt_tokens"),
        "last_processed_prompt_tokens": last.get("processed_prompt_tokens"),
        "last_prompt_tps_excluding_cache": last.get("prompt_tps_excluding_cache"),
        "last_effective_prompt_tps": last.get("effective_prompt_tps"),
        "last_cache_efficiency": last.get("cache_efficiency"),
        "last_ttft_s": last.get("first_token_s"),
        "last_decode_tps": last.get("decode_tps"),
        "last_ok": last.get("ok"),
        "prefill_step_size": defaults.get("effective_prefill_step_size"),
        "mlx_max_mb_per_buffer": defaults.get("mlx_max_mb_per_buffer"),
        "sparse_topk_blocks_override": defaults.get("sparse_topk_blocks_override"),
        "prompt_cache_min_suffix_tokens": defaults.get("effective_prompt_cache_min_suffix_tokens"),
        "prompt_cache_direct_suffix_ids": defaults.get("prompt_cache_direct_suffix_ids"),
        "resident_slots": defaults.get("prompt_cache_resident_slots"),
        "resident_max_total_tokens": defaults.get("prompt_cache_resident_max_total_tokens"),
    }


def records_for_target_tokens(target_tokens):
    return max(1, int(round(target_tokens / TOKENS_PER_SYNTH_RECORD)))


def wait_healthy(base, timeout_s):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = health(base, timeout=5)
        if last.get("status") == "healthy":
            return last
        time.sleep(5)
    return last or {"status": "offline", "error": "health wait timed out"}


def emit(event_file, event):
    event = {"at": round(time.time(), 3), **event}
    print(json.dumps(event, sort_keys=True), flush=True)
    with event_file.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def run_probe(
    name,
    cmd,
    *,
    event_file,
    raw_log,
    timeout,
    env,
    sample_memory=False,
    sampler_interval=5.0,
    sampler_ssh_timeout=10,
    sampler_output=None,
):
    emit(event_file, {"event": "probe_start", "name": name, "cmd": cmd})
    started = time.time()
    sampler = None
    if sample_memory and sampler_output is not None:
        sampler_cmd = [
            sys.executable,
            "tools/m3_memory_sampler.py",
            "--base",
            env.get("M3_ENDPOINT", DEFAULT_BASE),
            "--samples",
            "0",
            "--interval",
            str(sampler_interval),
            "--ssh-timeout",
            str(sampler_ssh_timeout),
            "--output",
            str(sampler_output),
        ]
        sampler = subprocess.Popen(
            sampler_cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        emit(
            event_file,
            {
                "event": "sampler_start",
                "name": name,
                "cmd": sampler_cmd,
                "output": str(sampler_output),
                "pid": sampler.pid,
            },
        )
    with raw_log.open("a") as log:
        log.write(f"\n===== {name} :: {' '.join(cmd)} =====\n")
        log.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            assert p.stdout is not None
            for line in p.stdout:
                log.write(line)
                log.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    parsed = {"line": stripped}
                emit(event_file, {"event": "probe_output", "name": name, "data": parsed})
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            code = p.wait(timeout=15)
            emit(event_file, {"event": "probe_timeout", "name": name, "timeout": timeout})
        except Exception as exc:
            p.terminate()
            code = p.wait(timeout=15)
            emit(event_file, {"event": "probe_exception", "name": name, "error": repr(exc)})
    if sampler is not None:
        sampler.terminate()
        try:
            _, stderr = sampler.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()
            _, stderr = sampler.communicate(timeout=10)
        emit(
            event_file,
            {
                "event": "sampler_done",
                "name": name,
                "returncode": sampler.returncode,
                "stderr_tail": (stderr or "")[-1000:],
                "output": str(sampler_output),
            },
        )
    emit(
        event_file,
        {
            "event": "probe_done",
            "name": name,
            "returncode": code,
            "elapsed_s": round(time.time() - started, 2),
            "health": compact_health(health(env.get("M3_ENDPOINT", DEFAULT_BASE), timeout=5)),
        },
    )
    return code


def start_cluster(event_file, env):
    emit(event_file, {"event": "start_cluster"})
    p = run_capture(["/bin/zsh", str(ROOT / "sync_rank1.sh")], timeout=60, env=env)
    emit(
        event_file,
        {
            "event": "sync_done",
            "returncode": p.returncode,
            "stderr": p.stderr[-1000:],
        },
    )
    if p.returncode != 0:
        return p.returncode
    subprocess.run(
        ["/usr/bin/screen", "-S", "minimax_m3", "-X", "quit"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "/usr/bin/screen",
            "-dmS",
            "minimax_m3",
            "/bin/zsh",
            str(ROOT / "auto_restart.sh"),
        ],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--records", type=int, default=600)
    parser.add_argument(
        "--record-set",
        default="",
        help=(
            "Comma-separated m3_perf_probe record counts to run each cycle. "
            "When omitted, --records is used."
        ),
    )
    parser.add_argument(
        "--token-target-set",
        default="",
        help=(
            "Comma-separated approximate prompt-token targets. Converted to "
            f"synthetic records using {TOKENS_PER_SYNTH_RECORD} tokens/record."
        ),
    )
    parser.add_argument("--sample-memory", action="store_true", help="Capture memory JSONL around each probe")
    parser.add_argument("--sampler-interval", type=float, default=5.0)
    parser.add_argument("--sampler-ssh-timeout", type=int, default=10)
    parser.add_argument("--start", action="store_true", help="Start cluster if /health is offline")
    parser.add_argument("--stop-after", action="store_true", help="Run stop_cluster.sh at the end")
    parser.add_argument("--rank0-wired-limit-gb", type=float, default=None)
    parser.add_argument("--rank1-wired-limit-gb", type=float, default=None)
    parser.add_argument(
        "--skip-memory-preflight",
        action="store_true",
        help=(
            "Do not require low wired memory before running. Use this only when "
            "a healthy model server is already intentionally loaded."
        ),
    )
    parser.add_argument("--health-timeout-s", type=int, default=240)
    args = parser.parse_args()

    env = load_env()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["M3_ENDPOINT"] = args.base.rstrip("/")

    rank0_limit = args.rank0_wired_limit_gb
    if rank0_limit is None:
        rank0_limit = float(env.get("M3_ORPHAN_RANK0_WIRED_GB") or env.get("M3_ORPHAN_STUDIO_WIRED_GB") or 30)
    rank1_limit = args.rank1_wired_limit_gb
    if rank1_limit is None:
        rank1_limit = float(env.get("M3_ORPHAN_RANK1_WIRED_GB") or env.get("M3_ORPHAN_MACBOOK_WIRED_GB") or 20)

    out_dir = ROOT / "overnight_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    event_file = out_dir / "events.jsonl"
    raw_log = out_dir / "raw.log"

    h = health(args.base)
    snapshot = wired_snapshot(env)
    emit(
        event_file,
        {
            "event": "memory_preflight",
            "skipped": bool(args.skip_memory_preflight),
            **snapshot,
        },
    )
    if not args.skip_memory_preflight and not memory_is_clean(snapshot, rank0_limit, rank1_limit):
        emit(
            event_file,
            {
                "event": "refuse_dirty_memory",
                "rank0_limit_gb": rank0_limit,
                "rank1_limit_gb": rank1_limit,
            },
        )
        return 3

    if h.get("status") != "healthy" and args.start:
        code = start_cluster(event_file, env)
        if code != 0:
            return code
        h = wait_healthy(args.base, args.health_timeout_s)
    emit(event_file, {"event": "health_initial", "health": compact_health(h)})
    if h.get("status") != "healthy":
        emit(event_file, {"event": "refuse_unhealthy_endpoint", "health": h})
        return 4

    if args.token_target_set.strip():
        try:
            token_targets = [
                int(item.strip()) for item in args.token_target_set.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            emit(event_file, {"event": "bad_token_target_set", "value": args.token_target_set, "error": str(exc)})
            return 2
        if not token_targets:
            emit(event_file, {"event": "bad_token_target_set", "value": args.token_target_set})
            return 2
        record_counts = [records_for_target_tokens(tokens) for tokens in token_targets]
        emit(
            event_file,
            {
                "event": "token_targets_resolved",
                "targets": token_targets,
                "records": record_counts,
                "tokens_per_record": TOKENS_PER_SYNTH_RECORD,
            },
        )
    elif args.record_set.strip():
        try:
            record_counts = [
                int(item.strip()) for item in args.record_set.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            emit(event_file, {"event": "bad_record_set", "value": args.record_set, "error": str(exc)})
            return 2
        if not record_counts:
            emit(event_file, {"event": "bad_record_set", "value": args.record_set})
            return 2
    else:
        record_counts = [args.records]

    failures = 0
    for cycle in range(1, args.cycles + 1):
        emit(event_file, {"event": "cycle_start", "cycle": cycle})
        failures += int(
            run_probe(
                "hot_cache",
                [sys.executable, "probes/m3_hot_cache_probe.py"],
                event_file=event_file,
                raw_log=raw_log,
                timeout=1800,
                env=env,
                sample_memory=args.sample_memory,
                sampler_interval=args.sampler_interval,
                sampler_ssh_timeout=args.sampler_ssh_timeout,
                sampler_output=out_dir / f"memory_c{cycle}_hot_cache.jsonl",
            )
            != 0
        )
        for records in record_counts:
            failures += int(
                run_probe(
                    f"perf_records_{records}",
                    [
                        sys.executable,
                        "probes/m3_perf_probe.py",
                        "--records",
                        str(records),
                        "--reset-cache",
                        "--session-prefix",
                        f"overnight-c{cycle}-r{records}-{int(time.time())}",
                    ],
                    event_file=event_file,
                    raw_log=raw_log,
                    timeout=7200,
                    env=env,
                    sample_memory=args.sample_memory,
                    sampler_interval=args.sampler_interval,
                    sampler_ssh_timeout=args.sampler_ssh_timeout,
                    sampler_output=out_dir / f"memory_c{cycle}_records_{records}.jsonl",
                )
                != 0
            )
        emit(event_file, {"event": "cycle_done", "cycle": cycle, "health": compact_health(health(args.base))})

    if args.stop_after:
        emit(event_file, {"event": "stop_cluster"})
        p = run_capture(["/bin/zsh", str(ROOT / "stop_cluster.sh")], timeout=1200, env=env)
        emit(event_file, {"event": "stop_done", "returncode": p.returncode, "tail": (p.stdout + p.stderr)[-2000:]})

    emit(event_file, {"event": "memory_final", **wired_snapshot(env)})
    emit(event_file, {"event": "runner_done", "failures": failures, "results_dir": str(out_dir)})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
