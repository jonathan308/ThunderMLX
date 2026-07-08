#!/usr/bin/env python3
"""Sample local/worker memory plus endpoint activity as JSONL.

Useful when a long prefill looks unstable in Activity Monitor. The sampler
does not control the cluster; it only records vm_stat and /health summaries.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request


PAGE_SIZE = 16384


def load_env(root):
    env = os.environ.copy()
    for name in (".env.local", "m3_cluster.env", ".env"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break
    return env


def peer_from_env(env):
    return (
        env.get("M3_PEER")
        or env.get("M3_RANK1_DIRECT_SSH")
        or env.get("M3_RANK1_FALLBACK_SSH")
        or env.get("M3_TAILSCALE_PEER")
        or ""
    )


def parse_vm_stat(text):
    values = {}
    for line in text.splitlines():
        match = re.match(r"([^:]+):\s+([0-9]+)", line.strip().replace(".", ""))
        if match:
            values[match.group(1)] = int(match.group(2))
    wired = values.get("Pages wired down", 0)
    active = values.get("Pages active", 0)
    return {
        "wired_gb": round(wired * PAGE_SIZE / 1024**3, 2),
        "active_gb": round(active * PAGE_SIZE / 1024**3, 2),
        "wired_active_gb": round((wired + active) * PAGE_SIZE / 1024**3, 2),
        "inactive_gb": round(values.get("Pages inactive", 0) * PAGE_SIZE / 1024**3, 2),
        "free_gb": round(values.get("Pages free", 0) * PAGE_SIZE / 1024**3, 2),
        "speculative_gb": round(values.get("Pages speculative", 0) * PAGE_SIZE / 1024**3, 2),
        "compressor_gb": round(
            values.get("Pages occupied by compressor", 0) * PAGE_SIZE / 1024**3,
            2,
        ),
    }


def run_vm_stat(cmd, timeout):
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return parse_vm_stat(proc.stdout)


def health(base, timeout):
    with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    active = data.get("active_request") or {}
    last = data.get("last_request") or {}
    return {
        "status": data.get("status"),
        "completed": data.get("requests_completed"),
        "failed": data.get("requests_failed"),
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
        "last_decode_tps": last.get("decode_tps"),
        "last_cache_efficiency": last.get("cache_efficiency"),
        "last_ok": last.get("ok"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=120, help="Number of samples; 0 runs until interrupted")
    parser.add_argument("--output", default="", help="Optional JSONL output path; defaults to stdout")
    parser.add_argument("--peer", default="")
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument("--ssh-timeout", type=int, default=10)
    parser.add_argument("--stop-when-idle", action="store_true")
    args = parser.parse_args()

    env = load_env(args.root)
    peer = args.peer or peer_from_env(env)
    out = open(args.output, "a", buffering=1) if args.output else sys.stdout
    i = 0
    try:
        while args.samples == 0 or i < args.samples:
            i += 1
            row = {"t": round(time.time(), 3), "peer": peer or None}
            try:
                row["health"] = health(args.base, timeout=5)
            except Exception as exc:
                row["health_error"] = repr(exc)
            try:
                row["rank0"] = run_vm_stat(["vm_stat"], timeout=5)
            except Exception as exc:
                row["rank0_error"] = repr(exc)
            if peer:
                try:
                    row["rank1"] = run_vm_stat(
                        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={args.ssh_timeout}", peer, "vm_stat"],
                        timeout=args.ssh_timeout + 5,
                    )
                except Exception as exc:
                    row["rank1_error"] = repr(exc)
            print(json.dumps(row, sort_keys=True), file=out, flush=True)
            if args.stop_when_idle and not (row.get("health") or {}).get("active"):
                break
            time.sleep(args.interval)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
