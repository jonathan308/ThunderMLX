#!/usr/bin/env python3
"""Long-decode soak — clean-run replication methodology (HANDOFF-2026-07-05).

N rounds of ~10k-token essays vs http://127.0.0.1:8080, Minimax-M3-No-Think,
temp 0.8, unique prompts, configurable gap between rounds. After each round the
authoritative numbers (tokens, decode_tps) are scraped from the server's
"released distributed generation slot" log line — client-side SSE chunk counts
undercount tokens (server coalesces deltas; verified round 1: 7204 chunks for
10000 real tokens).

Wedge handling: server watchdog force-exits after 180s decode stall
(M3_WATCHDOG_DECODE_TIMEOUT), which on stock MLX orphans wired memory on both
ranks. Client read timeout is 240s so the server-side event always fires first
and we observe it as a dropped stream. First wedge aborts the soak (continuing
against an orphaned cluster is meaningless) and exits 75.

Output: one console line per round (for the monitor) + JSONL in the repo dir.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

SERVER_LOG = Path("/private/tmp/minimax-m3-cluster-logs/startup.log")
READ_TIMEOUT = 240  # server watchdog (180s) fires first; this is the backstop

TOPICS = [
    "the history and future of transoceanic telegraph and fiber cables",
    "how the Antikythera mechanism was built, lost, and understood",
    "the ecology and engineering of beaver dams across North America",
    "the evolution of container shipping and its effect on world trade",
    "the physics and craft of violin making from Cremona to today",
    "the story of the London sewers and the Great Stink of 1858",
    "how weather forecasting evolved from folklore to ensemble models",
    "the rise, fall, and revival of urban cycling infrastructure",
    "the design and operation of the Panama Canal locks",
    "how the metric system spread across the world",
    "the engineering of long-span suspension bridges",
    "the natural and human history of the Great Lakes",
]

RELEASE_RE = re.compile(
    r"released distributed generation slot \(elapsed=([\d.]+)s, first_token=([\d.]+)s, "
    r"prompt_tps=([\d.]+), tokens=(\d+), tps=([\d.]+), decode_tps=([\d.]+)\)"
)


def wired_gb() -> float:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "wired" in line:
                return round(float(line.split()[-1].rstrip(".")) * 16384 / 1e9, 1)
    except Exception:
        pass
    return -1.0


def server_release_stats() -> dict:
    """Last 'released generation slot' line = authoritative round numbers."""
    try:
        tail = subprocess.run(
            ["tail", "-100", str(SERVER_LOG)], capture_output=True, text=True, timeout=10
        ).stdout
        for line in reversed(tail.splitlines()):
            m = RELEASE_RE.search(line)
            if m:
                return {
                    "server_elapsed_s": float(m.group(1)),
                    "server_ttft_s": float(m.group(2)),
                    "server_tokens": int(m.group(4)),
                    "server_decode_tps": float(m.group(6)),
                }
    except Exception:
        pass
    return {}


def run_round(i: int, args, log) -> dict:
    prompt = (
        f"Write a detailed, well-structured essay of roughly 8000 words about "
        f"{TOPICS[i % len(TOPICS)]}. Use section headings, concrete examples, "
        f"dates, and technical detail. Do not stop early; be exhaustive."
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.8,
        "stream": True,
    }
    rec = {"round": i + 1, "topic": TOPICS[i % len(TOPICS)], "start": datetime.now().isoformat()}
    t0 = time.time()
    ttft = None
    chunks = 0
    chars = 0
    finish = None
    status = "ok"
    err = ""
    try:
        with requests.post(args.api, json=body, stream=True, timeout=(15, READ_TIMEOUT)) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {}).get("content") or ""
                    if delta:
                        if ttft is None:
                            ttft = round(time.time() - t0, 1)
                        chunks += 1
                        chars += len(delta)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except requests.exceptions.ReadTimeout:
        status, err = "wedge", f"no tokens for {READ_TIMEOUT}s (client backstop)"
    except requests.exceptions.ChunkedEncodingError as e:
        status, err = "wedge", f"stream dropped mid-decode (server died?): {e}"
    except requests.exceptions.ConnectionError as e:
        status, err = "wedge", f"connection lost: {e}"
    except Exception as e:  # noqa: BLE001
        status, err = "error", repr(e)

    dur = round(time.time() - t0, 1)
    rec.update(
        seconds=dur, ttft_s=ttft, sse_chunks=chunks, chars=chars,
        finish_reason=finish, status=status, error=err, rank0_wired_gb=wired_gb(),
    )
    if status == "ok":
        time.sleep(1.0)  # let the server write its release line
        rec.update(server_release_stats())
    log.write(json.dumps(rec) + "\n")
    log.flush()
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--gap", type=int, default=30, help="seconds between rounds (replication default 30)")
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--model", default="Minimax-M3-No-Think")
    p.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--out-prefix", default="run")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"soak8_{args.out_prefix}_{ts}.jsonl"
    print(f"SOAK START {ts} model={args.model} rounds={args.rounds} max_tokens={args.max_tokens} "
          f"gap={args.gap}s log={path}", flush=True)
    clean = 0
    with open(path, "a") as log:
        for i in range(args.rounds):
            rec = run_round(i, args, log)
            if rec["status"] == "ok":
                clean += 1
                tok = rec.get("server_tokens", rec["sse_chunks"])
                tps = rec.get("server_decode_tps", "?")
                print(f"ROUND {rec['round']}/{args.rounds}: CLEAN {tok} tok in {rec['seconds']}s "
                      f"(decode {tps} t/s, ttft {rec.get('server_ttft_s', rec['ttft_s'])}s, "
                      f"wired {rec['rank0_wired_gb']}GB)", flush=True)
            else:
                print(f"ROUND {rec['round']}/{args.rounds}: {rec['status'].upper()} after "
                      f"{rec['sse_chunks']} chunks / {rec['seconds']}s — {rec['error']} "
                      f"(wired {rec['rank0_wired_gb']}GB)", flush=True)
                print(f"SOAK ABORTED at round {rec['round']}: first {rec['round']-1} rounds clean. "
                      f"Check both ranks for orphaned wired memory before anything else.", flush=True)
                return 75
            if i < args.rounds - 1:
                time.sleep(args.gap)
    print(f"SOAK COMPLETE: {clean}/{args.rounds} clean rounds." +
          (" Production jaccl config REPLICATED." if clean == args.rounds else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
