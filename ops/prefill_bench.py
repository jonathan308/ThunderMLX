#!/usr/bin/env python3
"""Prefill throughput probe — long, cache-busting prompts only.

Short prompts (a few hundred tokens) measure fixed dispatch/lockstep overhead,
not prefill: the pipeline prefills in 4096-token chunks with strict rank1->rank0
lockstep, so a 211-token prompt reports ~155 prompt_tps while a 16k prompt on a
healthy cluster historically reports high-300s. The dashboard shows the diluted
number on short traffic — do not trust it (bit us in prior runs).

Each probe: unique UUID header (busts prompt-cache prefix reuse) + filler text
to the target size, max_tokens tiny. Authoritative prompt_tps parsed from the
server release-slot log line; true token count from the tokenizer log line.

Exit: 0 ok, 75 wedge/stream-death (prefill wedges exist), 1 other failure.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime

import requests

SERVER_LOG = "/private/tmp/minimax-m3-cluster-logs/startup.log"
RELEASE_RE = re.compile(
    r"released distributed generation slot \(elapsed=([\d.]+)s, first_token=([\d.]+)s, "
    r"prompt_tps=([\d.]+), tokens=(\d+)")
TOKENIZED_RE = re.compile(r"tokenized prompt -> (\d+) tokens")

FILLER = (
    "The maintenance log for the coastal observatory records wind speed, wave "
    "height, barometric pressure, and the condition of each instrument after "
    "every storm season. Technicians rotate quarterly and each shift appends "
    "calibration notes, anomaly reports, and supply requests to the ledger. "
)


def build_prompt(target_tokens: int) -> str:
    # ~4 chars/token heuristic; unique header busts any cache prefix match
    header = f"Archive review session {uuid.uuid4()}.\n"
    body_chars = target_tokens * 4
    reps = max(1, body_chars // len(FILLER))
    parts = []
    for i in range(reps):
        parts.append(f"[entry {i}-{uuid.uuid4().hex[:8]}] {FILLER}")
    return header + "".join(parts) + "\nIn one short sentence: what is this archive about?"


def scrape_log() -> dict:
    try:
        tail = subprocess.run(["tail", "-60", SERVER_LOG], capture_output=True,
                              text=True, timeout=10).stdout
        out = {}
        for line in reversed(tail.splitlines()):
            if "server_prompt_tps" not in out:
                m = RELEASE_RE.search(line)
                if m:
                    out["server_elapsed_s"] = float(m.group(1))
                    out["server_ttft_s"] = float(m.group(2))
                    out["server_prompt_tps"] = float(m.group(3))
            if "server_prompt_tokens" not in out:
                m = TOKENIZED_RE.search(line)
                if m:
                    out["server_prompt_tokens"] = int(m.group(1))
            if len(out) >= 4:
                break
        return out
    except Exception:
        return {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[16384])
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--json", action="store_true", help="single-line JSON summary on stdout")
    args = p.parse_args()

    results = []
    for size in args.sizes:
        for rep in range(args.reps):
            body = {
                "model": "Minimax-M3-No-Think",
                "messages": [{"role": "user", "content": build_prompt(size)}],
                "max_tokens": 24,
                "temperature": 0.2,
                "stream": True,
            }
            t0 = time.time()
            try:
                with requests.post(args.api, json=body, stream=True, timeout=(15, 240)) as r:
                    r.raise_for_status()
                    for raw in r.iter_lines(decode_unicode=True):
                        if raw and raw.startswith("data:") and raw[5:].strip() == "[DONE]":
                            break
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError) as e:
                print(f"PREFILL WEDGE at size {size}: {e}", flush=True)
                return 75
            except Exception as e:  # noqa: BLE001
                print(f"PREFILL ERROR at size {size}: {e!r}", flush=True)
                return 1
            time.sleep(1.0)
            stats = scrape_log()
            stats.update(requested_tokens=size, rep=rep + 1,
                         wall_s=round(time.time() - t0, 1),
                         ts=datetime.now().isoformat())
            results.append(stats)
            if not args.json:
                print(f"prefill {stats.get('server_prompt_tokens', '?')} tok: "
                      f"{stats.get('server_prompt_tps', '?')} tok/s "
                      f"(ttft {stats.get('server_ttft_s', '?')}s)", flush=True)

    if args.json:
        best = max((r.get("server_prompt_tps", 0) for r in results), default=0)
        print(json.dumps({"prefill_tps_best": best, "probes": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
