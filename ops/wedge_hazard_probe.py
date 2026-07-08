#!/usr/bin/env python3
"""Wedge-hazard reproducer — concentrates the correlates of every real wedge
(#14 ~127 tok, #15 93 tok, #16 ~300 tok) instead of diluting them:

  idle gap (30-60s, keepwarm ticking)
    -> new generation whose prompt SHARES A LONG PREFIX with the previous one
       but diverges (forces the auto-session partial-prefix cache REBUILD that
       fired 1ms before wedge #15's freeze — reuse=171/213 signature)
    -> short decode (~800 tok; every real freeze was <300 tokens in)
    -> repeat

~80 hazard cycles/hour vs the marathon's ~8 transition events/hour. If the
correlates are causal, expected time-to-wedge drops ~10x; surviving N hundred
cycles is correspondingly stronger evidence than mixed-load hours.

Same contract as the soaks: exit 0 at --cycles/--until, 75 on wedge (stall or
stream death), JSONL per cycle. Run live_wedge_capture BEFORE any recovery if
this exits 75 (or run under a runner that does).
"""
import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta

import requests

API = "http://127.0.0.1:8080/v1/chat/completions"
READ_TIMEOUT = 240

# ~200-token shared instruction preamble: identical across consecutive cycles
# so the session fingerprint prefix-matches, then the topic line diverges ->
# auto_session_history_prefix_mismatch -> rebuild, every cycle.
PREAMBLE = (
    "You are the duty archivist for a long-running coastal observatory. Your "
    "reports are precise, structured, and written for the permanent record. "
    "Each report must include: current conditions, instrument status, notable "
    "deviations from seasonal norms, maintenance actions taken, supply levels, "
    "personnel notes, and a short forward outlook. Use plain professional "
    "language, avoid speculation beyond the forward outlook section, and keep "
    "measurements in metric units. When historical comparisons are relevant, "
    "reference the archive by season and year. The permanent record values "
    "completeness over brevity, but every sentence must carry information. "
)

TOPICS = [
    "a spring storm that damaged the anemometer mast",
    "the quarterly calibration of the tide gauges",
    "an unexplained pressure drop recorded overnight",
    "the replacement of the backup generator",
    "a visiting research team's equipment integration",
    "sediment buildup affecting the harbor sensors",
    "the annual inventory of spare instruments",
    "a heat wave stressing the cooling systems",
]


def one_cycle(i: int, args, log) -> dict:
    prompt = (PREAMBLE + f"\nToday's report (cycle {i}, ref {uuid.uuid4().hex[:6]}) concerns "
              f"{TOPICS[i % len(TOPICS)]}. Write the full report.")
    body = {"model": "Minimax-M3-No-Think",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens, "temperature": 0.7, "stream": True}
    rec = {"cycle": i, "start": datetime.now().isoformat()}
    t0 = time.time()
    chunks = 0
    status, err = "ok", ""
    try:
        with requests.post(API, json=body, stream=True, timeout=(15, READ_TIMEOUT)) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                if raw[5:].strip() == "[DONE]":
                    break
                chunks += 1
    except requests.exceptions.ReadTimeout:
        status, err = "wedge", f"no tokens for {READ_TIMEOUT}s"
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
        status, err = "wedge", f"stream/connection death: {e}"
    except Exception as e:  # noqa: BLE001
        status, err = "error", repr(e)
    rec.update(status=status, error=err, chunks=chunks, secs=round(time.time() - t0, 1))
    log.write(json.dumps(rec) + "\n")
    log.flush()
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=0, help="0 = run until --until")
    p.add_argument("--until", default="", help="stop time HH:MM (next occurrence)")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--gap-min", type=int, default=30)
    p.add_argument("--gap-max", type=int, default=60)
    p.add_argument("--out-prefix", default="hazard")
    args = p.parse_args()

    stop_at = None
    if args.until:
        hh, mm = map(int, args.until.split(":"))
        stop_at = datetime.now().replace(hour=hh, minute=mm, second=0)
        if stop_at <= datetime.now():
            stop_at += timedelta(days=1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"hazard_{args.out_prefix}_{ts}.jsonl"
    print(f"HAZARD START {ts} cycles={args.cycles or 'until ' + args.until} "
          f"max_tokens={args.max_tokens} gaps={args.gap_min}-{args.gap_max}s log={path}", flush=True)
    random.seed(42)
    i = 0
    with open(path, "a") as log:
        while True:
            i += 1
            if args.cycles and i > args.cycles:
                break
            if stop_at and datetime.now() >= stop_at:
                break
            gap = random.randint(args.gap_min, args.gap_max)
            time.sleep(gap)  # idle first: every real wedge followed an idle gap
            rec = one_cycle(i, args, log)
            if rec["status"] == "wedge":
                print(f"CYCLE {i}: WEDGE after {rec['chunks']} chunks / {rec['secs']}s — {rec['error']}", flush=True)
                print(f"HAZARD WEDGED at cycle {i} ({i-1} clean hazard cycles survived)", flush=True)
                return 75
            if rec["status"] == "error":
                print(f"CYCLE {i}: ERROR {rec['error']}", flush=True)
                return 1
            if i % 10 == 0:
                print(f"CYCLE {i}: clean streak continues ({rec['chunks']} chunks, {rec['secs']}s)", flush=True)
    print(f"HAZARD COMPLETE: {i - 1} hazard cycles, zero wedges.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
