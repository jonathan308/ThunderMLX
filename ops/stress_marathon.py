#!/usr/bin/env python3
"""All-features overnight stress marathon — rotates realistic load shapes
against the cluster until --until, recording a scoreboard. Halt-machines-up on
any anomaly (never reboots anything).

Units per rotation:
  essay      one 10k-token decode round (long-decode regime)
  agent      3 agent cycles (21 tool-call turns; short bursty decodes, cache churn)
  prefill    one 16k cache-busting prefill probe
  shortburst 12 short completions (60-200 tok) with 5-25s think gaps —
             the regime of BOTH real wedges (morning 127-tok + 20:13 93-tok)
  idlegap    30-90s deliberate idle, then the next unit hits a cold-ish link
             (the idle->burst transition is the documented wedge precondition)

Exit: 0 at --until, 75 on wedge (forensics captured first), 1 infra halt.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

REPO = Path.home() / "minimax-m3-cluster"
OPS = REPO / "ops"
API = "http://127.0.0.1:8080/v1/chat/completions"

SHORT_PROMPTS = [
    "Summarize the tradeoffs of TCP vs RDMA transports in 3 sentences.",
    "Write a haiku about distributed inference.",
    "What is 84 * 97? One sentence.",
    "Name three failure modes of distributed systems, one line each.",
    "Explain KV-cache reuse to a new engineer in 2 sentences.",
    "Draft a one-line commit message for a memory-leak fix.",
    "Why do laptops throttle under sustained load? 2 sentences.",
    "Give a two-sentence status update for a healthy cluster.",
]


def log_line(scoreboard, rec):
    rec["ts"] = datetime.now().isoformat()
    with open(scoreboard, "a") as f:
        f.write(json.dumps(rec) + "\n")
    flat = " ".join(f"{k}={v}" for k, v in rec.items() if k not in ("ts",))
    print(f"UNIT {flat}", flush=True)


def wired_gb():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("Pages wired down:"):
                return round(float(line.split()[-1].rstrip(".")) * 16384 / 1e9, 1)
    except Exception:
        pass
    return -1.0


def run_tool(cmd, timeout):
    """Run a soak/agent/probe subprocess; map its exit to unit status."""
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        if p.returncode == 0:
            return "ok", p.stdout[-400:]
        if p.returncode == 75:
            return "wedge", p.stdout[-800:] + p.stderr[-400:]
        return "infra", p.stdout[-400:] + p.stderr[-400:]
    except subprocess.TimeoutExpired:
        return "infra", f"unit timeout after {timeout}s"


def short_burst(n=12):
    """Short decodes with think gaps — the true agent-idle rhythm."""
    for i in range(n):
        body = {
            "model": "Minimax-M3-No-Think",
            "messages": [{"role": "user", "content": random.choice(SHORT_PROMPTS)}],
            "max_tokens": random.choice([60, 100, 160, 200]),
            "temperature": 0.4,
            "stream": True,
        }
        try:
            with requests.post(API, json=body, stream=True, timeout=(15, 240)) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if raw and raw.startswith("data:") and raw[5:].strip() == "[DONE]":
                        break
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            return "wedge", f"short-burst req {i+1}/{n}: {e}"
        except Exception as e:  # noqa: BLE001
            return "infra", f"short-burst req {i+1}/{n}: {e!r}"
        time.sleep(random.uniform(5, 25))
    return "ok", f"{n} short completions"


def halt_wedge(scoreboard, unit, detail):
    print(f"WEDGE during {unit} — capturing forensics, HALTING (no reboot)", flush=True)
    subprocess.run(["zsh", str(OPS / "live_wedge_capture.sh"), f"marathon_{unit}"],
                   timeout=180, check=False)
    attention = OPS / "MORNING_ATTENTION.txt"
    with open(attention, "a") as f:
        f.write(f"\n=== MARATHON WEDGE {datetime.now()} unit={unit} ===\n{detail}\n"
                f"Machines LEFT UP (orphaned if stock behavior). Scoreboard: {scoreboard}\n")
    log_line(scoreboard, {"unit": unit, "status": "wedge", "detail": detail[:300]})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--until", default="04:15", help="stop time HH:MM (next occurrence)")
    args = p.parse_args()

    hh, mm = map(int, args.until.split(":"))
    now = datetime.now()
    until = now.replace(hour=hh, minute=mm, second=0)
    if until <= now:
        until += timedelta(days=1)
    ts = now.strftime("%Y%m%d_%H%M%S")
    scoreboard = REPO / f"stress_marathon_{ts}.jsonl"
    print(f"MARATHON START {ts} until {until} scoreboard={scoreboard}", flush=True)

    random.seed(20260705)
    rotation = ["essay", "agent", "shortburst", "idlegap", "prefill", "shortburst",
                "essay", "idlegap", "agent", "prefill"]
    counts = {"ok": 0, "wedge": 0, "infra": 0}
    infra_streak = 0
    i = 0
    while datetime.now() < until:
        unit = rotation[i % len(rotation)]
        i += 1
        t0 = time.time()
        if unit == "essay":
            status, detail = run_tool(
                ["python3", "long_decode_soak8.py", "--rounds", "1", "--gap", "0",
                 "--out-prefix", f"marathon{i}"], timeout=900)
        elif unit == "agent":
            status, detail = run_tool(
                ["python3", "ops/agent_traffic_test.py", "--cycles", "3",
                 "--out-prefix", f"marathon{i}"], timeout=1200)
        elif unit == "prefill":
            status, detail = run_tool(
                ["python3", "ops/prefill_bench.py", "--sizes", "16384"], timeout=420)
            if status == "infra" and "75" in detail:
                status = "wedge"
        elif unit == "shortburst":
            status, detail = short_burst()
        else:  # idlegap
            gap = random.randint(30, 90)
            time.sleep(gap)
            status, detail = "ok", f"idled {gap}s"

        rec = {"unit": unit, "n": i, "status": status,
               "secs": round(time.time() - t0, 1), "wired0_gb": wired_gb(),
               "detail": detail[-200:].replace("\n", " | ")}
        counts[status] = counts.get(status, 0) + 1

        if status == "wedge":
            halt_wedge(scoreboard, unit, detail)
            return 75
        if status == "infra":
            infra_streak += 1
            log_line(scoreboard, rec)
            if infra_streak >= 3:
                print("3 consecutive infra errors — HALT (machines up)", flush=True)
                return 1
        else:
            infra_streak = 0
            log_line(scoreboard, rec)

    print(f"MARATHON COMPLETE at {datetime.now()}: "
          f"{counts.get('ok',0)} ok / {counts.get('infra',0)} infra / 0 wedges", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
