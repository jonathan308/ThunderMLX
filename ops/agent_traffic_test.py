#!/usr/bin/env python3
"""Agent-traffic wedge test — the OTHER decode regime.

The 2026-07-05 morning wedge hit a PLAIN SHORT decode (~127 tokens), so long-essay
soaks alone don't cover agent/coding traffic: many short decodes, tool-call turns,
growing multi-turn context with cache reuse, bursty cadence with think-time gaps.
This simulates a Codex/Claude-Code-style session against the OpenAI endpoint.

Cycle (x --cycles): short answer -> tool turn (tools advertised, forced budget)
-> follow-up on grown context -> medium analysis (~1200 tok) -> rapid-fire trio
of short turns. Context accumulates across turns within a session; sessions
rotate every --session-turns to exercise cache eviction/reuse.

Same contract as long_decode_soak8.py: exit 0 all-clean, 75 on wedge; JSONL per
turn; one console line per cycle.
"""
import argparse
import json
import sys
import time
from datetime import datetime

import requests

READ_TIMEOUT = 240

TOOLS = [
    {"type": "function", "function": {
        "name": "exec_command",
        "description": "Run a shell command and return stdout/stderr",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "the command"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
]

SEED_TASKS = [
    "We're debugging a Python web service that intermittently returns 502s behind nginx.",
    "We're profiling a Rust CLI that parses 10GB of JSONL too slowly.",
    "We're adding dark mode to a React dashboard with CSS variables.",
    "We're migrating a Postgres schema without downtime using triggers.",
    "We're fixing a flaky pytest suite that fails only in CI.",
    "We're optimizing an MLX training loop that underutilizes the GPU.",
]


def call(api, messages, max_tokens, temp=0.2, tools=None):
    body = {"model": "Minimax-M3-No-Think", "messages": messages,
            "max_tokens": max_tokens, "temperature": temp, "stream": True}
    if tools:
        body["tools"] = tools
    t0 = time.time()
    chunks, text, finish, tool_calls = 0, [], None, 0
    with requests.post(api, json=body, stream=True, timeout=(15, READ_TIMEOUT)) as r:
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
                d = ch.get("delta", {})
                if d.get("content"):
                    chunks += 1
                    text.append(d["content"])
                if d.get("tool_calls"):
                    tool_calls += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {"seconds": round(time.time() - t0, 1), "chunks": chunks,
            "text": "".join(text), "finish": finish, "tool_chunks": tool_calls}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--session-turns", type=int, default=12, help="rotate conversation after this many turns")
    p.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--think-gap", type=float, default=3.0, help="seconds between turns (agent think time)")
    p.add_argument("--out-prefix", default="agentsim")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"soak8_{args.out_prefix}_{ts}.jsonl"
    print(f"AGENT-TRAFFIC START {ts} cycles={args.cycles} log={path}", flush=True)

    messages = [{"role": "system", "content": "You are a terse, expert coding assistant."}]
    session_turns = 0
    turn_no = 0

    def turn(log, kind, user_msg, max_tokens, tools=None, temp=0.2):
        nonlocal messages, session_turns, turn_no
        turn_no += 1
        messages.append({"role": "user", "content": user_msg})
        rec = {"turn": turn_no, "kind": kind, "ctx_msgs": len(messages),
               "start": datetime.now().isoformat()}
        try:
            r = call(args.api, messages, max_tokens, temp, tools)
            rec.update(status="ok", **{k: v for k, v in r.items() if k != "text"})
            reply = r["text"] or "(tool call)"
            messages.append({"role": "assistant", "content": reply[:2000]})
        except requests.exceptions.ReadTimeout:
            rec.update(status="wedge", error=f"no tokens for {READ_TIMEOUT}s")
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            rec.update(status="wedge", error=f"stream/connection lost: {e}")
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=repr(e))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        session_turns += 1
        if session_turns >= args.session_turns:
            messages[:] = [messages[0]]
            session_turns = 0
        return rec["status"] == "ok"

    with open(path, "a") as log:
        for c in range(args.cycles):
            task = SEED_TASKS[c % len(SEED_TASKS)]
            steps = [
                ("short", f"{task} Where do we start? Two sentences max.", 200, None, 0.2),
                ("tool", "Inspect the relevant config/entrypoint first. Use a tool.", 512, TOOLS, 0.2),
                ("followup", "Tool output: '(3 matching files found, largest is 48KB, modified today)'. What next?", 300, TOOLS, 0.2),
                ("analysis", "Write a detailed step-by-step plan with commands and expected pitfalls.", 1200, None, 0.4),
                ("rapid1", "Shorter version, 3 bullets.", 120, None, 0.2),
                ("rapid2", "Which step is riskiest?", 120, None, 0.2),
                ("rapid3", "One-line summary for the commit message.", 60, None, 0.2),
            ]
            ok_all = True
            for kind, msg, mt, tools, temp in steps:
                if not turn(log, kind, msg, mt, tools, temp):
                    print(f"CYCLE {c+1}/{args.cycles}: WEDGE/ERROR at '{kind}' turn {turn_no} "
                          f"— aborting (see {path})", flush=True)
                    return 75
                time.sleep(args.think_gap)
            print(f"CYCLE {c+1}/{args.cycles}: CLEAN ({len(steps)} turns, ctx {len(messages)} msgs)", flush=True)

    print(f"AGENT-TRAFFIC COMPLETE: {args.cycles}/{args.cycles} cycles clean ({turn_no} turns).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
