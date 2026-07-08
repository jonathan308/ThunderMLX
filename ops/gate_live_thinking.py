#!/usr/bin/env python3
"""Gate: live thinking-delta streaming on tool turns (oMLX parity).

PASS requires, on a thinking-mode tool-bearing stream:
  1. >=10 reasoning deltas spread over >=5s (live, not one end-flush)
  2. first reasoning delta lands in the first half of the turn
  3. turn still finishes with well-formed tool_calls (args parse as JSON)
  4. no reasoning field on the final tool_calls delta (no duplication)
  5. stop mid-thinking releases the slot fast on a second request
  6. non-stream tool request returns reasoning_content + tool_calls
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"

WRITE_TOOL = [{
    "type": "function",
    "function": {
        "name": "Write",
        "description": "Write a file to disk",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
}]

PROMPT = ("Create a small Snake game as a single self-contained HTML file "
          "(canvas, arrow keys, score). Write it to "
          "~/Desktop/gate_snake.html using the Write tool.")


def pick_thinking_model():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
        models = [m["id"] for m in json.load(r)["data"]]
    thinking = [m for m in models if "think" in m.lower()
                and "nothink" not in m.lower() and "no-think" not in m.lower()]
    return (thinking or models)[0], models


def stream_request(model, tools, stop_after_reasoning=None):
    """Returns (events, wall) where events = [(t_rel, kind, size), ...]."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": 16384,
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    events = []
    final_delta = None
    t0 = time.time()
    reasoning_seen = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            t = time.time() - t0
            if delta.get("reasoning") or delta.get("reasoning_content"):
                size = len(delta.get("reasoning")
                           or delta.get("reasoning_content"))
                events.append((t, "reasoning", size))
                reasoning_seen += 1
                if stop_after_reasoning and reasoning_seen >= stop_after_reasoning:
                    return events, time.time() - t0, "EARLY_EXIT"
            if delta.get("content"):
                events.append((t, "content", len(delta["content"])))
            if delta.get("tool_calls"):
                events.append((t, "tool_calls", len(json.dumps(delta["tool_calls"]))))
                final_delta = delta
            if ch.get("finish_reason"):
                events.append((t, "finish:" + ch["finish_reason"], 0))
    return events, time.time() - t0, final_delta


def main():
    model, models = pick_thinking_model()
    print(f"models: {models}\nusing thinking model: {model}\n")

    # ---- Gate 1-4: live thinking on a tool turn -------------------------
    print("=== GATE 1: tool-turn stream (killer shape) ===")
    events, wall, final_delta = stream_request(model, WRITE_TOOL)
    r_events = [e for e in events if e[1] == "reasoning"]
    t_calls = [e for e in events if e[1] == "tool_calls"]
    fins = [e for e in events if e[1].startswith("finish")]
    print(f"wall={wall:.1f}s reasoning_deltas={len(r_events)} "
          f"content_deltas={len([e for e in events if e[1]=='content'])} "
          f"tool_call_deltas={len(t_calls)} finish={fins[-1][1] if fins else '?'}")
    if r_events:
        first_r, last_r = r_events[0][0], r_events[-1][0]
        print(f"reasoning window: first@{first_r:.1f}s last@{last_r:.1f}s "
              f"spread={last_r-first_r:.1f}s (turn {wall:.1f}s)")
    ok1 = len(r_events) >= 10
    ok2 = r_events and (r_events[-1][0] - r_events[0][0]) >= 5.0
    ok3 = r_events and r_events[0][0] <= wall * 0.5
    ok4 = bool(t_calls)
    ok5 = final_delta is not None and not final_delta.get("reasoning")
    args_ok = False
    if final_delta:
        try:
            fn = final_delta["tool_calls"][0]["function"]
            args = json.loads(fn["arguments"]) if isinstance(
                fn.get("arguments"), str) else fn.get("arguments", {})
            args_ok = bool(args.get("file_path")) and len(
                args.get("content", "")) > 500
            print(f"tool: {fn['name']} args_chars="
                  f"{len(json.dumps(args))} file={args.get('file_path')}")
        except Exception as exc:
            print(f"tool-arg parse FAILED: {exc}")
    for name, ok in [("live deltas >=10", ok1), ("spread >=5s", ok2),
                     ("first delta in first half", ok3),
                     ("tool_calls delivered", ok4),
                     ("no reasoning on final delta", ok5),
                     ("args parse + content>500", args_ok)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    gate1 = all([ok1, ok2, ok3, ok4, ok5, args_ok])

    # ---- Gate 5: stop mid-thinking on a tool turn -----------------------
    print("\n=== GATE 2: stop mid-thinking (tool turn) ===")
    events, wall, tag = stream_request(model, WRITE_TOOL, stop_after_reasoning=8)
    print(f"got {len([e for e in events if e[1]=='reasoning'])} reasoning "
          f"deltas then disconnected at {wall:.1f}s ({tag})")
    t_stop = time.time()
    urllib.request.urlopen(
        urllib.request.Request(BASE + "/v1/stop", data=b"{}",
                               headers={"Content-Type": "application/json"}),
        timeout=10).read()
    released = False
    for _ in range(30):
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            h = json.load(r)
        if not (h.get("active_request") or {}).get("id"):
            released = True
            break
        time.sleep(1)
    rel_s = time.time() - t_stop
    print(f"  [{'PASS' if released else 'FAIL'}] slot released in {rel_s:.1f}s")
    gate2 = released and rel_s < 20

    # ---- Gate 6: non-stream tool turn carries reasoning ------------------
    print("\n=== GATE 3: non-stream tool turn (reasoning_content present) ===")
    body = {"model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False, "max_tokens": 16384, "tools": WRITE_TOOL}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning"))
    has_calls = bool(msg.get("tool_calls"))
    print(f"reasoning_chars={len(msg.get('reasoning_content') or '')} "
          f"tool_calls={[c['function']['name'] for c in msg.get('tool_calls') or []]}")
    print(f"  [{'PASS' if has_reasoning else 'FAIL'}] reasoning_content present")
    print(f"  [{'PASS' if has_calls else 'FAIL'}] tool_calls present")
    gate3 = has_reasoning and has_calls

    print("\n" + ("GATE OVERALL: PASS" if all([gate1, gate2, gate3])
                  else "GATE OVERALL: FAIL"))
    return 0 if all([gate1, gate2, gate3]) else 1


if __name__ == "__main__":
    sys.exit(main())
