#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"


def health():
    with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
        return json.loads(r.read().decode())


def wait_idle(label, timeout=180):
    start = time.time()
    last = None
    while time.time() - start < timeout:
        last = health()
        if last.get("active_request") is None:
            print(label, "idle", json.dumps(last, sort_keys=True), flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"{label} not idle; last={last}")


def run(label, temperature, *, model, max_tokens, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                chunks += 1
    elapsed = time.time() - started
    print(label, "done", "chunks", chunks, "elapsed", round(elapsed, 2), flush=True)
    wait_idle(label)


def main():
    global BASE
    parser = argparse.ArgumentParser(description="MiniMax-M3 streaming decode smoke probe.")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sample-temperature", type=float, default=0.2)
    parser.add_argument(
        "--prompt",
        default="Write a numbered list of 80 short items about stable APIs.",
    )
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    print("initial", json.dumps(health(), sort_keys=True), flush=True)
    run("deterministic", 0, model=args.model, max_tokens=args.max_tokens, prompt=args.prompt)
    run("sampled", args.sample_temperature, model=args.model, max_tokens=args.max_tokens, prompt=args.prompt)
    print("PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAIL", repr(e), file=sys.stderr, flush=True)
        try:
            print("health", json.dumps(health(), sort_keys=True), file=sys.stderr, flush=True)
        except Exception:
            pass
        raise
