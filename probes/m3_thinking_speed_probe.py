#!/usr/bin/env python3
"""MiniMax-M3 thinking-mode speed probe.

Measures client-observed thinking/content timing plus server-side prompt and
decode metrics. The important split is time/tokens before the first visible
content delta, because thinking requests can have good decoder speed while the
answer appears late.
"""
import argparse
import json
import sys
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def request_json(method, path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def reset_cache():
    try:
        return request_json(
            "POST",
            "/admin/prompt-cache/reset",
            {"reason": "thinking speed probe reset", "clear_memory": False},
            timeout=20,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def active_tokens():
    try:
        active = health(timeout=3).get("active_request") or {}
        return int(active.get("tokens_emitted") or 0)
    except Exception:
        return 0


def wait_idle(before_completed, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        if (
            last.get("active_request") is None
            and last.get("requests_completed", 0) > before_completed
        ):
            return last
        time.sleep(0.2)
    raise RuntimeError(f"request did not become idle; last={last}")


def stream_case(name, *, model, prompt, max_tokens, temperature=0, thinking_budget=None):
    before = health()
    before_completed = before.get("requests_completed", 0)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking_budget is not None:
        payload["thinking_budget"] = thinking_budget

    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.time()
    chunks = 0
    reasoning_chunks = 0
    content_chunks = 0
    reasoning_chars = 0
    content_chars = 0
    first_delta_s = None
    first_reasoning_s = None
    first_content_s = None
    tokens_at_first_content = None

    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            evt = json.loads(item)
            chunks += 1
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                content = delta.get("content")
                if (reasoning or content) and first_delta_s is None:
                    first_delta_s = time.time() - started
                if reasoning:
                    reasoning_chunks += 1
                    reasoning_chars += len(reasoning)
                    if first_reasoning_s is None:
                        first_reasoning_s = time.time() - started
                if content:
                    content_chunks += 1
                    content_chars += len(content)
                    if first_content_s is None:
                        first_content_s = time.time() - started
                        tokens_at_first_content = active_tokens()

    final = wait_idle(before_completed)
    last = final.get("last_request") or {}
    total_tokens = int(last.get("tokens") or 0)
    first_content_tokens = tokens_at_first_content or 0
    pre_content_tokens = (
        min(first_content_tokens, total_tokens)
        if first_content_tokens > 0 else None
    )
    content_tokens = (
        max(0, total_tokens - first_content_tokens)
        if first_content_tokens > 0 else None
    )
    content_elapsed = (
        max(0.001, float(last.get("elapsed_s") or 0) - first_content_s)
        if first_content_s is not None else None
    )
    content_tps_est = (
        round(content_tokens / content_elapsed, 2)
        if content_tokens is not None and content_elapsed and content_tokens > 0
        else None
    )
    row = {
        "name": name,
        "model": model,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "chunks": chunks,
        "reasoning_chunks": reasoning_chunks,
        "content_chunks": content_chunks,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "client_elapsed_s": round(time.time() - started, 3),
        "first_delta_s": round(first_delta_s or 0, 3),
        "first_reasoning_s": round(first_reasoning_s or 0, 3),
        "first_content_s": round(first_content_s or 0, 3),
        "tokens_at_first_content": tokens_at_first_content,
        "pre_content_tokens_est": pre_content_tokens,
        "content_tokens_est": content_tokens,
        "content_tps_est": content_tps_est,
        "server_ttft_s": last.get("first_token_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_tokens": total_tokens,
        "server_request_tps": last.get("tps"),
        "server_decode_tps": last.get("decode_tps"),
        "requests_completed": final.get("requests_completed"),
        "requests_failed": final.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--include-budget",
        action="store_true",
        help=(
            "Also test explicit thinking_budget cases. Disabled by default "
            "because budgeted thinking has wedged distributed MiniMax-M3."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Cache/speed probe: think in one short paragraph, then provide "
            "exactly three concise bullets about stable MiniMax-M3 agent "
            "endpoints. Each bullet must be under 18 words."
        ),
    )
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    print(json.dumps({"initial_completed": initial.get("requests_completed"),
                      "initial_failed": initial.get("requests_failed")}), flush=True)
    if not args.keep_cache:
        print(json.dumps({"reset": reset_cache()}, sort_keys=True), flush=True)

    cases = [
        {
            "name": "no_think_baseline",
            "model": "Minimax-M3-No-Think",
            "thinking_budget": None,
        },
        {
            "name": "thinking_default",
            "model": "Minimax-M3",
            "thinking_budget": None,
        },
    ]
    if args.include_budget:
        cases.extend(
            [
                {
                    "name": "thinking_budget_256",
                    "model": "Minimax-M3",
                    "thinking_budget": 256,
                },
                {
                    "name": "thinking_budget_128",
                    "model": "Minimax-M3",
                    "thinking_budget": 128,
                },
            ]
        )
    rows = [
        stream_case(
            case["name"],
            model=case["model"],
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            thinking_budget=case["thinking_budget"],
        )
        for case in cases
    ]
    final = health()
    print(json.dumps({
        "summary": {
            "rows": rows,
            "completed_delta": final.get("requests_completed", 0)
            - initial.get("requests_completed", 0),
            "failed_delta": final.get("requests_failed", 0)
            - initial.get("requests_failed", 0),
        }
    }, sort_keys=True), flush=True)
    if final.get("requests_failed", 0) > initial.get("requests_failed", 0):
        raise SystemExit("thinking probe saw a request failure")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {exc!r}", file=sys.stderr, flush=True)
        raise
