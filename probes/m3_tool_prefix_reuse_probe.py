#!/usr/bin/env python3
"""Probe fresh one-turn OpenWebUI/tool-prefix cache reuse.

This differs from a normal multi-turn hot-cache test: every request is shaped
like a new one-turn chat with a different auto session, but the tool schema is
identical. Healthy behavior is one cold tool-prefill followed by tiny suffixes
that reuse the shared MiniMax/tool-template prefix.
"""
import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def request_json(method, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(timeout=5):
    return request_json("GET", "/health", timeout=timeout)


def reset_cache():
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": "tool prefix reuse probe", "clear_memory": False},
        timeout=30,
    )


def dummy_tools(count):
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "No-op compatibility probe tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(count)
    ]


def wait_idle(before_completed, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        pcache = last.get("prompt_cache") or {}
        if (
            not last.get("active_request")
            and not pcache.get("in_use")
            and int(last.get("requests_completed") or 0) > before_completed
        ):
            return last
        time.sleep(0.2)
    return last or health()


def stream_chat(label, text, *, model, tools_count, max_tokens, timeout):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": text},
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "tools": dummy_tools(tools_count),
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_piece_s = None
    chunks = 0
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            chunks += 1
            evt = json.loads(item)
            delta = (evt.get("choices") or [{}])[0].get("delta") or {}
            piece = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if piece and first_piece_s is None:
                first_piece_s = time.time() - started
            if piece:
                parts.append(piece)
    final = wait_idle(before)
    last = final.get("last_request") or {}
    prepare = last.get("prompt_cache_prepare") or (
        (final.get("prompt_cache") or {}).get("last_prepare_event") or {}
    )
    row = {
        "label": label,
        "text": text,
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "client_elapsed_s": round(time.time() - started, 3),
        "chunks": chunks,
        "server_ttft_s": last.get("first_token_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_decode_tps": last.get("decode_tps"),
        "cache_action": prepare.get("action"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_session_id": prepare.get("session_id"),
        "content": "".join(parts),
        "failed": final.get("requests_failed"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--tools", type=int, default=34)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-hot-ttft", type=float, default=1.0)
    parser.add_argument("--max-hot-suffix", type=int, default=64)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    h = health()
    if h.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {h}")
    if not args.keep_cache:
        print(json.dumps({"reset": reset_cache()}, sort_keys=True), flush=True)

    rows = [
        stream_chat("cold_seed", "hello", model=args.model, tools_count=args.tools,
                    max_tokens=args.max_tokens, timeout=args.timeout),
        stream_chat("fresh_hi", "hi", model=args.model, tools_count=args.tools,
                    max_tokens=args.max_tokens, timeout=args.timeout),
        stream_chat("fresh_ok", "say ok", model=args.model, tools_count=args.tools,
                    max_tokens=args.max_tokens, timeout=args.timeout),
        stream_chat("fresh_math", "what is 2+2?", model=args.model, tools_count=args.tools,
                    max_tokens=args.max_tokens, timeout=args.timeout),
    ]

    failures = []
    for row in rows[1:]:
        if (row.get("cache_suffix_tokens") or 10**9) > args.max_hot_suffix:
            failures.append(
                f"{row['label']}: suffix too large: {row.get('cache_suffix_tokens')}"
            )
        if (row.get("server_ttft_s") or 10**9) > args.max_hot_ttft:
            failures.append(
                f"{row['label']}: TTFT too slow: {row.get('server_ttft_s')}"
            )
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
