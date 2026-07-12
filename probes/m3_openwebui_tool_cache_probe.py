#!/usr/bin/env python3
"""OpenWebUI-shaped streaming/cache regression probe.

OpenWebUI and agent clients often attach tool schemas to every chat request,
even when the model only needs to answer normally. This probe verifies that
tool-capable requests still stream live and that the follow-up turn reuses the
hot prompt/KV cache instead of reprocessing the full prompt.
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


def reset_cache(reason):
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": reason, "clear_memory": False},
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
        prepare = pcache.get("last_prepare_event") or {}
        update = pcache.get("last_update_event") or {}
        prepare_at = float(prepare.get("at") or 0)
        update_at = float(update.get("at") or 0)
        prewarm_done = (
            prepare.get("action") != "prewarm_start"
            or (update.get("action") and update_at >= prepare_at)
        )
        if (
            not last.get("active_request")
            and not pcache.get("in_use")
            and int(last.get("requests_completed") or 0) > before_completed
            and prewarm_done
        ):
            return last
        time.sleep(0.2)
    return last or health()


def stream_chat(messages, *, model, max_tokens, tools_count, timeout=300):
    before = int(health().get("requests_completed") or 0)
    payload = {
        "model": model,
        "messages": messages,
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
    first_reasoning_s = None
    first_content_s = None
    chunks = 0
    content_parts = []
    reasoning_parts = []
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
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                content = delta.get("content")
                if (reasoning or content) and first_piece_s is None:
                    first_piece_s = time.time() - started
                if reasoning and first_reasoning_s is None:
                    first_reasoning_s = time.time() - started
                if content and first_content_s is None:
                    first_content_s = time.time() - started
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content:
                    content_parts.append(content)
    final = wait_idle(before)
    last = final.get("last_request") or {}
    shape = last.get("request_shape") or {}
    pcache = final.get("prompt_cache") or {}
    prepare = last.get("prompt_cache_prepare") or pcache.get("last_prepare_event") or {}
    update = pcache.get("last_update_event") or {}
    return {
        "client_elapsed_s": round(time.time() - started, 3),
        "client_first_piece_s": round(first_piece_s or 0.0, 3),
        "client_first_reasoning_s": round(first_reasoning_s or 0.0, 3),
        "client_first_content_s": round(first_content_s or 0.0, 3),
        "chunks": chunks,
        "content": "".join(content_parts),
        "reasoning_chars": len("".join(reasoning_parts)),
        "server_ttft_s": last.get("first_token_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_decode_tps": last.get("decode_tps"),
        "server_request_tps": last.get("tps"),
        "cache_action": prepare.get("action"),
        "cache_prompt_tokens": prepare.get("prompt_tokens"),
        "cache_reuse_tokens": prepare.get("reuse_tokens"),
        "cache_suffix_tokens": prepare.get("suffix_tokens"),
        "cache_reuse_ratio": prepare.get("reuse_ratio"),
        "cache_missed_tokens": prepare.get("missed_tokens"),
        "cache_miss_reason": prepare.get("miss_reason"),
        "cache_session_id": shape.get("cache_session_id"),
        "cache_session_source": shape.get("cache_session_source"),
        "cache_update_action": update.get("action"),
        "cache_update_reason": update.get("reason"),
        "cache_len": pcache.get("cache_len"),
        "failed": final.get("requests_failed"),
    }


def run_case(model, args):
    reset_cache(f"openwebui tool cache probe {model}")
    system = {
        "role": "system",
        "content": (
            "You are a concise local coding assistant. "
            "Reply directly and keep answers short."
        ),
    }
    user1 = {"role": "user", "content": "hello"}
    turn1 = stream_chat(
        [system, user1],
        model=model,
        max_tokens=args.first_max_tokens,
        tools_count=args.tools,
    )
    assistant = {"role": "assistant", "content": turn1["content"] or "Hello!"}
    user2 = {"role": "user", "content": "what did I just say?"}
    turn2 = stream_chat(
        [system, user1, assistant, user2],
        model=model,
        max_tokens=args.followup_max_tokens,
        tools_count=args.tools,
    )
    row = {"model": model, "turn1": turn1, "turn2": turn2}
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--tools", type=int, default=34)
    parser.add_argument("--first-max-tokens", type=int, default=128)
    parser.add_argument("--followup-max-tokens", type=int, default=96)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model id to test. Defaults to both public MiniMax ids.",
    )
    parser.add_argument("--max-hot-suffix", type=int, default=64)
    parser.add_argument("--max-hot-ttft", type=float, default=1.0)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    h = health()
    if h.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {h}")
    models = args.model or ["Minimax-M3-No-Think", "Minimax-M3"]
    failures = []
    failed_before = int(h.get("requests_failed") or 0)
    for model in models:
        row = run_case(model, args)
        turn2 = row["turn2"]
        if (turn2.get("cache_suffix_tokens") or 10**9) > args.max_hot_suffix:
            failures.append(f"{model}: hot suffix too large: {turn2.get('cache_suffix_tokens')}")
        if (turn2.get("server_ttft_s") or 10**9) > args.max_hot_ttft:
            failures.append(f"{model}: hot TTFT too slow: {turn2.get('server_ttft_s')}")
        failed_after = int(turn2.get("failed") or 0)
        if failed_after > failed_before:
            failures.append(
                f"{model}: server failures increased {failed_before}->{failed_after}"
            )
        failed_before = failed_after
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
