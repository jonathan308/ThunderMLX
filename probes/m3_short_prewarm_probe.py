#!/usr/bin/env python3
"""Probe visible-transcript prewarm behavior after very short responses.

This targets OpenWebUI/agent turns where the assistant says only a few tokens.
If visible prewarm is skipped, the next visible-history request may depend on
generated-token reuse instead of a canonical visible transcript cache.
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(timeout=8):
    with urllib.request.urlopen(BASE + "/health", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def set_runtime(values):
    return request_json("POST", "/admin/runtime-tuning", {"values": values}, timeout=20)


def reset_cache(reason):
    return request_json(
        "POST",
        "/admin/prompt-cache/reset",
        {"reason": reason, "clear_memory": False},
        timeout=30,
    )


def stream_chat(messages, *, model, max_tokens, session_id, timeout=300):
    before_completed = health().get("requests_completed", 0)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "metadata": {
            "session_id": session_id,
            "source": "m3_short_prewarm_probe",
        },
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
    content_parts = []
    reasoning_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            obj = json.loads(item)
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            piece = content or reasoning
            if piece and first_piece_s is None:
                first_piece_s = time.time() - started
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
            chunks += 1

    deadline = time.time() + 60
    h = health()
    while time.time() < deadline:
        h = health()
        if (
            not h.get("active_request")
            and h.get("requests_completed", 0) > before_completed
        ):
            break
        time.sleep(0.2)
    return {
        "client_elapsed_s": round(time.time() - started, 3),
        "client_ttft_s": round(first_piece_s or 0.0, 3),
        "chunks": chunks,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "health": h,
    }


def summarize(label, result):
    h = result["health"]
    pc = h.get("prompt_cache") or {}
    prepare = pc.get("last_prepare_event") or {}
    update = pc.get("last_update_event") or {}
    visible_prepare = prepare
    if prepare.get("action") == "prewarm_start" and update.get("action"):
        visible_prepare = {**prepare, "action": update.get("action")}
    last = h.get("last_request") or {}
    defaults = h.get("generation_defaults") or {}
    row = {
        "label": label,
        "status": h.get("status"),
        "completed": h.get("requests_completed"),
        "failed": h.get("requests_failed"),
        "prewarm_min_generated": defaults.get(
            "effective_visible_transcript_prewarm_min_generated",
            defaults.get("visible_transcript_prewarm_min_generated"),
        ),
        "server_ttft_s": last.get("first_token_s"),
        "server_prompt_tokens": last.get("prompt_tokens"),
        "server_prompt_tps": last.get("prompt_tps"),
        "server_decode_tps": last.get("decode_tps"),
        "server_output_tokens": last.get("tokens"),
        "cache_action": visible_prepare.get("action"),
        "cache_reuse_tokens": visible_prepare.get("reuse_tokens"),
        "cache_prompt_tokens": visible_prepare.get("prompt_tokens"),
        "cache_suffix_tokens": visible_prepare.get("suffix_tokens"),
        "cache_reuse_ratio": visible_prepare.get("reuse_ratio"),
        "cache_missed_tokens": visible_prepare.get("missed_tokens"),
        "cache_miss_reason": visible_prepare.get("miss_reason"),
        "cache_last_update_action": update.get("action"),
        "cache_last_update_reason": update.get("reason"),
        "cache_key_tokens": pc.get("key_tokens"),
        "cache_len": pc.get("cache_len"),
        "client_ttft_s": result["client_ttft_s"],
        "client_elapsed_s": result["client_elapsed_s"],
        "chunks": result["chunks"],
        "content": result["content"][:200],
        "content_chars": len(result["content"]),
        "reasoning_chars": len(result["reasoning"]),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def run_case(args, min_generated):
    session_id = f"{args.session_prefix}-min{min_generated}"
    print(json.dumps({
        "runtime": set_runtime({"visible_transcript_prewarm_min_generated": min_generated})
    }, sort_keys=True), flush=True)
    print(json.dumps({
        "reset": reset_cache(f"short prewarm probe min={min_generated}")
    }, sort_keys=True), flush=True)

    user0 = {"role": "user", "content": "Reply with exactly this text and nothing else: CACHE-OK"}
    first = stream_chat(
        [user0],
        model=args.model,
        max_tokens=args.first_max_tokens,
        session_id=session_id,
    )
    row1 = summarize(f"turn1_short_min{min_generated}", first)

    assistant0 = {"role": "assistant", "content": first["content"]}
    user1 = {"role": "user", "content": "What exact text did you just reply with?"}
    second = stream_chat(
        [user0, assistant0, user1],
        model=args.model,
        max_tokens=args.followup_max_tokens,
        session_id=session_id,
    )
    row2 = summarize(f"turn2_followup_min{min_generated}", second)
    return row1, row2


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--session-prefix", default=f"short-prewarm-{int(time.time())}")
    parser.add_argument("--first-max-tokens", type=int, default=8)
    parser.add_argument("--followup-max-tokens", type=int, default=48)
    parser.add_argument("--mins", default="16,1", help="Comma-separated min_generated values to test")
    parser.add_argument("--restore", type=int, default=16, help="Runtime min_generated value to restore")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")

    initial = health()
    if initial.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {initial}")
    initial_failed = int(initial.get("requests_failed") or 0)
    print(json.dumps({"initial_failed": initial_failed, "initial_completed": initial.get("requests_completed")}, sort_keys=True), flush=True)

    mins = [int(part.strip()) for part in args.mins.split(",") if part.strip()]
    results = []
    try:
        for value in mins:
            results.extend(run_case(args, value))
    finally:
        print(json.dumps({
            "restore": set_runtime({"visible_transcript_prewarm_min_generated": args.restore})
        }, sort_keys=True), flush=True)

    final = health()
    if int(final.get("requests_failed") or 0) > initial_failed:
        raise SystemExit("short prewarm probe saw request failure")
    print(json.dumps({
        "final_failed": final.get("requests_failed"),
        "final_completed": final.get("requests_completed"),
        "final_prewarm_min": (final.get("generation_defaults") or {}).get(
            "effective_visible_transcript_prewarm_min_generated"
        ),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
